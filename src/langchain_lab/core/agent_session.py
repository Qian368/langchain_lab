"""ChatSession：消息总线上的 Agent 会话 actor（面向网络服务化，事件驱动）

- 输入事件化：订阅 USER_MESSAGE_SUBMITTED（终端/http 适配器发布），按 session_id 过滤
- 输出事件化：展示/进度全部 publish（STREAM/TOOL/ASSISTANT/AGENT_TURN/...）
- 副作用订阅者化：不持有 store/view；每轮发布 SESSION_STATE_UPDATED（完整可序列化快照），
  持久化由订阅者（context_store）落库；恢复的历史由入口构造注入，会话不碰数据库
- 分支命令走注入回调（branch_loader/lister/creator），入口用 context_store 实现
- 压缩（可选）：SessionCompressor 注入，LLM 视图 = 摘要置顶 + 最近 N 轮
- 运行上限（max_llm_loop/max_tool_calls）由入口按 runtime.yaml 注入
- 不碰 MCP 连接：MCP 生命周期归 MCPSessionManager，clear_all() 由运行时入口 finally 调用
"""
import json
from typing import cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)
from langchain.tools import BaseTool

from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType
from langchain_lab.core.tools.tool_disclosure import get_disclosure_tools
from langchain_lab.core.tools.tool_middleware import execute_safe_tool_call


class ChatSession:
    """独立会话实例 = 天然隔离；并发多会话互不污染（依赖全部构造注入，无全局单例）"""

    def __init__(
        self,
        session_id: str,
        session_ctx: dict,
        bus: EventBus,
        llm,
        tools: list[BaseTool],  # 全量工具集（入口构建后注入）
        compressor: SessionCompressor | None = None,  # 可选，上下文压缩（投影策略，运行时调用）
        branch_id: str = "root",  # 当前分支
        system_prompt: str = "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务",
        max_llm_loop: int = 3,  # 单轮 LLM 最大迭代（入口按 runtime.yaml 注入）
        max_tool_calls: int = 4,  # 单轮最大工具调用（入口按 runtime.yaml 注入）
        history: list[BaseMessage] | None = None,  # 恢复：入口从 context_store 加载注入（含原 system）
        summary: str | None = None,  # 恢复：会话级压缩摘要
        branch_loader=None,  # async (branch_id)->list|None：加载指定分支历史（入口用 context_store 实现）
        branch_lister=None,  # async ()->list[dict]：列分支
        branch_creator=None,  # async (parent_id, point)->str：建分支返回新 id
    ):
        self._session_id = session_id
        self._session_ctx = session_ctx
        self._bus = bus
        self._llm = llm
        self._tools = tools
        self._compressor = compressor
        self._branch_id = branch_id
        self._max_llm_loop = max_llm_loop
        self._max_tool_calls = max_tool_calls
        # 恢复时注入历史（含 system），否则新建用 system_prompt 打底
        self._messages: list[BaseMessage] = (
            history if history is not None else [SystemMessage(content=system_prompt)]
        )
        self._summary: str | None = summary  # 会话级压缩摘要
        self._branch_loader = branch_loader
        self._branch_lister = branch_lister
        self._branch_creator = branch_creator
        # 订阅输入事件（网络服务化：输入适配器 publish，本会话响应；按 session_id 隔离）
        bus.subscribe(EventType.USER_MESSAGE_SUBMITTED, self._on_user_message)

    # —— 输入事件（async handler，由 event_bus 后台调度）——
    def _on_user_message(self, event: Event):
        """订阅输入：只响应本会话，返回协程由总线后台执行一轮"""
        if event.session_id != self._session_id:
            return None
        content = str((event.payload or {}).get("content", ""))
        return self._handle_turn(content)

    # —— 生命周期 ——
    async def start(self, resumed: bool = False) -> None:
        """会话开始：新建发 SESSION_CREATED；恢复（入口已注入 history）发 SESSION_RESUMED"""
        self._emit(EventType.SESSION_RESUMED if resumed else EventType.SESSION_CREATED)

    async def close(self) -> None:
        """会话关闭：发最终兜底快照 + 关闭事件，flush 等后台落库完成。不碰 MCP 连接"""
        self._emit_session_state()
        self._emit(EventType.SESSION_CLOSED)
        await self._bus.flush()

    # —— 一轮输入处理 ——
    async def _handle_turn(self, content: str) -> None:
        """处理一条输入：分支命令 或 普通消息→Agent 推理；末尾统一发状态快照（事件溯源）"""
        self._emit(EventType.AGENT_TURN_START)
        try:
            handled = False
            if content.startswith(":"):
                handled = await self._handle_command(content)
            if not handled:
                # 普通用户消息：追加 + 回显 + 披露工具 + Agent 循环
                self._messages.append(HumanMessage(content=content))
                self._emit(EventType.HUMAN_MESSAGE_RECEIVED, payload=content)
                visible_tools = await get_disclosure_tools(self._session_ctx, self._messages)
                llm_with_tools = self._llm.bind_tools(tools=visible_tools)
                await self._agent_loop(llm_with_tools=llm_with_tools)
                # 压缩检查：超阈值则把早期轮压缩进摘要（LLM 视图 = 摘要 + 最近 N 轮）
                if self._compressor is not None:
                    new_summary = await self._compressor.maybe_compress(
                        self._messages, self._summary
                    )
                    if new_summary is not None:
                        self._summary = new_summary
            # 命令或消息都会改变状态：发布完整快照，持久化订阅者落库（事件即真相）
            self._emit_session_state()
            self._emit(
                EventType.TURN_COMPLETE,
                payload={"branch_id": self._branch_id, "summary": self._summary},
            )
        except Exception as exc:  # noqa: BLE001 - 面向网络服务化，运行时异常必须抛给订阅者
            self._emit(EventType.ERROR_OCCURRED, payload={"error": repr(exc), "phase": "agent_turn"})
        finally:
            self._emit(EventType.AGENT_TURN_DONE)
            await self._bus.flush()  # 保证持久化订阅者已落库（后台 async handler 完成）

    def _emit_session_state(self) -> None:
        """发布当前会话完整状态快照（可序列化，供持久化订阅者落库 / 前端同步）"""
        self._emit(
            EventType.SESSION_STATE_UPDATED,
            payload={
                "messages": messages_to_dict(self._messages),
                "summary": self._summary,
                "branch_id": self._branch_id,
                "session_ctx": self._session_ctx,
            },
        )

    # —— 分支命令（走注入回调，运行时不碰持久化实现）——
    async def _handle_command(self, raw: str) -> bool:
        """处理 :branch/:branches/:switch；命中返回 True，否则 False（当作普通输入）"""
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == ":branches":
            if self._branch_lister is None:
                self._notice("无上下文存储，无法列出分支")
                return True
            branches = await self._branch_lister()
            if not branches:
                self._notice("暂无分支")
                return True
            lines = [
                f"  {b['branch_id']}  parent={b['parent_id'] or '-'}  point={b['branch_point_seq']}"
                + (" *" if b["branch_id"] == self._branch_id else "")
                for b in branches
            ]
            self._notice("分支列表：\n" + "\n".join(lines))
            return True

        if cmd == ":branch":
            if not arg.isdigit() or int(arg) < 1:
                self._notice("用法：:branch <N>  (N=对话消息序号，从1开始，不含系统提示)")
                return True
            n = int(arg)
            if n >= len(self._messages):
                self._notice(f"序号 {n} 超出当前消息数（{len(self._messages) - 1} 条对话）")
                return True
            if self._branch_creator is None:
                self._notice("无上下文存储，无法创建分支")
                return True
            new_branch = await self._branch_creator(self._branch_id, n)
            # 快照式：子分支初始 = 父分支拐点前（含系统提示 + 前 N 条对话）
            self._messages = self._messages[: n + 1]
            self._branch_id = new_branch
            self._notice(f"已创建分支 {new_branch}（拐点 seq={n}），后续输入进入该分支")
            return True

        if cmd == ":switch":
            if not arg:
                self._notice("用法：:switch <branch_id>")
                return True
            if self._branch_loader is None:
                self._notice("无上下文存储，无法切换分支")
                return True
            restored = await self._branch_loader(arg)
            if restored is None:
                self._notice(f"分支 {arg} 不存在或不属于本会话")
                return True
            self._branch_id = arg
            self._messages = restored
            self._notice(f"已切换到分支 {arg}（{len(restored) - 1} 条对话）")
            return True

        return False

    # —— 内部 ——
    def _notice(self, text: str) -> None:
        """会话内系统提示 → SESSION_NOTICE 事件（展示层渲染，网络化可路由到前端）"""
        self._emit(EventType.SESSION_NOTICE, payload=text)

    @property
    def _llm_messages(self) -> list[BaseMessage]:
        """LLM 视图：压缩后 = 摘要置顶 + 最近 N 轮真实消息；未压缩 = 全量（规范 7.2）"""
        if self._compressor is not None and self._summary:
            return self._compressor.build_llm_context(self._messages, self._summary)
        return self._messages

    def _emit(self, event_type: EventType, payload=None) -> None:
        """统一发布事件，session_id 自动填充（多会话隔离基础）"""
        self._bus.publish(Event(type=event_type, session_id=self._session_id, payload=payload))

    async def _agent_loop(self, llm_with_tools) -> None:
        """单次对话循环（搬 exp3 _single_chat_loop，展示全部 emit）"""
        tool_call_count = 0
        llm_loop_count = 0

        while True:
            # —— 上限注入：加在 current_messages 副本，不影响真实 messages ——
            injected_llm_limit_msg: SystemMessage | None = None
            injected_tool_limit_msg: SystemMessage | None = None

            if llm_loop_count >= self._max_llm_loop:  # 先判断，再自增，用 >=
                injected_llm_limit_msg = SystemMessage(
                    content="本轮对话已到达最大Agent轮次上限，请基于现有全部信息直接给出最终答案，禁止调用任何工具。",
                    additional_kwargs={"meta": "llm_loop_break"},
                )
            if tool_call_count >= self._max_tool_calls:  # 先判断，再自增，用 >=
                injected_tool_limit_msg = SystemMessage(
                    content="本轮对话工具调用次数已耗尽，不再执行后续工具，请基于已获得的工具结果总结回答，禁止继续调用工具",
                    additional_kwargs={"meta": "tool_call_break"},
                )

            # 基底用 LLM 视图（压缩后含摘要+最近轮），上限注入加在副本
            current_messages = self._llm_messages.copy()
            if injected_llm_limit_msg is not None:
                current_messages.append(injected_llm_limit_msg)
            if injected_tool_limit_msg is not None:
                current_messages.append(injected_tool_limit_msg)

            llm_loop_count += 1

            result = llm_with_tools.stream(current_messages)

            # —— 流式：逐片 emit 给展示；累积留在实例，完成后 emit 完整消息 ——
            full_chunk: AIMessage | AIMessageChunk | None = None
            for chunk in result:
                full_chunk = chunk if full_chunk is None else cast(AIMessage | AIMessageChunk, full_chunk + chunk)
                if chunk.content:
                    self._emit(EventType.STREAM_CHUNK, payload=chunk.content)

            if not full_chunk:
                break

            self._messages.append(full_chunk)
            self._emit(EventType.ASSISTANT_MESSAGE_EMITTED, payload=full_chunk)

            # —— 上限终止：LLM 收尾轮多一次自增，工具计数顶格停在上限 ——
            if llm_loop_count > self._max_llm_loop or tool_call_count >= self._max_tool_calls:
                break

            # —— 需要调用工具：逐个执行，拼 ToolMessage；完成后回到循环顶部再带工具调 LLM ——
            if full_chunk.tool_calls:
                for i, tc in enumerate(full_chunk.tool_calls, 1):
                    if tool_call_count >= self._max_tool_calls:
                        break
                    tool_call_count += 1
                    self._emit(
                        EventType.TOOL_CALL_START,
                        payload={"name": tc["name"], "id": tc["id"], "args": tc["args"], "index": i},
                    )
                    result = await execute_safe_tool_call(
                        tool_call=tc, tools=self._tools, session_ctx=self._session_ctx, tool_index=i
                    )
                    self._messages.append(
                        ToolMessage(
                            content=json.dumps(result["content"], ensure_ascii=False),
                            tool_call_id=result["tool_call_id"],
                            name=result["name"],
                        )
                    )
                    self._emit(EventType.TOOL_RESULT, payload=result)
                continue
            else:
                # 无工具调用，模型输出最终答案，退出 Agent 循环
                break
