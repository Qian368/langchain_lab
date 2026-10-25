"""ChatSession：消息总线上的 Agent 会话 actor（面向网络服务化，事件驱动）

- 输入事件化：定向订阅 USER_MESSAGE_SUBMITTED / COMPRESS_REQUESTED / SESSION_COMMAND（session_id 定向，O(1) 投递）
- 不解析用户输入：指令的识别与合法性校验由客户端/适配器完成，转成语义化事件后发布；运行时只响应事件
- 控制钩子（AgentHooks）：pre_user_message / post_llm / post_tool 三个同步可干预埋点，默认 None
- 单源消息：运行时只维护一份 _messages（即当前视图）；每轮新消息 append 累积，
  压缩时由 build_llm_context 改写 _messages = System+Human摘要+最近 keep 轮（早期轮移出，已归档）
- 每轮增量归档：_handle_turn 末尾发 MESSAGES_ARCHIVED（本轮新增真实消息，事件自动带 event_id），ContextStore append 落库
- 压缩摘要归档：压缩完成后发 SUMMARY_ARCHIVED（携带当前 branch_id），摘要消息按分支落 archive kind=summary
- 输出事件化：展示/进度全部 publish（STREAM/TOOL/ASSISTANT/AGENT_TURN/...）
- 持久化单一真相源：不持有 store；每轮只发归档事件；恢复视图由入口从 archive 构建注入（KV 逐 token 一致）
- 压缩后置事件化：每轮结束发 COMPRESS_REQUESTED(force=False) 自动触发，前端可手动提前(force=True)
- 分支：store 从 archive 构建拐点视图（System + 前 point 轮真实消息，支持压缩后从历史轮次分叉），
  由 branch_creator 回调返回 (new_id, point_view)；运行时不碰持久化实现
- 删除/重命名：走注入回调（branch_deleter/session_renamer/branch_renamer），运行时只响应语义化事件
- 重跑（regenerate）：只允许当前最后一条人类消息对应轮。复用该轮已有工具结果（不重放副作用），
  条件化注入 meta='rerun' SystemMessage（仅当该轮有工具调用；加在副本，不持久化/不归档/不恢复），
  重新生成回答并增量归档；旧回答发 MESSAGE_SUPERSEDED 由 store 软删（前端只看到新回答）
- 运行上限（max_llm_loop/max_tool_calls）由入口按 runtime.yaml 注入；不碰 MCP 连接
- 事件通道抽象：依赖 EventChannel（Port），不感知底层是内存总线还是 MQ
"""
import asyncio
import inspect
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

from langchain_lab.core.agent_hooks import AgentHooks
from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.event_bus import EventChannel
from langchain_lab.core.events import Event, EventType
from langchain_lab.core.tools.tool_disclosure import get_disclosure_tools
from langchain_lab.core.tools.tool_middleware import execute_safe_tool_call


class ChatSession:
    """独立会话实例 = 天然隔离；并发多会话互不污染（依赖全部构造注入，无全局单例）"""

    def __init__(
        self,
        session_id: str,
        session_ctx: dict,
        bus: EventChannel,
        llm,
        tools: list[BaseTool],  # 全量工具集（入口构建后注入）
        compressor: SessionCompressor | None = None,  # 可选，上下文压缩
        hooks: AgentHooks | None = None,  # 控制型钩子（校验/改写/拒绝；默认 None 不侵入）
        branch_id: str = "root",  # 当前分支
        system_prompt: str = "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务",
        max_llm_loop: int = 3,  # 单轮 LLM 最大迭代（入口按 runtime.yaml 注入）
        max_tool_calls: int = 4,  # 单轮最大工具调用（入口按 runtime.yaml 注入）
        history: list[BaseMessage] | None = None,  # 恢复：入口从 archive 构建的 llm 视图注入（含 system/摘要），无则新建
        summary: str | None = None,  # 恢复：会话级压缩摘要
        branch_loader=None,  # async (branch_id)->list|None：加载指定分支视图（入口用 context_store 实现）
        branch_lister=None,  # async ()->list[dict]：列分支
        branch_creator=None,  # async (parent_id, point)->(new_id, point_view)：建分支，返回新 id 与拐点视图
        branch_deleter=None,  # async (branch_id)->list[str]：删除分支（级联软删子树），返回被删 id 列表
        session_renamer=None,  # async (name)->None：重命名会话（当前分支为 root 时）
        branch_renamer=None,  # async (branch_id, name)->None：重命名分支
    ):
        self._session_id = session_id
        self._session_ctx = session_ctx
        self._bus = bus
        self._llm = llm
        self._tools = tools
        self._compressor = compressor
        self._hooks = hooks or AgentHooks()  # 空钩子集，无侵入
        self._branch_id = branch_id
        self._max_llm_loop = max_llm_loop
        self._max_tool_calls = max_tool_calls
        # 恢复时注入视图历史（含 system/摘要），否则新建用 system_prompt 打底
        self._messages: list[BaseMessage] = (
            history if history is not None else [SystemMessage(content=system_prompt)]
        )
        self._summary: str | None = summary  # 会话级压缩摘要
        self._branch_loader = branch_loader
        self._branch_lister = branch_lister
        self._branch_creator = branch_creator
        self._branch_deleter = branch_deleter
        self._session_renamer = session_renamer
        self._branch_renamer = branch_renamer
        self._compress_lock = asyncio.Lock()  # 压缩串行化（自动/手动并发不竞态）
        # 订阅输入事件（定向订阅自己的 session_id，O(1) 投递，不被其他会话事件触发）
        bus.subscribe(self._on_user_message, EventType.USER_MESSAGE_SUBMITTED, session_id=self._session_id)
        bus.subscribe(self._on_compress_requested, EventType.COMPRESS_REQUESTED, session_id=self._session_id)
        bus.subscribe(self._on_command, EventType.SESSION_COMMAND, session_id=self._session_id)

    # —— 控制钩子执行器（统一返回值语义）——
    @staticmethod
    async def _run_hook(hook, arg):
        """执行钩子：None=放行原样；非None=替换；async 自动 await；异常冒泡=拒绝"""
        if hook is None:
            return arg
        result = hook(arg)
        if inspect.isawaitable(result):
            result = await result
        return arg if result is None else result

    # —— 输入事件（async handler，由通道后台调度；定向订阅已保证是本站会话事件）——
    def _on_user_message(self, event: Event):
        """订阅普通用户消息（适配器已确认非指令）：返回协程由通道后台执行一轮"""
        content = str((event.payload or {}).get("content", ""))
        return self._handle_turn(content)

    def _on_compress_requested(self, event: Event):
        """订阅压缩触发：自动(force=False)或前端手动(force=True) → 异步生成摘要 + 重建视图"""
        if self._compressor is None:
            return None
        return self._do_compress(event)

    async def _do_compress(self, event: Event) -> None:
        """串行压缩：生成摘要 + 改写 _messages = 摘要+最近 keep 轮 + 摘要消息按当前分支归档"""
        payload = event.payload or {}
        force = bool(payload.get("force", False))
        keep = payload.get("keep")
        async with self._compress_lock:
            new_summary = await self._compressor.maybe_compress(
                self._messages, self._summary, force=force, keep_turns=keep
            )
            if new_summary is not None:
                self._summary = new_summary
                # 视图重建：早期轮从 _messages 移出（已由 MESSAGES_ARCHIVED 增量归档），摘要置顶
                self._messages = self._compressor.build_llm_context(self._messages, new_summary)
                # 摘要消息按当前分支归档（store 订阅 SUMMARY_ARCHIVED 落 archive kind=summary，event_id 幂等）
                self._emit(
                    EventType.SUMMARY_ARCHIVED,
                    payload={"branch_id": self._branch_id, "summary": new_summary},
                )
                # 视图变化立即同步（前端）
                self._emit_session_state()

    def _on_command(self, event: Event):
        """订阅控制指令事件（适配器已解析+校验，运行时只按 action 响应）

        regenerate 为独立路径（自带 turn 边界/归档/作废/flush），不从 _handle_command_event 的通用 try 走，
        避免与通用控制指令的双重 turn 边界与 flush。
        """
        payload = event.payload or {}
        if payload.get("action") == "regenerate":
            return self._do_regenerate()
        return self._handle_command_event(event)

    async def _handle_command_event(self, event: Event) -> None:
        """按 action 处理分支/删除/重命名等控制指令；不解析原始文本"""
        self._emit(EventType.AGENT_TURN_START)
        payload = event.payload or {}
        action = payload.get("action")
        try:
            if action == "branch_list":
                await self._do_branch_list()
            elif action == "branch_create":
                await self._do_branch_create(payload.get("point"))
            elif action == "branch_switch":
                await self._do_branch_switch(payload.get("branch_id"))
            elif action == "branch_delete":
                await self._do_branch_delete(payload.get("branch_id"))
            elif action == "rename":
                await self._do_rename(payload.get("name"))
            else:
                self._notice(f"未知指令 action={action}")
            self._emit_session_state()
        except Exception as exc:  # noqa: BLE001
            self._emit(EventType.ERROR_OCCURRED, payload={"error": repr(exc), "phase": "command"})
        finally:
            self._emit(EventType.AGENT_TURN_DONE)
            await self._bus.flush()

    # —— 分支操作（走注入回调，运行时不碰持久化实现）——
    async def _do_branch_list(self) -> None:
        if self._branch_lister is None:
            self._notice("无上下文存储，无法列出分支")
            return
        branches = await self._branch_lister()
        if not branches:
            self._notice("暂无分支")
            return
        lines = [
            f"  {b['branch_id']}  parent={b['parent_id'] or '-'}  point={b['branch_point_seq']}  {b.get('name','')}"
            + (" *" if b["branch_id"] == self._branch_id else "")
            for b in branches
        ]
        self._notice("分支列表：\n" + "\n".join(lines))

    async def _do_branch_create(self, point) -> None:
        """创建子分支：store 从 archive 构建拐点视图（System + 前 point 轮真实消息），返回 (new_id, point_view)

        支持压缩后从历史轮次分叉：拐点视图基于 archive 无损全量真实消息，而非当前视图快照切片（
        压缩后早期轮已移出视图，旧实现 _messages[:point+1] 在拐点早于压缩点时静默错误）。
        """
        if not isinstance(point, int) or point < 1:
            self._notice("用法：branch_create 需 point>=1（对话消息序号）")
            return
        if self._branch_creator is None:
            self._notice("无上下文存储，无法创建分支")
            return
        try:
            new_branch, point_view = await self._branch_creator(self._branch_id, point)
        except ValueError as exc:
            self._notice(str(exc))  # 拐点越界等，交给 store 校验后提示
            return
        self._messages = point_view
        self._branch_id = new_branch
        self._notice(f"已创建分支 {new_branch}（拐点 seq={point}），后续输入进入该分支")

    async def _do_branch_switch(self, branch_id) -> None:
        if not branch_id:
            self._notice("用法：branch_switch 需 branch_id")
            return
        if self._branch_loader is None:
            self._notice("无上下文存储，无法切换分支")
            return
        try:
            restored = await self._branch_loader(branch_id)
        except ValueError as exc:
            self._notice(str(exc))
            return
        if restored is None:
            self._notice(f"分支 {branch_id} 不存在或不属于本会话")
            return
        self._branch_id = branch_id
        self._messages = restored
        self._notice(f"已切换到分支 {branch_id}（{len(restored) - 1} 条对话）")

    async def _do_branch_delete(self, branch_id) -> None:
        """删除分支（级联软删子树）。删 root=删整个会话（发 SESSION_DELETED，入口退出）；删当前分支则切回根分支"""
        if not branch_id:
            self._notice("用法：branch_delete 需 branch_id（root=删除整个会话）")
            return
        if self._branch_deleter is None:
            self._notice("无上下文存储，无法删除分支")
            return
        try:
            deleted = await self._branch_deleter(branch_id)
        except ValueError as exc:
            self._notice(str(exc))
            return
        self._notice(f"已删除分支 {branch_id}（含 {len(deleted) - 1} 个后代分支）")
        if self._branch_id not in deleted:
            return
        # 当前分支被删：root 被删=删整个会话；否则切回根分支
        if branch_id == "root" or "root" in deleted:
            self._notice("根分支（整个会话）已删除，本次会话结束")
            self._emit(EventType.SESSION_DELETED, payload={"branch_id": branch_id})
        else:
            self._branch_id = "root"
            try:
                restored = await self._branch_loader("root")
                self._messages = restored if restored else [SystemMessage(content=self._messages[0].content)]
                self._notice("当前分支已删除，已切回根分支")
            except Exception:  # noqa: BLE001
                self._messages = [SystemMessage(content=self._messages[0].content)]
                self._notice("当前分支已删除，已切回根分支（视图重建失败，重置为新会话）")

    async def _do_rename(self, name) -> None:
        """重命名：当前分支为 root → 会话名；否则分支名"""
        if not name or not name.strip():
            self._notice("用法：rename 需非空名字")
            return
        name = name.strip()
        if self._branch_id == "root":
            if self._session_renamer is None:
                self._notice("无上下文存储，无法重命名会话")
                return
            await self._session_renamer(name)
            self._notice(f"会话已重命名为「{name}」")
        else:
            if self._branch_renamer is None:
                self._notice("无上下文存储，无法重命名分支")
                return
            await self._branch_renamer(self._branch_id, name)
            self._notice(f"分支 {self._branch_id} 已重命名为「{name}」")

    # —— 重跑：只允许当前最后一条人类消息对应轮，复用工具结果，条件化注解，新回答归档、旧回答作废 ——
    async def _do_regenerate(self) -> None:
        """重跑当前最后一条人类消息对应轮：

        - 复用该轮已有工具结果（ToolMessage 留在上下文作为已发生事实，不重放副作用）
        - 条件化注解：仅当该轮有工具调用时，注入 meta='rerun' SystemMessage（临时加在副本，
          不持久化、不进 _messages、不归档、不恢复）；无工具调用不注入
        - 新回答 append _messages 并增量归档；旧回答发 MESSAGE_SUPERSEDED 由 store 软删（前端只看到新回答）
        - 自带 turn 边界/归档/作废/flush，独立于 _handle_command_event（避免双重 turn 边界）
        """
        human_idxs = [
            i
            for i, m in enumerate(self._messages)
            if isinstance(m, HumanMessage) and m.additional_kwargs.get("meta") != "summary"
        ]
        if not human_idxs:
            self._notice("没有可重跑的轮次（暂无用户消息）")
            return
        h_idx = human_idxs[-1]  # 只允许最后一条人类消息对应轮
        tail = self._messages[h_idx + 1 :]
        tool_results = [m for m in tail if isinstance(m, ToolMessage)]
        old_answer = [m for m in tail if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)]
        # 复用该轮工具结果、裁掉旧回答 → 重跑基底（h_n + tool_results），旧回答不入视图
        base_len = h_idx + 1 + len(tool_results)
        self._messages = self._messages[: h_idx + 1] + tool_results
        # 条件化注解：仅该轮有工具调用时注入（副本临时，不持久化）
        extra: list[SystemMessage] | None = None
        if tool_results:
            extra = [
                SystemMessage(
                    content=(
                        "这是一次对上一轮回答的重跑。该轮已执行的工具调用及其结果已在上文上下文中。"
                        "请优先基于已有工具结果给出新的最终回答；若确实需要新的外部信息，才调用工具，"
                        "避免重复调用已经执行过且结果仍有效的工具。"
                    ),
                    additional_kwargs={"meta": "rerun"},
                )
            ]
        self._emit(EventType.AGENT_TURN_START)
        try:
            visible_tools = await get_disclosure_tools(self._session_ctx, self._messages)
            llm_with_tools = self._llm.bind_tools(tools=visible_tools)
            await self._agent_loop(llm_with_tools=llm_with_tools, injected_extra=extra)
            # 增量归档：本轮重跑新增的真实消息（base 之后的新回答；已归档的 h_n/复用工具结果不重复落库）
            self._archive_new_messages(base_len)
            # 作废旧回答：store 订阅 MESSAGE_SUPERSEDED 软删 archive 中该分支最后一条匹配的 AI 消息
            for m in old_answer:
                self._emit(
                    EventType.MESSAGE_SUPERSEDED,
                    payload={"branch_id": self._branch_id, "superseded_content": m.content},
                )
            self._emit_session_state()
            self._emit(
                EventType.TURN_COMPLETE,
                payload={"branch_id": self._branch_id, "summary": self._summary, "regenerated": True},
            )
        except Exception as exc:  # noqa: BLE001 - 面向网络服务化，运行时异常必须抛给订阅者
            self._emit(EventType.ERROR_OCCURRED, payload={"error": repr(exc), "phase": "regenerate"})
        finally:
            self._emit(EventType.AGENT_TURN_DONE)
            # 后置自动压缩触发（异步，不阻塞对话）
            self._emit(EventType.COMPRESS_REQUESTED, payload={"force": False})
            await self._bus.flush()

    # —— 生命周期 ——
    async def start(self, resumed: bool = False) -> None:
        """会话开始：新建发 SESSION_CREATED；恢复（入口已注入 history）发 SESSION_RESUMED"""
        self._emit(EventType.SESSION_RESUMED if resumed else EventType.SESSION_CREATED)

    async def close(self) -> None:
        """会话关闭：发关闭事件，flush 等后台落库完成。不碰 MCP 连接"""
        self._emit(EventType.SESSION_CLOSED)
        await self._bus.flush()

    # —— 一轮普通消息处理 ——
    async def _handle_turn(self, content: str) -> None:
        """处理一条普通用户消息（适配器已确认非指令）→ Agent 推理；末尾发增量归档 + 视图同步"""
        self._emit(EventType.AGENT_TURN_START)
        start_idx = len(self._messages)  # 本轮开始前的消息数，用于增量归档
        try:
            # 控制钩子① pre_user_message：输入合法性校验/脱敏/改写（None=原样，str=替换，raise=拒绝）
            content = await self._run_hook(self._hooks.pre_user_message, content)
            self._messages.append(HumanMessage(content=content))
            self._emit(EventType.HUMAN_MESSAGE_RECEIVED, payload=content)
            visible_tools = await get_disclosure_tools(self._session_ctx, self._messages)
            llm_with_tools = self._llm.bind_tools(tools=visible_tools)
            await self._agent_loop(llm_with_tools=llm_with_tools)
            # 每轮增量归档：本轮新增真实消息（排除 system/摘要），append 落库（事件自动带 event_id 幂等）
            self._archive_new_messages(start_idx)
            # 视图同步（前端展示），持久化依赖归档事件（单一真相源）
            self._emit_session_state()
            self._emit(
                EventType.TURN_COMPLETE,
                payload={"branch_id": self._branch_id, "summary": self._summary},
            )
        except Exception as exc:  # noqa: BLE001 - 面向网络服务化，运行时异常必须抛给订阅者
            self._emit(EventType.ERROR_OCCURRED, payload={"error": repr(exc), "phase": "agent_turn"})
        finally:
            self._emit(EventType.AGENT_TURN_DONE)
            # 后置自动压缩触发（异步，不阻塞对话；压缩完成经 CONTEXT_COMPRESSED 更新 _summary）
            self._emit(EventType.COMPRESS_REQUESTED, payload={"force": False})
            await self._bus.flush()

    def _archive_new_messages(self, start_idx: int) -> None:
        """增量归档本轮新增真实消息（human/ai/tool，排除 system 与 meta=summary 摘要），append 落库"""
        new_msgs = self._messages[start_idx:]
        real = [
            messages_to_dict(m)
            for m in new_msgs
            if not (isinstance(m, SystemMessage) or m.additional_kwargs.get("meta") == "summary")
        ]
        if real:
            self._emit(
                EventType.MESSAGES_ARCHIVED,
                payload={"messages": real, "branch_id": self._branch_id},
            )

    def _emit_session_state(self) -> None:
        """发布当前视图完整快照（可序列化，供前端同步 / 日志订阅）"""
        self._emit(
            EventType.SESSION_STATE_UPDATED,
            payload={
                "messages": messages_to_dict(self._messages),
                "summary": self._summary,
                "branch_id": self._branch_id,
                "session_ctx": self._session_ctx,
            },
        )

    # —— 内部 ——
    def _notice(self, text: str) -> None:
        """会话内系统提示 → SESSION_NOTICE 事件（展示层渲染，网络化可路由到前端）"""
        self._emit(EventType.SESSION_NOTICE, payload=text)

    @property
    def _llm_messages(self) -> list[BaseMessage]:
        """LLM 视图：单源，直接返回当前 _messages（即视图本身，不再每次投影过滤）"""
        return self._messages

    def _emit(self, event_type: EventType, payload=None) -> None:
        """统一发布事件，session_id 自动填充（多会话隔离基础）；event_id 由 Event 自动生成（幂等去重）"""
        self._bus.publish(Event(type=event_type, session_id=self._session_id, payload=payload))

    async def _agent_loop(self, llm_with_tools, injected_extra: list[BaseMessage] | None = None) -> None:
        """单次对话循环（搬 exp3 _single_chat_loop，展示全部 emit）

        injected_extra：重跑等场景的临时注入消息（如 meta='rerun' 注解），加在副本、不进 _messages，
        不持久化、不归档、不恢复；仅本轮 LLM 可见。
        """
        tool_call_count = 0
        llm_loop_count = 0

        while True:
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

            # 基底用当前视图（_messages 即视图），上限注入 + 临时注入（重跑注解）都加在副本
            current_messages = self._messages.copy()
            if injected_llm_limit_msg is not None:
                current_messages.append(injected_llm_limit_msg)
            if injected_tool_limit_msg is not None:
                current_messages.append(injected_tool_limit_msg)
            if injected_extra:
                current_messages.extend(injected_extra)

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

            # —— 控制钩子② post_llm：LLM 合法输出校验/替换（None=原样，非None=替换，raise=拒绝）——
            full_chunk = await self._run_hook(self._hooks.post_llm, full_chunk)
            self._messages.append(full_chunk)
            # 完整消息事件：payload 用 messages_to_dict（JSON 可序列化，MQ 契约）；展示层看流式+状态，不依赖此 payload 内容
            self._emit(EventType.ASSISTANT_MESSAGE_EMITTED, payload=messages_to_dict(full_chunk))

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
                    # —— 控制钩子③ post_tool：工具执行结果校验/脱敏/替换（None=原样，dict=替换，raise=拒绝）——
                    result = await self._run_hook(self._hooks.post_tool, result)
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
