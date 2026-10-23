"""ChatSession：事件驱动的 Agent 会话（事件源，一个会话一个实例）

- 搬自 exp3_multi_chat 的循环逻辑，业务/展示/状态解耦：
  - 业务累积留在实例（_messages 全量，压缩后不删）
  - 展示全部 emit 事件，由订阅者（TerminalView 等）渲染
  - 输入交互通过注入的 view.prompt_async()（交互层归 view）
- 压缩（可选）：SessionCompressor 注入，LLM 视图 = 摘要置顶 + 最近 N 轮（_llm_messages），
  _messages 保持全量供持久化/前端折叠
- 不碰 MCP 连接：MCP 连接生命周期归 MCPSessionManager，clear_all() 由运行时入口 finally 调用
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
)
from langchain.tools import BaseTool

from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType
from langchain_lab.core.session_compressor import SessionCompressor
from langchain_lab.core.tools.tool_disclosure import get_disclosure_tools
from langchain_lab.core.tools.tool_middleware import execute_safe_tool_call

MAX_LLM_LOOP = 3
MAX_TOOL_CALLS = 4


class ChatSession:
    """独立会话实例 = 天然隔离；并发多会话互不污染（R1/R3：依赖全部构造注入，无全局单例）"""

    def __init__(
        self,
        session_id: str,
        session_ctx: dict,
        bus: EventBus,
        llm,
        view,  # TerminalView：提供 prompt_async 输入 + 订阅事件渲染（duck-typing）
        tools: list[BaseTool],  # 全量工具集（入口构建后注入）
        store=None,  # SessionStore：可选，持久化（每轮 save + close 兜底）
        compressor: SessionCompressor | None = None,  # 可选，上下文压缩
        system_prompt: str = "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务",
    ):
        self._session_id = session_id
        self._session_ctx = session_ctx
        self._bus = bus
        self._llm = llm
        self._view = view
        self._tools = tools
        self._store = store
        self._compressor = compressor
        self._messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._summary: str | None = None  # 会话级压缩摘要（存 sessions.summary）

    # —— 生命周期 ——
    async def resume(self) -> None:
        """从持久化恢复（store 提供 load），无持久化/无记录则保持新建状态"""
        if self._store is not None:
            restored = await self._store.load(self._session_id)
            if restored is not None:
                self._messages = restored
                self._summary = await self._store.load_summary(self._session_id)

    async def run(self) -> None:
        """主循环：输入 → 披露工具 → Agent 循环（搬 exp3 main + _single_chat_loop）"""
        self._emit(EventType.SESSION_CREATED)

        while True:
            human_prompt = await self._view.prompt_async("请向你的助手发送消息(shift + enter 换行)：\n> ")
            if human_prompt == "0":
                break
            self._messages.append(HumanMessage(content=human_prompt))
            self._emit(EventType.HUMAN_MESSAGE_RECEIVED, payload=human_prompt)

            visible_tools = await get_disclosure_tools(self._session_ctx, self._messages)
            llm_with_tools = self._llm.bind_tools(tools=visible_tools)

            await self._agent_loop(llm_with_tools=llm_with_tools)

            # 压缩检查：超阈值则把早期轮压缩进摘要（LLM 视图 = 摘要 + 最近 N 轮）
            if self._compressor is not None:
                new_summary = await self._compressor.maybe_compress(self._messages, self._summary)
                if new_summary is not None:
                    self._summary = new_summary

            # 每轮结束落库（崩溃安全，带摘要）；持久化归 store，MCP 连接归 MCPSessionManager
            if self._store is not None:
                await self._store.save(
                    self._session_id, self._messages, session_ctx=self._session_ctx, summary=self._summary
                )

    async def close(self) -> None:
        """会话关闭：保证落库（flush 等后台写库）+ 保存兜底 + 事件。不碰 MCP 连接"""
        self._emit(EventType.SESSION_CLOSED)
        await self._bus.flush()  # 等后台 async handler（若有）完成
        if self._store is not None:
            await self._store.save(
                self._session_id, self._messages, session_ctx=self._session_ctx, summary=self._summary
            )

    # —— 内部 ——
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
        """单次对话循环（搬 exp3 _single_chat_loop，printer.*/print() 全部换成 emit）"""
        tool_call_count = 0
        llm_loop_count = 0

        while True:
            # —— 上限注入：加在 current_messages 副本，不影响真实 messages（天然不进持久化）——
            injected_llm_limit_msg: SystemMessage | None = None
            injected_tool_limit_msg: SystemMessage | None = None

            if llm_loop_count >= MAX_LLM_LOOP:  # 先判断，再自增，用 >=
                injected_llm_limit_msg = SystemMessage(
                    content="本轮对话已到达最大Agent轮次上限，请基于现有全部信息直接给出最终答案，禁止调用任何工具。",
                    additional_kwargs={"meta": "llm_loop_break"},
                )
            if tool_call_count >= MAX_TOOL_CALLS:  # 先判断，再自增，用 >=
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

            # —— 流式：逐片 emit 给展示；累积留在实例，完成后 emit 完整消息给持久化 ——
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
            if llm_loop_count > MAX_LLM_LOOP or tool_call_count >= MAX_TOOL_CALLS:
                break

            # —— 需要调用工具：逐个执行，拼 ToolMessage；完成后回到循环顶部再带工具调 LLM ——
            if full_chunk.tool_calls:
                for i, tc in enumerate(full_chunk.tool_calls, 1):
                    if tool_call_count >= MAX_TOOL_CALLS:
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
