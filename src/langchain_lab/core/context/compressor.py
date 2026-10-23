"""上下文压缩：LLM 摘要生成 + 双视图投影委托（事件化：触发与结果解耦）

- 数据模型保留全量，双视图是投影（规范 7.2，委托 views 纯函数）
- 摘要存会话级 sessions.summary（各分支共享，规范 7.1）
- 触发：COMPRESS_REQUESTED（force=False 自动 / force=True 前端手动提前，可 keep 覆盖保留轮数）
- 结果：发布 CONTEXT_COMPRESSED（summary + compressed_turns + forced），前端可感知手动/自动
- 压缩由 ChatSession 后置触发（不阻塞主对话），本组件只负责生成摘要 + 发结果事件
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_lab.core.context.views import build_display_context, build_llm_context, split_turns
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


class SessionCompressor:
    """会话上下文压缩器（每会话实例，投影策略由运行时调用/事件驱动）"""

    def __init__(
        self,
        summarize_llm,
        keep_recent_turns: int,
        bus: EventBus,
        session_id: str,
    ):
        self._summarize_llm = summarize_llm
        self._keep_recent_turns = keep_recent_turns
        self._bus = bus
        self._session_id = session_id

    # ---------- 轮次切分（委托 views 纯函数） ----------
    def split_turns(self, messages: list[BaseMessage]) -> list[list[BaseMessage]]:
        return split_turns(messages)

    # ---------- 压缩 ----------
    async def maybe_compress(
        self,
        messages: list[BaseMessage],
        prev_summary: str | None,
        force: bool = False,
        keep_turns: int | None = None,
    ) -> str | None:
        """生成摘要，返回新 summary；无可压早期轮返回 None

        - force=False：自动，turns 超过 keep_recent_turns 才压
        - force=True ：前端手动提前，跳过阈值判断（可用 keep_turns 覆盖保留轮数，真正提前压）
        - early = turns[:-keep]，保留最近 keep 轮（默认 keep_recent_turns）
        """
        turns = self.split_turns(messages)
        keep = keep_turns if keep_turns is not None else self._keep_recent_turns
        keep = max(keep, 1)
        early_turns = turns[:-keep]
        if not early_turns:
            return None  # 没有早期轮可压（消息不足保留数）
        if not force and len(turns) <= self._keep_recent_turns:
            return None  # 自动模式：未超阈值不压
        new_summary = await self._summarize(early_turns, prev_summary)
        self._bus.publish(
            Event(
                type=EventType.CONTEXT_COMPRESSED,
                session_id=self._session_id,
                payload={
                    "summary": new_summary,
                    "compressed_turns": len(early_turns),
                    "forced": force,
                },
            )
        )
        return new_summary

    async def _summarize(
        self, early_turns: list[list[BaseMessage]], prev_summary: str | None
    ) -> str:
        prompt = self._build_summary_prompt(early_turns, prev_summary)
        resp = await self._summarize_llm.ainvoke([HumanMessage(content=prompt)])
        return resp.content.strip()

    @staticmethod
    def _build_summary_prompt(
        early_turns: list[list[BaseMessage]], prev_summary: str | None
    ) -> str:
        lines = [
            "你是会话摘要器。将以下早期对话压缩为要点摘要，保留关键事实、用户偏好、已确认结论与未完成事项。用中文，简洁分点。"
        ]
        if prev_summary:
            lines.append(f"\n[已有摘要]\n{prev_summary}")
        lines.append("\n[早期对话]")
        for turn in early_turns:
            for m in turn:
                role = (
                    "user"
                    if isinstance(m, HumanMessage)
                    else "assistant"
                    if isinstance(m, AIMessage)
                    else "tool"
                )
                content = m.content if isinstance(m.content, str) else str(m.content)
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ---------- 双视图投影（委托 views，keep_recent_turns 注入） ----------
    def build_llm_context(
        self, messages: list[BaseMessage], summary: str | None
    ) -> list[BaseMessage]:
        return build_llm_context(messages, summary, self._keep_recent_turns)

    def build_display_context(
        self, messages: list[BaseMessage], summary: str | None
    ) -> list[dict]:
        return build_display_context(messages, summary, self._keep_recent_turns)
