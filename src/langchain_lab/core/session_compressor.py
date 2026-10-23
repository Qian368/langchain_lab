"""上下文压缩：LLM 摘要生成 + 双视图投影

- 数据模型保留全量，双视图是投影（规范 7.2，不重复存储）
  - LLM 视图：System + 摘要置顶 + 最近 keep_recent_turns 轮真实消息
  - 前端视图：真实消息 + 早期折叠占位
- 摘要存会话级 sessions.summary（各分支共享，规范 7.1）
- R1/R3：每会话一个实例，依赖全部构造注入
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


class SessionCompressor:
    """会话上下文压缩器（每会话实例）"""

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

    # ---------- 轮次切分 ----------
    @staticmethod
    def split_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
        """按 HumanMessage 起始切分轮次（跳过开头 system prompt）"""
        turns: list[list[BaseMessage]] = []
        cur: list[BaseMessage] = []
        for m in messages[1:]:
            if isinstance(m, HumanMessage) and cur:
                turns.append(cur)
                cur = []
            cur.append(m)
        if cur:
            turns.append(cur)
        return turns

    # ---------- 压缩 ----------
    async def maybe_compress(
        self, messages: list[BaseMessage], prev_summary: str | None
    ) -> str | None:
        """超阈值则压缩早期轮生成摘要，返回新 summary；未超返回 None"""
        turns = self.split_turns(messages)
        if len(turns) <= self._keep_recent_turns:
            return None
        early_turns = turns[: -self._keep_recent_turns]
        new_summary = await self._summarize(early_turns, prev_summary)
        self._bus.publish(
            Event(
                type=EventType.CONTEXT_COMPRESSED,
                session_id=self._session_id,
                payload={"summary": new_summary, "compressed_turns": len(early_turns)},
            )
        )
        return new_summary

    async def _summarize(self, early_turns: list[list[BaseMessage]], prev_summary: str | None) -> str:
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

    # ---------- 双视图投影 ----------
    def build_llm_context(self, messages: list[BaseMessage], summary: str | None) -> list[BaseMessage]:
        """LLM 视图：System + 摘要置顶 + 最近 N 轮真实消息（未压缩=全量）"""
        turns = self.split_turns(messages)
        head = list(messages[:1])  # system prompt
        if summary:
            head.append(SystemMessage(content=f"[早期上下文摘要]\n{summary}"))
        recent = [m for turn in turns[-self._keep_recent_turns :] for m in turn]
        return head + recent

    def build_display_context(
        self, messages: list[BaseMessage], summary: str | None
    ) -> list[dict]:
        """前端视图：真实消息 + 早期折叠占位（过滤摘要，不含 summary 本身）"""
        turns = self.split_turns(messages)
        if not summary or len(turns) <= self._keep_recent_turns:
            return [{"kind": "message", "data": m} for m in messages]
        early = [m for turn in turns[: -self._keep_recent_turns] for m in turn]
        recent = [m for turn in turns[-self._keep_recent_turns :] for m in turn]
        result: list[dict] = [
            {"kind": "collapse", "collapsed_count": len(early), "summary": summary}
        ]
        result += [{"kind": "message", "data": m} for m in recent]
        return result
