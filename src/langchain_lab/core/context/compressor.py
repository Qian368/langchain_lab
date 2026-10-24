"""上下文压缩：LLM 摘要生成 + 双视图投影委托（事件化：触发与结果解耦）

- 数据模型保留全量，双视图是投影（规范 7.2，委托 views 纯函数）
- 摘要存会话级 sessions.summary（各分支共享，规范 7.1）
- 触发：COMPRESS_REQUESTED（force=False 自动 / force=True 前端手动提前，可 keep 覆盖保留轮数）
- 结果：发布 CONTEXT_COMPRESSED（summary + compressed_turns + forced），前端可感知手动/自动
- 压缩由 ChatSession 后置触发（不阻塞主对话），本组件只负责生成摘要 + 发结果事件
- 【KV 前缀一致性】压缩请求 = LLM 视图去掉最近 keep 轮后【原封不动】（保留 system/摘要/各角色，
  不过滤不重排）+ 尾部追加一条固定 HumanMessage 压缩指令；与主对话历史前缀一致，可命中前置 KV 缓存
- 摘要消息角色：HumanMessage（additional_kwargs.meta=summary），固定视图第 2 条；只拼一条、迭代覆盖，不堆叠
- 压缩指令文本由 runtime.yaml compression.summarize_prompt 配置注入（None 用内置默认）
"""
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_lab.core.context.views import build_display_context, build_llm_context, split_turns
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType

_DEFAULT_SUMMARIZE_PROMPT = (
    "请你将以上对话总结为一段简洁的中文摘要，保留关键事实、用户意图、结论、"
    "已执行的工具结果与未完成事项，不要遗漏重要信息。"
)


class SessionCompressor:
    """会话上下文压缩器（每会话实例，投影策略由运行时调用/事件驱动）"""

    def __init__(
        self,
        summarize_llm,
        keep_recent_turns: int,
        bus: EventBus,
        session_id: str,
        summarize_prompt: str | None = None,
    ):
        self._summarize_llm = summarize_llm
        self._keep_recent_turns = keep_recent_turns
        self._bus = bus
        self._session_id = session_id
        self._summarize_prompt = summarize_prompt or _DEFAULT_SUMMARIZE_PROMPT

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
        - 压缩输入 = 视图去掉最近 keep 轮原封不动 + 追加固定压缩指令（KV 前缀一致）
        """
        if self._summarize_llm is None:
            return None
        keep = max(keep_turns if keep_turns is not None else self._keep_recent_turns, 1)
        view = self.build_llm_context(messages, prev_summary)
        request, early_turns = self.build_compress_input(view, keep)
        if request is None:
            return None  # 没有早期轮可压（消息不足保留数）
        if not force and len(split_turns(view)) <= self._keep_recent_turns:
            return None  # 自动模式：未超阈值不压
        new_summary = await self._summarize(request)
        self._bus.publish(
            Event(
                type=EventType.CONTEXT_COMPRESSED,
                session_id=self._session_id,
                payload={
                    "summary": new_summary,
                    "compressed_turns": early_turns,
                    "forced": force,
                },
            )
        )
        return new_summary

    def build_compress_input(
        self, view_messages: list[BaseMessage], keep_turns: int
    ) -> tuple[list[BaseMessage] | None, int]:
        """压缩请求 = 视图去掉最近 keep 轮（原封不动）+ 追加固定 Human 压缩指令

        返回 (request, early_turns)；无可压早期轮返回 (None, 0)
        - view 结构：System + [Human摘要, meta=summary] + 轮*N
        - 头部固定段（system+摘要）与早期轮保留原样、不重排角色、不过滤 system
        - 压缩指令只在压缩请求里作为末尾 HumanMessage，不进 _messages（不污染主对话视图）
        """
        turns = split_turns(view_messages)
        if len(turns) <= keep_turns:
            return None, 0  # 没有早期轮
        early_turns = turns[:-keep_turns]
        early_flat = [m for t in early_turns for m in t]
        turn_ids = {id(m) for t in turns for m in t}
        head = [m for m in view_messages if id(m) not in turn_ids]  # system + 摘要
        request = head + early_flat + [HumanMessage(content=self._summarize_prompt)]
        return request, len(early_turns)

    async def _summarize(self, request: list[BaseMessage]) -> str:
        """原封不动发给压缩 LLM（保持前缀一致）；指令已由调用方作为末尾 HumanMessage 追加"""
        resp = await self._summarize_llm.ainvoke(request)
        return resp.content.strip()

    # ---------- 双视图投影（委托 views，keep_recent_turns 注入） ----------
    def build_llm_context(
        self, messages: list[BaseMessage], summary: str | None
    ) -> list[BaseMessage]:
        return build_llm_context(messages, summary, self._keep_recent_turns)

    def build_display_context(
        self, messages: list[BaseMessage], summary: str | None
    ) -> list[dict]:
        return build_display_context(messages, summary, self._keep_recent_turns)
