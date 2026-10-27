"""上下文压缩：LLM 摘要生成 + 视图重建（事件化：触发与结果解耦）

- 压缩只做两件事：① 生成摘要 ② 触发方(ChatSession)重建 _messages = System+Human摘要+最近 keep 轮
  早期轮从视图移出——它们已由每轮 MESSAGES_ARCHIVED 增量归档，全量历史可重建
- 触发：COMPRESS_REQUESTED（force=False 自动 / force=True 前端手动提前，可 keep 覆盖保留轮数）
- 结果：发布 CONTEXT_COMPRESSED（summary + compressed_turns + forced），前端可感知手动/自动
- 【KV 前缀一致性】压缩请求 = 当前视图(累积 _messages)去掉最近 keep 轮【原封不动】+ 追加固定 HumanMessage
  压缩指令；与主对话历史前缀一致，可命中前置 KV 缓存
- 摘要消息角色：HumanMessage（additional_kwargs.meta=summary），固定视图第 2 条；只拼一条、迭代覆盖
- 压缩指令文本由 runtime.yaml compression.summarize_prompt 配置注入（None 用内置默认）
- 事件通道：依赖 EventChannel（Port）；只发布 CONTEXT_COMPRESSED（通配，前端/持久化都收），不订阅
"""

from langchain_core.messages import BaseMessage, HumanMessage

from langchain_lab.core.bus.event_bus import EventChannel
from langchain_lab.core.bus.events import Event, EventType
from langchain_lab.core.context.views import build_display_context, build_llm_context, split_turns

_DEFAULT_SUMMARIZE_PROMPT = "请你将以上对话总结为一段简洁的中文摘要，保留关键事实、用户意图、结论、已执行的工具结果与未完成事项，不要遗漏重要信息。"


class SessionCompressor:
    """会话上下文压缩器（每会话实例；输入=当前累积视图 _messages，由 ChatSession 传入）"""

    def __init__(
        self,
        summarize_llm,
        keep_recent_turns: int,
        bus: EventChannel,
        session_id: str,
        retention_budget: int = 160000,  # 保留预算 token（= 模型窗口×(1-headroom)，通常由运行时注入；默认保守大值避免误触发）
        summarize_prompt: str | None = None,
    ):
        self._summarize_llm = summarize_llm
        self._keep_recent_turns = keep_recent_turns
        self._retention_budget = retention_budget
        self._bus = bus
        self._session_id = session_id
        self._summarize_prompt = summarize_prompt or _DEFAULT_SUMMARIZE_PROMPT

    def update_config(self, keep_recent_turns: int, retention_budget: int):
        """提供控制面配置使用"""
        self._keep_recent_turns = keep_recent_turns
        self._retention_budget = retention_budget

    # ---------- 轮次切分（委托 views 纯函数） ----------
    def split_turns(self, messages: list[BaseMessage]) -> list[list[BaseMessage]]:
        return split_turns(messages)

    # ---------- 压缩 ----------
    async def maybe_compress(
        self,
        messages: list[BaseMessage],
        prev_summary: str | None,
        force: bool = False,
        keep_recent_turns: int | None = None,
        actual_tokens: int | None = None,
        retention_budget: int | None = None,
    ) -> str | None:
        """生成摘要，返回新 summary；无可压早期轮返回 None

        - messages：当前累积视图（ChatSession 传入；含旧摘要消息则先被 split_turns 跳过）
        - force=False：自动，保留轮数超下限 或 实际 token 超保留预算 才压
        - force=True ：前端手动提前，跳过阈值判断（可用 keep_turns 覆盖保留轮数，真正提前压）
        - actual_tokens：会话累计实际 token（ChatSession 每轮流式 usage_metadata 累计）；None=无用量信息，退化为纯轮数判断
        - retention_budget：模型窗口 × (1 - headroom)（默认用构造时值）；None 用构造时
        - 压缩请求 = 当前视图去掉最近 keep 轮原封不动 + 追加固定压缩指令（KV 前缀一致）
        - 只返回摘要；视图重建由 ChatSession 调用 build_llm_context 完成
        """
        if self._summarize_llm is None:
            return None
        _recent_turns = max(keep_recent_turns if keep_recent_turns is not None else self._keep_recent_turns, 1)
        request, early_turns = self.build_compress_input(messages, _recent_turns)
        if request is None:
            return None  # 没有早期轮可压（视图消息不足保留数）
        _budget = max(retention_budget if retention_budget is not None else self._retention_budget, 1)
        if not force and len(self.split_turns(messages)) <= self._keep_recent_turns and actual_tokens is not None and actual_tokens <= _budget:
            return None  # 自动模式：保留轮数未超下限 且 实际 token 未超预算 不压
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

    def build_compress_input(self, view_messages: list[BaseMessage], keep_recent_turns: int) -> tuple[list[BaseMessage] | None, int]:
        """压缩请求 = 当前视图去掉最近 keep 轮（原封不动）+ 追加固定 Human 压缩指令

        返回 (request, early_turns)；无可压早期轮返回 (None, 0)
        - view 结构：System + [Human摘要, meta=summary] + 轮*N
        - 头部固定段（system+摘要）与早期轮保留原样、不重排角色、不过滤 system
        - 压缩指令只在压缩请求里作为末尾 HumanMessage，不进 _messages（不污染主对话视图）
        """
        turns = split_turns(view_messages)
        if len(turns) <= keep_recent_turns:
            return None, 0  # 没有早期轮
        early_turns = turns[:-keep_recent_turns]
        early_flat = [m for t in early_turns for m in t]
        turn_ids = {id(m) for t in turns for m in t}
        head = [m for m in view_messages if id(m) not in turn_ids]  # system + 摘要
        request = head + early_flat + [HumanMessage(content=self._summarize_prompt)]
        return request, len(early_turns)

    async def _summarize(self, request: list[BaseMessage]) -> str:
        """原封不动发给压缩 LLM（保持前缀一致）；指令已由调用方作为末尾 HumanMessage 追加"""
        resp = await self._summarize_llm.ainvoke(request)
        return resp.content.strip()

    # ---------- 视图重建（压缩后 ChatSession 调用，改写 _messages） ----------
    def build_llm_context(
        self,
        messages: list[BaseMessage],
        summary: str | None,
        keep_recent_turns: int | None = None,
    ) -> list[BaseMessage]:
        """重建视图 = System + Human摘要 + 最近 keep 轮（早期轮移出，已由归档表保存）

        keep_recent_turns 可覆盖（手动提前压缩传 keep）；None 用构造时默认。
        """
        return build_llm_context(messages, summary, keep_recent_turns or self._keep_recent_turns)

    def build_display_context(self, messages: list[BaseMessage], summary: str | None) -> list[dict]:
        return build_display_context(messages, summary, self._keep_recent_turns)
