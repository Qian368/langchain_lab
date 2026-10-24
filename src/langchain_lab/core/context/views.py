"""双视图投影（纯函数）：轮次切分 / LLM 视图 / 前端视图

- 数据模型保留全量，双视图是投影（规范 7.2，不重复存储）
- LLM 视图：System + 摘要(HumanMessage, meta=summary) 固定第 2 条 + 最近 keep_recent_turns 轮真实消息
- 前端视图：真实消息 + 早期折叠占位（过滤摘要，不含 summary 本身）
- 摘要消息带 additional_kwargs.meta="summary"，split_turns 跳过、不当作真实轮（迭代覆盖时视图也只拼一条）
- 不持久化、不依赖总线，纯函数；运行时通过 compressor 委托调用
"""
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def split_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """按 HumanMessage 起始切分轮次（跳过开头 system prompt 与摘要消息 meta=summary）"""
    turns: list[list[BaseMessage]] = []
    cur: list[BaseMessage] = []
    for m in messages[1:]:
        if isinstance(m, SystemMessage):
            continue  # 多段 system（如压缩摘要置顶）不打断轮次
        if m.additional_kwargs.get("meta") == "summary":
            continue  # 摘要消息（HumanMessage 置顶第 2 条）不计为真实轮、不打断
        if isinstance(m, HumanMessage) and cur:
            turns.append(cur)
            cur = []
        cur.append(m)
    if cur:
        turns.append(cur)
    return turns


def build_llm_context(
    messages: list[BaseMessage], summary: str | None, keep_recent_turns: int
) -> list[BaseMessage]:
    """LLM 视图：System + 摘要(Human) 固定第 2 条 + 最近 N 轮真实消息（未压缩=全量）

    摘要只拼一条、不堆叠多轮；每次迭代用新摘要覆盖更新，视图前缀形态稳定（KV 缓存友好）。
    """
    head = list(messages[:1])  # system prompt
    if summary:
        head.append(
            HumanMessage(
                content=f"[对话历史摘要]\n{summary}", additional_kwargs={"meta": "summary"}
            )
        )
    if summary:
        # 有摘要：跳过被压缩的早期轮，只取最近 N 轮
        turns = split_turns(messages)
        recent = [m for turn in turns[-keep_recent_turns:] for m in turn]
        return head + recent
    return messages  # 未压缩：直接全量


def build_display_context(
    messages: list[BaseMessage], summary: str | None, keep_recent_turns: int
) -> list[dict]:
    """前端视图：真实消息 + 早期折叠占位（过滤摘要，不含 summary 本身）"""
    turns = split_turns(messages)
    if not summary or len(turns) <= keep_recent_turns:
        return [{"kind": "message", "data": m} for m in messages]
    early = [m for turn in turns[:-keep_recent_turns] for m in turn]
    recent = [m for turn in turns[-keep_recent_turns:] for m in turn]
    return [{"kind": "collapse", "collapsed_count": len(early), "summary": summary}] + [
        {"kind": "message", "data": m} for m in recent
    ]
