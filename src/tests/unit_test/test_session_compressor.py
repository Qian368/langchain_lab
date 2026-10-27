"""SessionCompressor 单测：轮次切分 / 摘要生成 / 双视图投影 / 手动提前压缩（不连真实 LLM）

覆盖 KV 前缀一致性：压缩请求 = 视图去掉最近 keep 轮【原封不动】+ 追加固定 Human 指令。
"""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_lab.core.bus.event_bus import EventBus
from langchain_lab.core.bus.events import EventType
from langchain_lab.core.context.compressor import SessionCompressor


class _FakeLlm:
    """假摘要 LLM：返回固定摘要"""

    async def ainvoke(self, messages):
        return SimpleNamespace(content="压缩摘要")


def _msgs_with_turns(n: int):
    """构造 system + n 轮(每轮 human+ai)消息"""
    msgs = [SystemMessage(content="sys")]
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    return msgs


def _comp(**kw):
    return SessionCompressor(None, keep_recent_turns=kw.get("k", 6), bus=EventBus(), session_id="s1")


def test_split_turns():
    msgs = _msgs_with_turns(2)
    comp = _comp(k=1)
    turns = comp.split_turns(msgs)
    assert len(turns) == 2
    assert turns[0][0].content == "h0"
    assert turns[1][0].content == "h1"


def test_split_turns_skips_summary_message():
    """摘要消息（HumanMessage meta=summary）不当作真实轮"""
    msgs = _msgs_with_turns(2)
    view = [msgs[0], HumanMessage(content="[对话历史摘要]\n摘要", additional_kwargs={"meta": "summary"}), *msgs[1:]]
    comp = _comp(k=1)
    assert len(comp.split_turns(view)) == 2  # 只数到 h0/a0, h1/a1 两轮


def test_maybe_compress_below_threshold_returns_none():
    comp = SessionCompressor(_FakeLlm(), keep_recent_turns=6, bus=EventBus(), session_id="s1")
    assert asyncio.run(comp.maybe_compress(_msgs_with_turns(3), None)) is None


def test_maybe_compress_generates_summary_and_event():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.CONTEXT_COMPRESSED, lambda e: seen.append(e.payload))
    comp = SessionCompressor(_FakeLlm(), keep_recent_turns=2, bus=bus, session_id="s1")

    summary = asyncio.run(comp.maybe_compress(_msgs_with_turns(8), None))
    assert summary == "压缩摘要"
    assert seen and seen[0]["compressed_turns"] == 6  # 8 轮 - 保留 2 轮 = 压缩 6 轮
    assert seen[0]["forced"] is False  # 自动压缩


def test_force_compress_early():
    """手动提前压缩：force=True 且 keep 更小 → 未到阈值也压缩"""
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.CONTEXT_COMPRESSED, lambda e: seen.append(e.payload))
    comp = SessionCompressor(_FakeLlm(), keep_recent_turns=6, bus=bus, session_id="s1")

    summary = asyncio.run(comp.maybe_compress(_msgs_with_turns(4), None, force=True, keep_recent_turns=2))
    assert summary == "压缩摘要"
    assert seen[0]["compressed_turns"] == 2  # 4 轮保留最近 2 轮，压前 2 轮
    assert seen[0]["forced"] is True  # 手动触发标记，前端可感知


def test_force_compress_no_early_returns_none():
    """force 但消息不足保留数 → 无可压早期轮返回 None"""
    comp = SessionCompressor(_FakeLlm(), keep_recent_turns=6, bus=EventBus(), session_id="s1")
    assert asyncio.run(comp.maybe_compress(_msgs_with_turns(2), None, force=True)) is None


def test_build_llm_context_with_summary():
    """LLM 视图：System + Human(摘要,meta=summary) 固定第 2 条 + 最近 1 轮"""
    comp = SessionCompressor(None, keep_recent_turns=1, bus=EventBus(), session_id="s1")
    ctx = comp.build_llm_context(_msgs_with_turns(2), "摘要")
    assert isinstance(ctx[1], HumanMessage)
    assert ctx[1].content == "[对话历史摘要]\n摘要"
    assert ctx[1].additional_kwargs.get("meta") == "summary"
    assert [m.content for m in ctx] == ["sys", "[对话历史摘要]\n摘要", "h1", "a1"]


def test_build_compress_input_prefix_preserved():
    """压缩请求 = 视图去掉最近 keep 轮【原封不动】（不过滤 system、不重排角色）+ 末尾固定 Human 指令"""
    comp = SessionCompressor(None, keep_recent_turns=2, bus=EventBus(), session_id="s1")
    # build_compress_input 的输入是完整视图（System + 摘要 + 全部轮），不是 build_llm_context 的压缩输出
    view = [
        SystemMessage(content="sys"),
        HumanMessage(content="[对话历史摘要]\n摘要", additional_kwargs={"meta": "summary"}),
        *_msgs_with_turns(5)[1:],
    ]
    request, early = comp.build_compress_input(view, 2)
    assert early == 3  # 压掉前 3 轮
    # 结构：System + Human摘要 + 前3轮原样 + 末尾 Human 压缩指令
    assert isinstance(request[0], SystemMessage)
    assert request[1].additional_kwargs.get("meta") == "summary"
    # 早期轮原封不动（顺序/角色未改）
    assert request[2].content == "h0" and request[3].content == "a0"
    assert request[4].content == "h1" and request[5].content == "a1"
    assert request[6].content == "h2" and request[7].content == "a2"
    # 末尾追加固定指令（HumanMessage）
    assert isinstance(request[-1], HumanMessage)
    assert "总结" in request[-1].content


def test_build_compress_input_no_early():
    """消息不足保留数 → (None, 0)"""
    comp = SessionCompressor(None, keep_recent_turns=2, bus=EventBus(), session_id="s1")
    request, early = comp.build_compress_input(_msgs_with_turns(2), 2)
    assert request is None and early == 0


def test_build_display_context_collapse():
    """前端视图：折叠占位 + 最近 1 轮真实消息（不含摘要）"""
    comp = SessionCompressor(None, keep_recent_turns=1, bus=EventBus(), session_id="s1")
    disp = comp.build_display_context(_msgs_with_turns(2), "摘要")
    assert disp[0]["kind"] == "collapse"
    assert disp[0]["collapsed_count"] == 2  # h0,a0
    assert [d["kind"] for d in disp[1:]] == ["message", "message"]
    assert [d["data"].content for d in disp[1:]] == ["h1", "a1"]
