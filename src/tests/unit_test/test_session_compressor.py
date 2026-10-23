"""SessionCompressor 单测：轮次切分 / 摘要生成 / 双视图投影（不连真实 LLM）"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import EventType
from langchain_lab.core.session_compressor import SessionCompressor


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


def test_split_turns():
    msgs = _msgs_with_turns(2)
    turns = SessionCompressor.split_turns(msgs)
    assert len(turns) == 2
    assert turns[0][0].content == "h0"
    assert turns[1][0].content == "h1"


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


def test_build_llm_context_with_summary():
    comp = SessionCompressor(None, keep_recent_turns=1, bus=EventBus(), session_id="s1")
    ctx = comp.build_llm_context(_msgs_with_turns(2), "摘要")
    # System + 摘要置顶 + 最近 1 轮(h1,a1)
    assert [m.content for m in ctx] == ["sys", "[早期上下文摘要]\n摘要", "h1", "a1"]


def test_build_display_context_collapse():
    comp = SessionCompressor(None, keep_recent_turns=1, bus=EventBus(), session_id="s1")
    disp = comp.build_display_context(_msgs_with_turns(2), "摘要")
    # 前端视图：折叠占位 + 最近 1 轮真实消息（不含摘要）
    assert disp[0]["kind"] == "collapse"
    assert disp[0]["collapsed_count"] == 2  # h0,a0
    assert [d["kind"] for d in disp[1:]] == ["message", "message"]
    assert [d["data"].content for d in disp[1:]] == ["h1", "a1"]
