"""ChatSession 行为测试：事件流 + 消息累积 + 压缩事件化触发（不连真实 LLM/工具）

用假 stream 验证：流式 chunk 逐片 emit、累积完成后 emit 完整消息、messages 状态正确。
压缩事件化：publish COMPRESS_REQUESTED → 异步生成摘要更新 _summary + 发 CONTEXT_COMPRESSED。
"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


class _FakeStream:
    """假 LLM 流式：固定分片"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeLlmWithTools:
    """假 llm_with_tools：stream 返回固定分片，无 tool_calls"""

    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, messages):
        return _FakeStream(self._chunks)


class _FakeSummarizeLlm:
    """假摘要 LLM：返回固定摘要"""

    async def ainvoke(self, messages):
        return SimpleNamespace(content="摘要")


def _history(n: int):
    """system + n 轮(human+ai)消息，供恢复注入"""
    msgs = [SystemMessage(content="sys")]
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    return msgs


def test_agent_loop_emits_and_accumulates():
    """流式 chunk 逐片 emit；累积完成后 emit 完整消息；messages 正确累积"""
    async def scenario():
        bus = EventBus()
        seen: list[tuple] = []
        bus.subscribe(EventType.STREAM_CHUNK, lambda e: seen.append((e.type, e.payload)))
        bus.subscribe(EventType.ASSISTANT_MESSAGE_EMITTED, lambda e: seen.append((e.type, None)))

        chat = ChatSession(
            session_id="s1",
            session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus,
            llm=None,  # _agent_loop 用传入的 llm_with_tools，不依赖 self._llm
            tools=[],
        )
        fake = _FakeLlmWithTools([AIMessageChunk(content="你"), AIMessageChunk(content="好")])
        await chat._agent_loop(llm_with_tools=fake)

        # 事件序列：两个 STREAM_CHUNK + 一个 ASSISTANT_MESSAGE_EMITTED
        assert [t for t, _ in seen] == [
            EventType.STREAM_CHUNK,
            EventType.STREAM_CHUNK,
            EventType.ASSISTANT_MESSAGE_EMITTED,
        ]
        assert seen[0][1] == "你"
        assert seen[1][1] == "好"
        # 消息累积：系统提示 + 累积的 AIMessage
        assert len(chat._messages) == 2
        assert chat._messages[-1].content == "你好"

    asyncio.run(scenario())


def test_compress_requested_updates_summary():
    """前端手动触发压缩：publish COMPRESS_REQUESTED(force=True) → 生成摘要更新 _summary + CONTEXT_COMPRESSED"""
    async def scenario():
        bus = EventBus()
        compressed = []
        bus.subscribe(EventType.CONTEXT_COMPRESSED, lambda e: compressed.append(e.payload))

        comp = SessionCompressor(_FakeSummarizeLlm(), keep_recent_turns=6, bus=bus, session_id="s1")
        chat = ChatSession(
            session_id="s1",
            session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus,
            llm=None,
            tools=[],
            compressor=comp,
            history=_history(3),  # 恢复注入历史
        )
        # 前端手动提前压缩：保留最近 2 轮
        bus.publish(Event(EventType.COMPRESS_REQUESTED, "s1", {"force": True, "keep": 2}))
        await bus.flush()

        assert chat._summary == "摘要"  # 运行时已拿到新摘要（供 LLM 视图）
        assert compressed and compressed[0]["forced"] is True
        assert compressed[0]["compressed_turns"] == 1  # 3 轮保留最近 2 轮，压掉前 1 轮

    asyncio.run(scenario())
