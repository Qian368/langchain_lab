"""ChatSession 行为测试：事件流 + 消息累积（不连真实 LLM/工具）

用假 stream 验证：流式 chunk 逐片 emit、累积完成后 emit 完整消息、messages 状态正确。
"""
import asyncio

from langchain_core.messages import AIMessageChunk

from langchain_lab.core.agent_session import ChatSession
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
            view=object(),
            tools=[],
            store=None,
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
