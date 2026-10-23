"""ChatSession 行为测试：事件流 + 消息累积 + 压缩事件化 + 控制指令事件（不连真实 LLM/工具）"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeLlmWithTools:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, messages):
        return _FakeStream(self._chunks)


class _FakeSummarizeLlm:
    async def ainvoke(self, messages):
        return SimpleNamespace(content="摘要")


def _history(n: int):
    msgs = [SystemMessage(content="sys")]
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    return msgs


def test_agent_loop_emits_and_accumulates():
    async def scenario():
        bus = EventBus()
        seen: list[tuple] = []
        bus.subscribe(EventType.STREAM_CHUNK, lambda e: seen.append((e.type, e.payload)))
        bus.subscribe(EventType.ASSISTANT_MESSAGE_EMITTED, lambda e: seen.append((e.type, None)))

        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=None, tools=[],
        )
        fake = _FakeLlmWithTools([AIMessageChunk(content="你"), AIMessageChunk(content="好")])
        await chat._agent_loop(llm_with_tools=fake)

        assert [t for t, _ in seen] == [
            EventType.STREAM_CHUNK, EventType.STREAM_CHUNK, EventType.ASSISTANT_MESSAGE_EMITTED,
        ]
        assert seen[0][1] == "你"
        assert seen[1][1] == "好"
        assert len(chat._messages) == 2
        assert chat._messages[-1].content == "你好"

    asyncio.run(scenario())


def test_compress_requested_updates_summary():
    async def scenario():
        bus = EventBus()
        compressed = []
        bus.subscribe(EventType.CONTEXT_COMPRESSED, lambda e: compressed.append(e.payload))

        comp = SessionCompressor(_FakeSummarizeLlm(), keep_recent_turns=6, bus=bus, session_id="s1")
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=None, tools=[], compressor=comp, history=_history(3),
        )
        bus.publish(Event(EventType.COMPRESS_REQUESTED, "s1", {"force": True, "keep": 2}))
        await bus.flush()

        assert chat._summary == "摘要"
        assert compressed and compressed[0]["forced"] is True
        assert compressed[0]["compressed_turns"] == 1

    asyncio.run(scenario())


def test_session_command_branch_create():
    async def scenario():
        bus = EventBus()
        created = []

        async def fake_creator(parent, point):
            bid = f"b_{point}"
            created.append(bid)
            return bid

        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=None, tools=[], history=_history(3), branch_creator=fake_creator,
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "branch_create", "point": 2}))
        await bus.flush()

        assert chat._branch_id == "b_2"
        assert created == ["b_2"]
        assert [m.content for m in chat._messages] == ["sys", "h0", "a0"]

    asyncio.run(scenario())


def test_session_command_unknown_action_notice():
    async def scenario():
        bus = EventBus()
        notices = []
        bus.subscribe(EventType.SESSION_NOTICE, lambda e: notices.append(e.payload))
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=None, tools=[],
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "nope"}))
        await bus.flush()
        assert notices and "nope" in notices[0]

    asyncio.run(scenario())
