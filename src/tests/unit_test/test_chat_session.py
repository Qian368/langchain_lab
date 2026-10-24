"""ChatSession 行为测试：事件流 + 消息累积 + 压缩重建视图 + 增量归档 + 控制指令 + 控制钩子（不连真实 LLM/工具）"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_lab.core.agent_hooks import AgentHooks
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


class _FakeLlm:
    def __init__(self, chunks=None):
        self._chunks = chunks or [AIMessageChunk(content="ok")]

    def bind_tools(self, tools):
        return _FakeLlmWithTools(self._chunks)


class _FakeSummarizeLlm:
    async def ainvoke(self, messages):
        return SimpleNamespace(content="摘要")


@tool
def _fake_echo(x: int) -> str:
    return f"v{x}"


def _history(n: int):
    msgs = [SystemMessage(content="sys")]
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    return msgs


def test_agent_loop_emits_and_accumulates():
    async def scenario():
        bus = EventBus()
        seen = []
        bus.subscribe(EventType.STREAM_CHUNK, lambda e: seen.append((e.type, e.payload)))
        bus.subscribe(EventType.ASSISTANT_MESSAGE_EMITTED, lambda e: seen.append((e.type, None)))
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=None, tools=[])
        await chat._agent_loop(llm_with_tools=_FakeLlmWithTools([AIMessageChunk(content="你"), AIMessageChunk(content="好")]))
        assert [t for t, _ in seen] == [EventType.STREAM_CHUNK, EventType.STREAM_CHUNK, EventType.ASSISTANT_MESSAGE_EMITTED]
        assert chat._messages[-1].content == "你好"

    asyncio.run(scenario())


def test_compress_requested_updates_summary_and_rebuilds():
    """前端手动触发压缩：生成摘要 + 重建 _messages（早期轮移出、Human摘要置顶第2条、保留最近keep轮）"""
    async def scenario():
        bus = EventBus()
        comp = SessionCompressor(_FakeSummarizeLlm(), keep_recent_turns=6, bus=bus, session_id="s1")
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=None, tools=[], compressor=comp, history=_history(3))
        bus.publish(Event(EventType.COMPRESS_REQUESTED, "s1", {"force": True, "keep": 2}))
        await bus.flush()

        assert chat._summary == "摘要"
        # _messages 重建：System + Human摘要 + 最近 2 轮（早期 1 轮移出）
        assert chat._messages[0].content == "sys"
        assert isinstance(chat._messages[1], HumanMessage)
        assert chat._messages[1].additional_kwargs.get("meta") == "summary"
        assert [m.content for m in chat._messages[2:]] == ["h1", "a1", "h2", "a2"]

    asyncio.run(scenario())


def test_llm_messages_is_single_source():
    """LLM 视图 = 单源 _messages（压缩后即重建后的视图，不再每次投影过滤）"""
    async def scenario():
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=EventBus(), llm=None, tools=[], history=_history(2))
        assert chat._llm_messages is chat._messages  # 同一对象，非投影副本
    asyncio.run(scenario())


def test_turn_archives_new_messages():
    """每轮增量归档：_handle_turn 发 MESSAGES_ARCHIVED（本轮 human+ai，不含 system）"""
    async def scenario():
        bus = EventBus()
        archived = []
        bus.subscribe(EventType.MESSAGES_ARCHIVED, lambda e: archived.append(e.payload))
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=_FakeLlm(), tools=[], history=_history(0))
        bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, "s1", {"content": "hi"}))
        await bus.flush()

        assert archived
        types = [d.get("type") for d in archived[0]["messages"]]
        assert "human" in types and "ai" in types  # 本轮新增 human+ai，system 被排除
        assert archived[0]["branch_id"] == "root"

    asyncio.run(scenario())


def test_session_command_branch_create():
    async def scenario():
        bus = EventBus()
        created = []

        async def fake_creator(parent, point):
            bid = f"b_{point}"
            created.append(bid)
            return bid

        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=None, tools=[], history=_history(3), branch_creator=fake_creator)
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "branch_create", "point": 2}))
        await bus.flush()
        assert chat._branch_id == "b_2"
        assert [m.content for m in chat._messages] == ["sys", "h0", "a0"]

    asyncio.run(scenario())


def test_session_command_unknown_action_notice():
    async def scenario():
        bus = EventBus()
        notices = []
        bus.subscribe(EventType.SESSION_NOTICE, lambda e: notices.append(e.payload))
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=None, tools=[])
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "nope"}))
        await bus.flush()
        assert notices and "nope" in notices[0]

    asyncio.run(scenario())


# —— 控制钩子 ——
def test_run_hook_semantics():
    async def scenario():
        async def ahook(x):
            return "a:" + x
        assert await ChatSession._run_hook(None, "v") == "v"
        assert await ChatSession._run_hook(lambda c: c + "!", "v") == "v!"
        assert await ChatSession._run_hook(lambda c: None, "v") == "v"
        assert await ChatSession._run_hook(ahook, "v") == "a:v"
    asyncio.run(scenario())


def test_pre_user_message_rewrite():
    async def scenario():
        bus = EventBus()
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=_FakeLlm(), tools=[], hooks=AgentHooks(pre_user_message=lambda c: "[改写]" + c))
        bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, "s1", {"content": "原"}))
        await bus.flush()
        assert any(isinstance(m, HumanMessage) and m.content == "[改写]原" for m in chat._messages)
    asyncio.run(scenario())


def test_pre_user_message_reject():
    async def scenario():
        bus = EventBus()
        errors = []
        bus.subscribe(EventType.ERROR_OCCURRED, lambda e: errors.append(e.payload))
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus, llm=_FakeLlm(), tools=[], hooks=AgentHooks(pre_user_message=lambda c: (_ for _ in ()).throw(ValueError("非法输入"))))
        bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, "s1", {"content": "坏"}))
        await bus.flush()
        assert errors and "非法输入" in errors[0]["error"]
        assert not any(isinstance(m, HumanMessage) for m in chat._messages)
    asyncio.run(scenario())


def test_post_llm_replace():
    async def scenario():
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=EventBus(), llm=None, tools=[], hooks=AgentHooks(post_llm=lambda msg: AIMessage(content="替换输出")))
        await chat._agent_loop(llm_with_tools=_FakeLlmWithTools([AIMessageChunk(content="原始")]))
        assert chat._messages[-1].content == "替换输出"
    asyncio.run(scenario())


def test_post_tool_replace():
    async def scenario():
        def sanitize(result):
            result["content"] = "脱敏后"
            return result
        chat = ChatSession(session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=EventBus(), llm=None, tools=[_fake_echo], hooks=AgentHooks(post_tool=sanitize))
        fake = _FakeLlmWithTools([AIMessageChunk(content="", tool_calls=[{"name": "_fake_echo", "id": "t1", "args": {"x": 1}}])])
        await chat._agent_loop(llm_with_tools=fake)
        assert chat._messages[-1].content == '"脱敏后"'
    asyncio.run(scenario())
