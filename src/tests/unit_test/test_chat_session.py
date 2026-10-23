"""ChatSession 行为测试：事件流 + 消息累积 + 压缩事件化 + 控制指令事件 + 控制钩子（不连真实 LLM/工具）"""
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
    """带 bind_tools 的假 LLM：供 _handle_turn 使用"""

    def __init__(self, chunks=None):
        self._chunks = chunks or [AIMessageChunk(content="ok")]

    def bind_tools(self, tools):
        return _FakeLlmWithTools(self._chunks)


class _FakeSummarizeLlm:
    async def ainvoke(self, messages):
        return SimpleNamespace(content="摘要")


@tool
def _fake_echo(x: int) -> str:
    """echo 工具，供 post_tool 测试"""
    return f"v{x}"


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


# —— 控制钩子（AgentHooks）——
def test_run_hook_semantics():
    """_run_hook 返回值语义：None=原样；非None=替换；async 自动 await"""
    async def scenario():
        async def ahook(x):
            return "a:" + x

        assert await ChatSession._run_hook(None, "v") == "v"  # 空钩子放行
        assert await ChatSession._run_hook(lambda c: c + "!", "v") == "v!"  # 同步替换
        assert await ChatSession._run_hook(lambda c: None, "v") == "v"  # None 原样
        assert await ChatSession._run_hook(ahook, "v") == "a:v"  # async 替换

    asyncio.run(scenario())


def test_pre_user_message_rewrite():
    """pre_user_message 改写输入：HumanMessage 存改写后内容"""
    async def scenario():
        bus = EventBus()
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=_FakeLlm(), tools=[],
            hooks=AgentHooks(pre_user_message=lambda c: "[改写]" + c),
        )
        bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, "s1", {"content": "原"}))
        await bus.flush()
        # system + HumanMessage(改写) + AIMessage(fake)
        assert any(isinstance(m, HumanMessage) and m.content == "[改写]原" for m in chat._messages)

    asyncio.run(scenario())


def test_pre_user_message_reject():
    """pre_user_message 抛异常 → ERROR_OCCURRED + 用户消息不追加（拒绝/中断）"""
    async def scenario():
        bus = EventBus()
        errors = []
        bus.subscribe(EventType.ERROR_OCCURRED, lambda e: errors.append(e.payload))
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=_FakeLlm(), tools=[],
            hooks=AgentHooks(pre_user_message=lambda c: (_ for _ in ()).throw(ValueError("非法输入"))),
        )
        bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, "s1", {"content": "坏"}))
        await bus.flush()
        assert errors and "非法输入" in errors[0]["error"]
        # 无新 HumanMessage（拒绝未追加）
        assert not any(isinstance(m, HumanMessage) for m in chat._messages)

    asyncio.run(scenario())


def test_post_llm_replace():
    """post_llm 替换输出：messages 存替换后的 AIMessage"""
    async def scenario():
        bus = EventBus()

        def replace(msg):
            return AIMessage(content="替换输出")

        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=None, tools=[], hooks=AgentHooks(post_llm=replace),
        )
        await chat._agent_loop(llm_with_tools=_FakeLlmWithTools([AIMessageChunk(content="原始")]))
        assert chat._messages[-1].content == "替换输出"

    asyncio.run(scenario())


def test_post_tool_replace():
    """post_tool 替换结果：ToolMessage 存脱敏后内容"""
    async def scenario():
        bus = EventBus()

        def sanitize(result):
            result["content"] = "脱敏后"
            return result

        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=None, tools=[_fake_echo], hooks=AgentHooks(post_tool=sanitize),
        )
        # 一轮带 tool_call：执行后走 post_tool
        fake = _FakeLlmWithTools([AIMessageChunk(content="", tool_calls=[{"name": "_fake_echo", "id": "t1", "args": {"x": 1}}])])
        await chat._agent_loop(llm_with_tools=fake)
        assert chat._messages[-1].content == '"脱敏后"'  # ToolMessage json.dumps 后

    asyncio.run(scenario())
