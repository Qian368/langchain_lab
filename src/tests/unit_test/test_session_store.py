"""ContextStore 单测：save/load 往返（含 tool_calls 无损）+ list_sessions 过滤（不依赖 LLM）"""
import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_lab.core.context.context_store import ContextStore


def _run(coro):
    return asyncio.run(coro)


def test_save_load_roundtrip():
    """完整消息往返：类型、tool_calls、tool_call_id 无损"""
    db = Path(tempfile.mkdtemp()) / "s.db"
    store = ContextStore(db)
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="hello", tool_calls=[{"name": "f", "args": {"x": 1}, "id": "call_1"}]),
        ToolMessage(content="{}", tool_call_id="call_1", name="f"),
    ]
    _run(store.save("s1", msgs, session_ctx={"user_id": "u1", "tenant_id": "t1"}))

    loaded = _run(store.load("s1"))
    assert loaded is not None
    assert [m.type for m in loaded] == ["system", "human", "ai", "tool"]
    assert loaded[2].tool_calls[0]["id"] == "call_1"  # tool_call 无损
    assert loaded[3].tool_call_id == "call_1"  # tool_call_id 无损
    assert loaded[3].name == "f"


def test_load_missing_returns_none():
    """会话不存在返回 None"""
    store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
    assert _run(store.load("nope")) is None


def test_subscriber_persists_snapshot():
    """订阅者链路：发布 SESSION_STATE_UPDATED 快照 → 落库 → load 可读（事件即真相）"""
    from langchain_core.messages import messages_to_dict

    from langchain_lab.core.event_bus import EventBus
    from langchain_lab.core.events import Event, EventType

    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        bus.publish(
            Event(
                type=EventType.SESSION_STATE_UPDATED,
                session_id="s1",
                payload={
                    "messages": messages_to_dict(msgs),
                    "session_ctx": {"user_id": "u1", "tenant_id": "t1"},
                    "branch_id": "root",
                    "summary": None,
                },
            )
        )
        await bus.flush()
        loaded = await store.load("s1")
        assert loaded is not None and [m.type for m in loaded] == ["system", "human"]

    asyncio.run(scenario())


def test_list_sessions_filter_by_user_tenant():
    """SaaS 检索：按 user/tenant 过滤会话"""
    db = Path(tempfile.mkdtemp()) / "s.db"
    store = ContextStore(db)
    _run(store.save("s1", [HumanMessage(content="a")], {"user_id": "u1", "tenant_id": "t1"}))
    _run(store.save("s2", [HumanMessage(content="b")], {"user_id": "u1", "tenant_id": "t2"}))
    _run(store.save("s3", [HumanMessage(content="c")], {"user_id": "u2", "tenant_id": "t1"}))

    assert [s["session_id"] for s in _run(store.list_sessions("u1", "t1"))] == ["s1"]
    assert [s["session_id"] for s in _run(store.list_sessions("u1", "t2"))] == ["s2"]
