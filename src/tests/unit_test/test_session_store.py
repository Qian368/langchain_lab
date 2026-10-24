"""ContextStore 单测：save/load 往返 + 订阅者快照 + list 过滤 + 增量归档/视图分离（不依赖 LLM）"""
import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict

from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


def _run(coro):
    return asyncio.run(coro)


def test_save_load_roundtrip():
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
    assert loaded is not None and [m.type for m in loaded] == ["system", "human", "ai", "tool"]
    assert loaded[2].tool_calls[0]["id"] == "call_1"
    assert loaded[3].tool_call_id == "call_1"


def test_load_missing_returns_none():
    store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
    assert _run(store.load("nope")) is None


def test_subscriber_persists_snapshot():
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        bus.publish(Event(type=EventType.SESSION_STATE_UPDATED, session_id="s1", payload={"messages": messages_to_dict(msgs), "session_ctx": {"user_id": "u1", "tenant_id": "t1"}, "branch_id": "root", "summary": None}))
        await bus.flush()
        loaded = await store.load("s1")
        assert loaded is not None and [m.type for m in loaded] == ["system", "human"]
    asyncio.run(scenario())


def test_list_sessions_filter_by_user_tenant():
    db = Path(tempfile.mkdtemp()) / "s.db"
    store = ContextStore(db)
    _run(store.save("s1", [HumanMessage(content="a")], {"user_id": "u1", "tenant_id": "t1"}))
    _run(store.save("s2", [HumanMessage(content="b")], {"user_id": "u1", "tenant_id": "t2"}))
    _run(store.save("s3", [HumanMessage(content="c")], {"user_id": "u2", "tenant_id": "t1"}))
    assert [s["session_id"] for s in _run(store.list_sessions("u1", "t1"))] == ["s1"]
    assert [s["session_id"] for s in _run(store.list_sessions("u1", "t2"))] == ["s2"]


def test_archived_append_and_full_history():
    """增量归档：MESSAGES_ARCHIVED 逐轮 append → load_full_history 重建全量（append-only）"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": "root", "messages": messages_to_dict([HumanMessage(content="h0"), AIMessage(content="a0")])}))
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": "root", "messages": messages_to_dict([HumanMessage(content="h1"), AIMessage(content="a1")])}))
        await bus.flush()
        full = await store.load_full_history("s1")
        assert [m.content for m in full] == ["h0", "a0", "h1", "a1"]
    asyncio.run(scenario())


def test_view_snapshot_separate_from_archive():
    """压缩改写视图后：load_branch 读当前视图（含摘要、不含早期轮），load_full_history 仍全量"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        # 早期轮已增量归档（压缩前每轮追加）
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": "root", "messages": messages_to_dict([HumanMessage(content="h0"), AIMessage(content="a0")])}))
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": "root", "messages": messages_to_dict([HumanMessage(content="h1"), AIMessage(content="a1")])}))
        # 压缩后当前视图 = System + Human摘要 + 最近轮
        view = [
            SystemMessage(content="sys"),
            HumanMessage(content="[对话历史摘要]\n摘要", additional_kwargs={"meta": "summary"}),
            HumanMessage(content="h1"),
            AIMessage(content="a1"),
        ]
        bus.publish(Event(EventType.SESSION_STATE_UPDATED, "s1", {"messages": messages_to_dict(view), "session_ctx": {"user_id": "u1", "tenant_id": "t1"}, "branch_id": "root", "summary": "摘要"}))
        await bus.flush()
        # 视图：当前视图（含摘要，h0,a0 已被压缩移出）
        v = await store.load_branch("s1")
        assert [m.content for m in v] == ["sys", "[对话历史摘要]\n摘要", "h1", "a1"]
        # 全量：归档仍含 h0,a0,h1,a1
        full = await store.load_full_history("s1")
        assert [m.content for m in full] == ["h0", "a0", "h1", "a1"]
    asyncio.run(scenario())
