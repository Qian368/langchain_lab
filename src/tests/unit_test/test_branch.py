"""ContextStore 分支单测：创建/加载/列出/摘要共享（事件驱动，不依赖 LLM）"""
import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, messages_to_dict

from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


def _run(coro):
    return asyncio.run(coro)


def _msgs(n: int):
    """system + n 轮(human+ai)"""
    msgs = [SystemMessage(content="sys")]
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    return msgs


def _seed(store, bus, sid, msgs, ctx, summary=None, branch_id="root"):
    """事件驱动归档：建会话 + MESSAGES_ARCHIVED 落真实消息(排除 system) + 可选 CONTEXT_COMPRESSED 更新会话级摘要"""

    async def _seed_async():
        await store.ensure_session(sid, ctx, system_prompt="sys")
        real = [m for m in msgs if not isinstance(m, SystemMessage)]
        if real:
            bus.publish(Event(EventType.MESSAGES_ARCHIVED, sid, {"branch_id": branch_id, "messages": messages_to_dict(real)}))
        if summary:
            bus.publish(Event(EventType.CONTEXT_COMPRESSED, sid, {"summary": summary}))
        await bus.flush()

    _run(_seed_async())


def _new_store_bus():
    store = ContextStore(Path(tempfile.mkdtemp()) / "b.db")
    bus = EventBus()
    store.subscribe_to_bus(bus)
    return store, bus


def test_create_and_list_branch():
    store, bus = _new_store_bus()
    _seed(store, bus, "s1", _msgs(3), {"user_id": "u1", "tenant_id": "t1"})
    new, _ = _run(store.create_branch("s1", "root", 2))

    branches = _run(store.list_branches("s1"))
    ids = [b["branch_id"] for b in branches]
    assert "root" in ids and new in ids
    root = next(b for b in branches if b["branch_id"] == "root")
    assert root["parent_id"] is None and root["branch_point_seq"] is None
    sub = next(b for b in branches if b["branch_id"] == new)
    assert sub["parent_id"] == "root" and sub["branch_point_seq"] == 2


def test_load_branch_snapshot():
    store, bus = _new_store_bus()
    _seed(store, bus, "s1", _msgs(2), {"user_id": "u1", "tenant_id": "t1"})
    new, _ = _run(store.create_branch("s1", "root", 1))
    # 子分支增量：拐点(第1条 h0)后新增 a0；h0 由继承提供，只归档新增避免重复
    _seed(store, bus, "s1", [AIMessage(content="a0")], {"user_id": "u1", "tenant_id": "t1"}, branch_id=new)
    loaded = _run(store.load_view("s1", new, 6))
    assert [m.content for m in loaded] == ["sys", "h0", "a0"]


def test_load_branch_unknown_raises():
    store, bus = _new_store_bus()
    _seed(store, bus, "s1", _msgs(1), {"user_id": "u1", "tenant_id": "t1"})
    try:
        _run(store.load_full_history("s1", "nope"))
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_summary_shared_across_branches():
    """摘要存会话级，各分支共享（规范 7.1）"""
    store, bus = _new_store_bus()
    _seed(store, bus, "s1", _msgs(6), {"user_id": "u1", "tenant_id": "t1"}, summary="摘要")
    new, _ = _run(store.create_branch("s1", "root", 3))
    _seed(store, bus, "s1", _msgs(6)[:4], {"user_id": "u1", "tenant_id": "t1"}, branch_id=new, summary="摘要")
    assert _run(store.load_summary("s1")) == "摘要"
