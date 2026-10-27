"""ContextStore 单测（单一真相源+树形）：ensure_session + 归档 + load_view 构建视图 + 分支拐点视图 + 幂等去重（不依赖 LLM）"""

import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from langchain_lab.core.bus.event_bus import EventBus
from langchain_lab.core.bus.events import Event, EventType
from langchain_lab.core.context.context_store import ContextStore


def _run(coro):
    return asyncio.run(coro)


def test_ensure_session_and_view_new():
    """新建：ensure_session 存 system_prompt + 归档 → load_view 返回 [system]+全量真实消息"""

    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, system_prompt="sys")
        bus.publish(
            Event(
                EventType.MESSAGES_ARCHIVED,
                "s1",
                {"branch_id": "root", "messages": messages_to_dict([HumanMessage(content="h0"), AIMessage(content="a0")])},
            )
        )
        await bus.flush()
        view = await store.load_view("s1", "root", keep_recent_turns=6)
        assert view is not None
        assert [m.content for m in view] == ["sys", "h0", "a0"]
        assert view[0].type == "system"

    asyncio.run(scenario())


def test_view_rebuilds_after_compress():
    """已压缩：摘要经 SUMMARY_ARCHIVED 进 archive，load_view 从 archive 取最近摘要构建 System+摘要+最近 keep 轮（KV 一致）"""

    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, system_prompt="sys")
        bus.publish(
            Event(
                EventType.MESSAGES_ARCHIVED,
                "s1",
                {
                    "branch_id": "root",
                    "messages": messages_to_dict(
                        [
                            HumanMessage(content="h0"),
                            AIMessage(content="a0"),
                            HumanMessage(content="h1"),
                            AIMessage(content="a1"),
                            HumanMessage(content="h2"),
                            AIMessage(content="a2"),
                        ]
                    ),
                },
            )
        )
        bus.publish(Event(EventType.SUMMARY_ARCHIVED, "s1", {"summary": "摘要", "branch_id": "root"}))
        await bus.flush()
        view = await store.load_view("s1", "root", keep_recent_turns=2)
        assert [m.type for m in view] == ["system", "human", "human", "ai", "human", "ai"]
        assert view[1].additional_kwargs.get("meta") == "summary"
        assert view[1].content == "[对话历史摘要]\n摘要"
        assert [m.content for m in view[2:]] == ["h1", "a1", "h2", "a2"]  # 最近 keep=2 轮

    asyncio.run(scenario())


def test_archived_idempotent_dedup():
    """幂等：同 event_id 重复归档 → INSERT OR IGNORE 不双写"""

    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, "sys")
        ev = Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": "root", "messages": messages_to_dict([HumanMessage(content="h0")])})
        bus.publish(ev)
        bus.publish(ev)  # 同一事件重放
        await bus.flush()
        full = await store.load_full_history("s1")
        assert [m.content for m in full] == ["h0"]  # 只写一次

    asyncio.run(scenario())


def test_create_branch_returns_point_view():
    """分支：create_branch 从 archive 构建拐点视图（System + 前 point 轮），支持压缩后从历史轮次分叉"""

    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, "sys")
        bus.publish(
            Event(
                EventType.MESSAGES_ARCHIVED,
                "s1",
                {
                    "branch_id": "root",
                    "messages": messages_to_dict(
                        [
                            HumanMessage(content="h0"),
                            AIMessage(content="a0"),
                            HumanMessage(content="h1"),
                            AIMessage(content="a1"),
                        ]
                    ),
                },
            )
        )
        await bus.flush()
        new_id, point_view = await store.create_branch("s1", "root", 2, keep_recent_turns=6)
        assert point_view[0].content == "sys"
        assert [m.content for m in point_view[1:]] == ["h0", "a0"]  # 前 2 轮
        branches = await store.list_branches("s1")
        assert any(b["branch_id"] == new_id and b["branch_point_seq"] == 2 for b in branches)

    asyncio.run(scenario())


def test_create_branch_point_out_of_range():
    """拐点越界：create_branch 抛 ValueError（不静默截断）"""

    async def scenario():
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, "sys")
        try:
            await store.create_branch("s1", "root", 99, keep_recent_turns=6)
            raise AssertionError("应抛 ValueError")
        except ValueError:
            pass

    asyncio.run(scenario())


def test_load_view_missing_returns_none():
    """会话不存在 → load_view 返回 None"""
    store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
    assert _run(store.load_view("nope", "root", 6)) is None


def test_list_sessions_filter_by_user_tenant():
    """SaaS 检索：按 user/tenant 过滤会话"""

    async def scenario():
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, "sys")
        await store.ensure_session("s2", {"user_id": "u1", "tenant_id": "t2"}, "sys")
        await store.ensure_session("s3", {"user_id": "u2", "tenant_id": "t1"}, "sys")
        assert [s["session_id"] for s in await store.list_sessions("u1", "t1")] == ["s1"]
        assert [s["session_id"] for s in await store.list_sessions("u1", "t2")] == ["s2"]

    asyncio.run(scenario())
