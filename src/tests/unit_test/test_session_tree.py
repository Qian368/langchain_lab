"""树形会话测试：回溯分支历史(含继承)、任意位置分叉、摘要进归档、两级删除、命名、软删过滤
（不依赖真实 LLM；ChatSession 用注入回调）"""
import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, messages_to_dict

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


def _run(coro):
    return asyncio.run(coro)


def _arch(messages):
    return messages_to_dict(messages)


async def _seed_root(store: ContextStore, bus: EventBus, sid="s1", n=3) -> None:
    """注册会话并归档 n 轮真实消息到 root（h0,a0, h1,a1, ...）"""
    await store.ensure_session(sid, {"user_id": "u1", "tenant_id": "t1"}, "sys")
    msgs = []
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    bus.publish(Event(EventType.MESSAGES_ARCHIVED, sid, {"branch_id": "root", "messages": _arch(msgs)}))
    await bus.flush()


# —— 回溯分支历史 ——
def test_branch_history_inherits_ancestors():
    """branch_history(b1) = 继承 root 前 point 条 + b1 自有增量（无重复）"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await _seed_root(store, bus, n=3)  # root: h0,a0,h1,a1,h2,a2
        b1, _pv = await store.create_branch("s1", "root", 4)  # 拐点=root 第4条(h1)
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": b1, "messages": _arch([HumanMessage(content="hB0"), AIMessage(content="aB0")])}))
        await bus.flush()
        hist = await store.branch_history("s1", b1)
        assert [m.content for m in hist] == ["h0", "a0", "h1", "a1", "hB0", "aB0"]
        full = await store.load_full_history("s1", b1)
        assert [m.content for m in full] == ["h0", "a0", "h1", "a1", "hB0", "aB0"]
    asyncio.run(scenario())


def test_create_branch_from_branch_inherits():
    """从分支再分叉：拐点视图含继承历史（任意位置分叉，不止自有增量）"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await _seed_root(store, bus, n=3)
        b1, _ = await store.create_branch("s1", "root", 4)
        b2, pv = await store.create_branch("s1", b1, 2)
        assert pv[0].content == "sys"
        assert [m.content for m in pv[1:]] == ["h0", "a0"]
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": b2, "messages": _arch([HumanMessage(content="hB20")])}))
        await bus.flush()
        hist = await store.branch_history("s1", b2)
        assert [m.content for m in hist] == ["h0", "a0", "hB20"]
    asyncio.run(scenario())


# —— 摘要进归档 + 恢复取最近 ——
def test_summary_archived_and_recent_in_load_view():
    """SUMMARY_ARCHIVED → 摘要 kind=summary 进归档；load_view 取最近一条重建视图"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await _seed_root(store, bus, n=3)
        bus.publish(Event(EventType.SUMMARY_ARCHIVED, "s1", {"summary": "摘要一", "branch_id": "root"}))
        bus.publish(Event(EventType.SUMMARY_ARCHIVED, "s1", {"summary": "摘要二", "branch_id": "root"}))
        await bus.flush()
        view = await store.load_view("s1", "root", keep_recent_turns=2)
        assert view[0].content == "sys"
        assert isinstance(view[1], HumanMessage) and view[1].additional_kwargs.get("meta") == "summary"
        assert view[1].content == "[对话历史摘要]\n摘要二"
        assert [m.content for m in view[2:]] == ["h2", "a2"]
    asyncio.run(scenario())


def test_summary_archived_to_current_branch():
    """摘要按当前分支落归档：非 root 分支压缩摘要归属正确（不在 root）"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await _seed_root(store, bus, n=3)
        b1, _ = await store.create_branch("s1", "root", 4)
        # 在 b1 分支压缩 → 摘要落 b1
        bus.publish(Event(EventType.SUMMARY_ARCHIVED, "s1", {"summary": "b1摘要", "branch_id": b1}))
        await bus.flush()
        # root 归档不含该摘要
        assert not any(m.type == "human" and m.additional_kwargs.get("meta") == "summary" for m in await store.branch_history("s1", "root"))
        # b1 恢复取到该摘要
        v = await store.load_view("s1", b1, keep_recent_turns=6)
        assert isinstance(v[1], HumanMessage) and v[1].additional_kwargs.get("meta") == "summary"
        assert v[1].content == "[对话历史摘要]\nb1摘要"
    asyncio.run(scenario())


# —— 两级删除 ——
def test_delete_branch_soft_and_gc():
    """删除分支：软删子树 + 标记 archive；list_branches 过滤；gc_purge 物理清理"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await _seed_root(store, bus, n=3)
        b1, _ = await store.create_branch("s1", "root", 4)
        bus.publish(Event(EventType.MESSAGES_ARCHIVED, "s1", {"branch_id": b1, "messages": _arch([HumanMessage(content="hB0")])}))
        await bus.flush()
        deleted = await store.delete_branch("s1", b1)
        assert deleted == [b1]
        ids = [b["branch_id"] for b in await store.list_branches("s1")]
        assert b1 not in ids and "root" in ids
        try:
            await store.load_view("s1", b1, 6)
            raise AssertionError("软删分支应不可读")
        except ValueError:
            pass
        assert [m.content for m in await store.branch_history("s1", "root")] == ["h0", "a0", "h1", "a1", "h2", "a2"]
        r = await store.gc_purge()
        assert r["branches"] >= 1 and r["archive"] >= 1
    asyncio.run(scenario())


def test_delete_branch_cascades_descendants():
    """删除父分支：级联软删所有后代"""
    async def scenario():
        bus = EventBus()
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        store.subscribe_to_bus(bus)
        await _seed_root(store, bus, n=3)
        b1, _ = await store.create_branch("s1", "root", 4)
        b11, _ = await store.create_branch("s1", b1, 2)
        deleted = await store.delete_branch("s1", b1)
        assert set(deleted) == {b1, b11}
        ids = [b["branch_id"] for b in await store.list_branches("s1")]
        assert b1 not in ids and b11 not in ids
    asyncio.run(scenario())


def test_delete_session_soft():
    """删除整个会话：session+全部分支软删；list_sessions 过滤"""
    async def scenario():
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, "sys")
        await store.ensure_session("s2", {"user_id": "u1", "tenant_id": "t1"}, "sys")
        await store.delete_session("s1")
        ids = [s["session_id"] for s in await store.list_sessions("u1", "t1")]
        assert "s1" not in ids and "s2" in ids
        assert (await store.load_view("s1", "root", 6)) is None
        r = await store.gc_purge()
        assert r["sessions"] == 1
    asyncio.run(scenario())


# —— 命名 ——
def test_rename_session_and_branch():
    async def scenario():
        store = ContextStore(Path(tempfile.mkdtemp()) / "s.db")
        await store.ensure_session("s1", {"user_id": "u1", "tenant_id": "t1"}, "sys", name="旧名")
        b1, _ = await store.create_branch("s1", "root", 2, name="分支A")
        await store.rename_session("s1", "会话新名")
        await store.rename_branch("s1", b1, "分支B")
        sessions = await store.list_sessions("u1", "t1")
        assert sessions[0]["name"] == "会话新名"
        branches = await store.list_branches("s1")
        assert any(b["branch_id"] == b1 and b["name"] == "分支B" for b in branches)
        b2, _ = await store.create_branch("s1", "root", 2)
        branches = await store.list_branches("s1")
        assert any(b["branch_id"] == b2 and b["name"] for b in branches)
    asyncio.run(scenario())


# —— ChatSession 删除 / 重命名指令 ——
def test_session_command_branch_delete_switch_root():
    """删除当前分支（非 root）：软删后自动切回根分支"""
    async def scenario():
        bus = EventBus()
        deleted = []
        async def deleter(bid):
            deleted.append(bid)
            return [bid]
        async def loader(bid):
            return [SystemMessage(content="sys")]
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus,
            llm=None, tools=[], branch_id="b1", history=[SystemMessage(content="sys"), HumanMessage(content="h")],
            branch_deleter=deleter, branch_loader=loader,
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "branch_delete", "branch_id": "b1"}))
        await bus.flush()
        assert deleted == ["b1"]
        assert chat._branch_id == "root"
        assert [m.content for m in chat._messages] == ["sys"]
    asyncio.run(scenario())


def test_session_command_branch_delete_root_emits_deleted():
    """删除 root（整个会话）：发 SESSION_DELETED，入口可退出"""
    async def scenario():
        bus = EventBus()
        emitted = []
        bus.subscribe(EventType.SESSION_DELETED, lambda e: emitted.append(e.payload))
        async def deleter(bid):
            return [bid]
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus,
            llm=None, tools=[], branch_id="root", history=[SystemMessage(content="sys")],
            branch_deleter=deleter,
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "branch_delete", "branch_id": "root"}))
        await bus.flush()
        assert emitted and emitted[0]["branch_id"] == "root"
    asyncio.run(scenario())


def test_session_command_rename():
    """rename：当前 root → 会话名；非 root → 分支名"""
    async def scenario():
        bus = EventBus()
        renames = []
        async def sren(n):
            renames.append(("session", n))
        async def bren(bid, n):
            renames.append(("branch", bid, n))
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"}, bus=bus,
            llm=None, tools=[], branch_id="root", history=[SystemMessage(content="sys")],
            session_renamer=sren, branch_renamer=bren,
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "rename", "name": "会话名"}))
        await bus.flush()
        assert renames == [("session", "会话名")]
        renames.clear()
        chat._branch_id = "b1"
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "rename", "name": "分支名"}))
        await bus.flush()
        assert renames == [("branch", "b1", "分支名")]
    asyncio.run(scenario())
