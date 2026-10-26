"""ContextStore 分支单测：创建/快照加载/列出/摘要共享（不依赖 LLM）"""
import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_lab.core.context.context_store import ContextStore


def _run(coro):
    return asyncio.run(coro)


def _msgs(n: int):
    """system + n 轮(human+ai)"""
    msgs = [SystemMessage(content="sys")]
    for i in range(n):
        msgs += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    return msgs


def test_create_and_list_branch():
    store = ContextStore(Path(tempfile.mkdtemp()) / "b.db")
    _run(store.save("s1", _msgs(3), {"user_id": "u1", "tenant_id": "t1"}))
    new = _run(store.create_branch("s1", "root", branch_point_seq=2))

    branches = _run(store.list_branches("s1"))
    ids = [b["branch_id"] for b in branches]
    assert "root" in ids and new in ids
    root = [b for b in branches if b["branch_id"] == "root"][0]
    assert root["parent_id"] is None and root["branch_point_seq"] is None
    sub = [b for b in branches if b["branch_id"] == new][0]
    assert sub["parent_id"] == "root" and sub["branch_point_seq"] == 2


def test_load_branch_snapshot():
    store = ContextStore(Path(tempfile.mkdtemp()) / "b.db")
    _run(store.save("s1", _msgs(2), {"user_id": "u1", "tenant_id": "t1"}))
    new = _run(store.create_branch("s1", "root", branch_point_seq=1))
    # 子分支快照：分叉后 _messages = 拐点前(sys,h0,a0)
    _run(store.save("s1", _msgs(2)[:3], branch_id=new))
    loaded = _run(store.load_branch("s1", new))
    assert [m.content for m in loaded] == ["sys", "h0", "a0"]


def test_load_branch_unknown_raises():
    store = ContextStore(Path(tempfile.mkdtemp()) / "b.db")
    _run(store.save("s1", _msgs(1), {"user_id": "u1", "tenant_id": "t1"}))
    try:
        _run(store.load_branch("s1", "nope"))
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_summary_shared_across_branches():
    """摘要存会话级，各分支共享（规范 7.1）"""
    store = ContextStore(Path(tempfile.mkdtemp()) / "b.db")
    _run(store.save("s1", _msgs(6), {"user_id": "u1", "tenant_id": "t1"}, summary="摘要"))
    new = _run(store.create_branch("s1", "root", 3))
    _run(store.save("s1", _msgs(6)[:4], branch_id=new, summary="摘要"))
    assert _run(store.load_summary("s1")) == "摘要"
