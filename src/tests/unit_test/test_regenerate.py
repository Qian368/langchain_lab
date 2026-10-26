"""regenerate 重跑 + 前端过滤测试：
- build_display_context 无摘要分支也过滤 system（前端过滤 bug 修复）
- 条件化注解：仅当该轮有工具调用时注入 meta='rerun'（临时副本，不进 _messages/不归档）
- 无工具轮不注入注解；重跑只生成新回答、复用工具结果
- 新回答增量归档；旧回答发 MESSAGE_SUPERSEDED 由 store 软删（load_view 只见新回答）
- 全部走事件总线，不连真实 LLM/工具
"""
import asyncio
import os
import tempfile

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)
from langchain_core.tools import tool

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.context.views import build_display_context
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeLlmWithTools:
    def __init__(self, chunks, seen):
        self._chunks = chunks
        self._seen = seen

    def stream(self, messages):
        self._seen.append(messages)  # 记录 LLM 实际看到的上下文（验证条件化注解是否注入）
        return _FakeStream(self._chunks)


class _FakeLlm:
    def __init__(self, chunks=None, seen=None):
        self._chunks = chunks or [AIMessageChunk(content="ok")]
        self._seen = seen if seen is not None else []

    def bind_tools(self, tools):
        return _FakeLlmWithTools(self._chunks, self._seen)


@tool
def _fake_echo(x: int) -> str:
    return f"v{x}"


# —— 前端过滤（build_display_context 无摘要分支过滤 system）——
def test_display_context_filters_system_no_summary():
    """无摘要分支：不发 system prompt / meta 注入消息（前端过滤 bug 修复验证）"""
    msgs = [SystemMessage("sys"), HumanMessage("h0"), AIMessage("a0")]
    ctx = build_display_context(msgs, None, 6)
    assert all(item["kind"] == "message" for item in ctx)
    assert all(not isinstance(item["data"], SystemMessage) for item in ctx)
    assert [item["data"].content for item in ctx] == ["h0", "a0"]  # system 被过滤


def test_display_context_filters_system_with_summary():
    """有摘要分支：collapse 占位 + 最近轮真实消息（不含 system / 摘要 / 注入）"""
    msgs = [
        SystemMessage("sys"),
        HumanMessage("[对话历史摘要]\nsum", additional_kwargs={"meta": "summary"}),
        HumanMessage("h0"), AIMessage("a0"),
        HumanMessage("h1"), AIMessage("a1"),
        HumanMessage("h2"), AIMessage("a2"),
    ]
    ctx = build_display_context(msgs, "sum", 2)  # keep=2
    kinds = [item["kind"] for item in ctx]
    assert kinds == ["collapse", "message", "message", "message", "message"]  # 早期 1 轮折叠
    recent = [item["data"].content for item in ctx if item["kind"] == "message"]
    assert recent == ["h1", "a1", "h2", "a2"]  # 不含 system / 摘要
    assert ctx[0]["summary"] == "sum"


# —— regenerate：无工具轮（不注入注解）——
def test_regenerate_no_tools_no_annotation():
    """无工具调用轮：重跑不注入 meta=rerun 注解，只重新生成回答，旧回答作废"""
    async def scenario():
        bus = EventBus()
        seen = []
        superseded = []
        bus.subscribe(EventType.MESSAGE_SUPERSEDED, lambda e: superseded.append(e.payload))
        msgs = [SystemMessage("sys"), HumanMessage("h0"), AIMessage("a0"), HumanMessage("h1"), AIMessage("旧回答")]
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=_FakeLlm([AIMessageChunk(content="新回答")], seen), tools=[], history=msgs,
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "regenerate"}))
        await bus.flush()
        # 注解不注入（无工具轮）
        assert not any(m.additional_kwargs.get("meta") == "rerun" for m in chat._messages)
        assert not any(m.additional_kwargs.get("meta") == "rerun" for batch in seen for m in batch)
        # 旧回答被替换为新回答（视图干净）
        assert chat._messages[-1].content == "新回答"
        assert not any(isinstance(m, AIMessage) and m.content == "旧回答" for m in chat._messages)
        # 作废事件带旧回答 content
        assert superseded and superseded[0]["superseded_content"] == "旧回答"

    asyncio.run(scenario())


# —— regenerate：有工具轮（条件化注入注解，临时不进 _messages）——
def test_regenerate_with_tools_conditional_annotation():
    """有工具调用轮：重跑注入 meta=rerun 注解（LLM 可见），复用工具结果，新回答归档，旧回答作废"""
    async def scenario():
        bus = EventBus()
        seen = []
        archived = []
        bus.subscribe(EventType.MESSAGES_ARCHIVED, lambda e: archived.append(e.payload))
        msgs = [
            SystemMessage("sys"), HumanMessage("h0"), AIMessage("a0"),
            HumanMessage("h1"),
            AIMessage(content="", tool_calls=[{"name": "t", "id": "c1", "args": {}}]),
            ToolMessage(content='{"r":1}', tool_call_id="c1", name="t"),
            AIMessage("旧回答"),
        ]
        chat = ChatSession(
            session_id="s1", session_ctx={"user_id": "u1", "tenant_id": "t1"},
            bus=bus, llm=_FakeLlm([AIMessageChunk(content="新回答")], seen), tools=[], history=msgs,
        )
        bus.publish(Event(EventType.SESSION_COMMAND, "s1", {"action": "regenerate"}))
        await bus.flush()
        # 注解【确实注入给 LLM 看】（副本），但【不进 _messages】（不持久化）
        assert any(m.additional_kwargs.get("meta") == "rerun" for batch in seen for m in batch)
        assert not any(m.additional_kwargs.get("meta") == "rerun" for m in chat._messages)
        # 复用工具结果 + 新回答
        assert any(isinstance(m, ToolMessage) for m in chat._messages)  # 工具结果保留为事实
        assert chat._messages[-1].content == "新回答"
        # 新回答增量归档（不含已归档的 h1/工具结果/旧回答）
        assert archived
        archived_msgs = archived[0]["messages"]
        assert any(d.get("data", {}).get("content") == "新回答" for d in archived_msgs)
        assert not any(d.get("data", {}).get("content") == "旧回答" for d in archived_msgs)

    asyncio.run(scenario())


# —— store：MESSAGE_SUPERSEDED 软删旧回答（load_view 只见新回答）——
def test_supersede_soft_deletes_old_answer():
    """作废旧回答软删：archive 里匹配的 AI 消息置 deleted=1，重建/恢复视图不再出现"""
    async def scenario():
        with tempfile.TemporaryDirectory() as td:
            store = ContextStore(os.path.join(td, "t.db"))
            sid = "s1"
            await store.ensure_session(sid, {"user_id": "u1", "tenant_id": "t1"}, system_prompt="sys", name="n")
            store._append_archive_sync(sid, "root", "evt_h", [messages_to_dict(HumanMessage(content="h1"))])
            store._append_archive_sync(sid, "root", "evt_a", [messages_to_dict(AIMessage(content="旧回答"))])
            # 重跑产生新回答归档后，软删旧回答
            store._append_archive_sync(sid, "root", "evt_a2", [messages_to_dict(AIMessage(content="新回答"))])
            store._supersede_sync(sid, "root", "旧回答")
            full = await store.load_full_history(sid, "root")  # 只读 deleted=0
            contents = [getattr(m, "content", None) for m in full]
            assert "新回答" in contents and "旧回答" not in contents  # 旧回答被软删，新回答在
            # 恢复视图（load_view）同样只见新回答
            view = await store.load_view(sid, "root", 6)
            view_contents = [getattr(m, "content", None) for m in view]
            assert "新回答" in view_contents and "旧回答" not in view_contents

    asyncio.run(scenario())
