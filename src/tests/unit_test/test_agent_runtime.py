"""AgentRuntime 宿主测试：实例池 + create/resume 生命周期 + 操作面事件（mock 全局配置，不连真实 LLM）

- create_session → 实例池注册、返回会话
- resume_session → 从持久化恢复（同 db 新 runtime 重建实例）
- list_sessions → 按身份列出
- send_message / handle_command → 走 runtime 操作面发布事件（消息面/指令面分开）
- 不调 close()（避开停机语义的 MCP clear_all），靠临时目录隔离
"""
import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from langchain_lab.core.agent_runtime import AgentRuntime
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import EventType

_MOCK_CFG = SimpleNamespace(
    project=SimpleNamespace(session_ctx={"user_id": "u1", "tenant_id": "t1"}),
    runtime=SimpleNamespace(
        agent=SimpleNamespace(session_db_path="", default_system_prompt="sys", max_llm_loop=3, max_tool_calls=4),
        compression=SimpleNamespace(enabled=False, keep_recent_turns=6, summarize_model=None, summarize_prompt=""),
    ),
    llm=SimpleNamespace(default_instance="x", instances={}),
)


def _new_runtime(td):
    return AgentRuntime(db_path=os.path.join(td, "t.db"))


async def _scenario(fn):
    with tempfile.TemporaryDirectory() as td, patch("langchain_lab.core.agent_runtime.get_config", return_value=_MOCK_CFG):
        await fn(td)


def test_create_session_registers_pool():
    """create_session：返回会话并注册进实例池"""
    async def fn(td):
        rt = _new_runtime(td)
        chat = await rt.create_session(name="对话A")
        assert rt.get(chat._session_id) is chat
        assert chat._session_id
        assert len(rt._sessions) == 1
    asyncio.run(_scenario(fn))


def test_resume_session_from_persist():
    """resume_session：同 db 新 runtime 从持久化恢复实例（KV 视图重建）"""
    async def fn(td):
        rt1 = _new_runtime(td)
        chat = await rt1.create_session(name="对话A")
        sid = chat._session_id
        # 新 runtime 同一 db：列表可见 + 可恢复
        rt2 = _new_runtime(td)
        sessions = await rt2.list_sessions()
        assert any(s["session_id"] == sid for s in sessions)
        chat2 = await rt2.resume_session(sid)
        assert chat2 is not None and chat2._session_id == sid
        assert len(chat2._messages) >= 1  # 恢复视图含 system
    asyncio.run(_scenario(fn))


def test_list_sessions_by_identity():
    async def fn(td):
        rt = _new_runtime(td)
        await rt.create_session(name="A")
        await rt.create_session(name="B")
        sessions = await rt.list_sessions()
        assert len(sessions) == 2
        assert {s["name"] for s in sessions} == {"A", "B"}
    asyncio.run(_scenario(fn))


def test_send_message_event():
    """消息面：send_message 发布 USER_MESSAGE_SUBMITTED（走 runtime 操作面，非直接 new ChatSession）"""
    async def fn(td):
        rt = _new_runtime(td)
        chat = await rt.create_session()
        sid = chat._session_id
        seen = []
        rt.bus.subscribe(EventType.USER_MESSAGE_SUBMITTED, lambda e: seen.append(e.payload))
        rt.send_message(sid, "hi")
        await rt.bus.flush()
        assert seen and seen[0]["content"] == "hi"
    asyncio.run(_scenario(fn))


def test_handle_command_event():
    """指令面：handle_command 发布 SESSION_COMMAND（action 语义化，运行时只响应）"""
    async def fn(td):
        rt = _new_runtime(td)
        chat = await rt.create_session()
        sid = chat._session_id
        seen = []
        rt.bus.subscribe(EventType.SESSION_COMMAND, lambda e: seen.append(e.payload))
        rt.handle_command(sid, "rename", {"name": "改名"})
        await rt.bus.flush()
        assert seen and seen[0]["action"] == "rename" and seen[0]["name"] == "改名"
    asyncio.run(_scenario(fn))


def test_unregistered_session_raises():
    """操作面校验：未 create/resume 的会话发消息应报错"""
    async def fn(td):
        rt = _new_runtime(td)
        try:
            rt.send_message("nope", "hi")
            assert False, "应抛出 KeyError"
        except KeyError:
            pass
    asyncio.run(_scenario(fn))
