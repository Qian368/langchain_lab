"""事件总线单测：同步派发 / async handler+flush / 类型隔离 / 多会话过滤"""

import asyncio

from langchain_lab.core.bus.event_bus import EventBus
from langchain_lab.core.bus.events import Event, EventType


def test_sync_handler_receives_event():
    """同步 handler 收到事件，session_id/type 正确"""
    bus = EventBus()
    got: list[Event] = []

    bus.subscribe(EventType.SESSION_CREATED, lambda e: got.append(e))
    bus.publish(Event(type=EventType.SESSION_CREATED, session_id="s1"))

    assert len(got) == 1
    assert got[0].session_id == "s1"
    assert got[0].type == EventType.SESSION_CREATED


def test_subscription_scoped_by_event_type():
    """只订阅的类型触发，未订阅类型不触发"""
    bus = EventBus()
    got: list[Event] = []

    bus.subscribe(EventType.STREAM_CHUNK, lambda e: got.append(e))
    bus.publish(Event(type=EventType.SESSION_CLOSED, session_id="s1"))

    assert got == []


def test_async_handler_flushed():
    """async handler 后台执行，publish 返回时未完成；flush 后完成（落库语义）"""

    async def scenario():
        bus = EventBus()
        got: list[str] = []

        async def handler(event: Event) -> None:
            await asyncio.sleep(0.01)
            got.append(event.session_id)

        bus.subscribe(EventType.TURN_COMPLETE, handler)
        bus.publish(Event(type=EventType.TURN_COMPLETE, session_id="s1"))

        assert got == []  # async handler 尚未完成
        await bus.flush()
        assert got == ["s1"]  # flush 后完成

    asyncio.run(scenario())


def test_publish_does_not_block_stream():
    """流式 chunk 的 publish 同步非阻塞（不 await 直接返回）"""
    bus = EventBus()
    bus.subscribe(EventType.STREAM_CHUNK, lambda e: None)
    bus.publish(Event(type=EventType.STREAM_CHUNK, session_id="s1", payload="hi"))  # 不抛即通过


def test_multi_session_filterable():
    """session_id 在事件里，订阅者可按会话过滤（多会话隔离基础）"""
    bus = EventBus()
    seen: list[str] = []

    def handler(event: Event) -> None:
        if event.session_id == "s2":
            seen.append(event.session_id)

    bus.subscribe(EventType.HUMAN_MESSAGE_RECEIVED, handler)
    bus.publish(Event(type=EventType.HUMAN_MESSAGE_RECEIVED, session_id="s1"))
    bus.publish(Event(type=EventType.HUMAN_MESSAGE_RECEIVED, session_id="s2"))

    assert seen == ["s2"]
