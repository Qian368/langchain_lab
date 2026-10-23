"""事件总线：同步派发 + async handler 后台调度。发布永不阻塞，轮次末 flush 落库。

约束：publish 必须在运行中的 event loop 内调用（ChatSession.run 是 async 上下文，天然满足）。
async handler 由 create_task 后台执行、调用即返回；需要保证落库时，在轮次结束/会话关闭处 await flush()。
"""
import asyncio
import inspect
from collections.abc import Awaitable, Callable

from langchain_lab.core.events import Event, EventType

# 同步 handler：直接返回；异步 handler：返回协程
Handler = Callable[[Event], None] | Callable[[Event], Awaitable[None]]


class EventBus:
    """按事件类型注册订阅者；发布时同步派发，async handler 后台调度"""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}
        self._pending: list[asyncio.Task] = []  # 跟踪后台 async handler，flush 用

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """订阅某类事件。可多个 handler 订阅同一类型"""
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """取消订阅。调用方需保证该 handler 已订阅过，否则抛 ValueError"""
        self._handlers.get(event_type, []).remove(handler)

    def publish(self, event: Event) -> None:
        """
        同步派发：同步 handler 立即执行；async handler 用 create_task 后台跑（不阻塞流式 chunk）。
        需在运行中的 event loop 内调用（async 上下文）。
        """
        for handler in self._handlers.get(event.type, []):
            result = handler(event)
            if inspect.iscoroutine(result):
                task = asyncio.create_task(result)
                self._pending.append(task)
                task.add_done_callback(self._discard_pending)

    async def flush(self) -> None:
        """等待所有后台 async handler 完成（持久化写库用）。TurnComplete / close 时调用，保证落库后再继续"""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    def _discard_pending(self, task: asyncio.Task) -> None:
        """任务完成后的清理：从 _pending 移除，避免泄漏；并发移除时容忍 ValueError"""
        try:
            self._pending.remove(task)
        except ValueError:
            pass
