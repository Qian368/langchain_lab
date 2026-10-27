"""事件通道抽象（面向 MQ 过渡）+ 内存实现

分层（Port / Adapter）：
- EventChannel（Protocol，业务依赖的唯一接口）：subscribe / publish / flush / close
- EventBus（Adapter，当前默认=内存实现）：同步派发 + async handler 后台调度；定向(session_id)+通配(None)路由
- MQEventChannel（未来，同一 Port）：publish=序列化入队，partition key=session_id，at-least-once + event_id 幂等

路由语义：publish 时投递给「该类型通配 handler」+「该类型匹配 session_id 的定向 handler」。
  - 会话处理（ChatSession）定向订阅自己 session_id → O(1)，不被其他会话事件触发
  - 持久化/渲染/审计（ContextStore/渲染器）通配订阅 → 收所有，数量恒定
  1000 会话时一次发布只有目标会话 + 几个通配订阅者被调用，不再是 O(N) 广播。

MQ 过渡契约：publish 不阻塞业务（内存同步派发 / MQ 异步入队）；flush 确认投递/消费完成；
handler 必须可重入幂等（at-least-once 会重复投递，event_id 去重）；partition key=session_id 保证会话内有序。

订阅签名（兼容新旧写法）：
- 旧式通配：subscribe(event_type, handler)
- 新式定向/通配：subscribe(handler, event_type, session_id=session_id or None)
"""

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from typing import Protocol

from langchain_lab.core.bus.events import Event, EventType

# 同步 handler 直接返回；异步 handler 返回协程
Handler = Callable[[Event], None] | Callable[[Event], Awaitable[None]]


class EventChannel(Protocol):
    """事件通道端口。业务只依赖此接口；底层换 MQ 时订阅方代码零改动"""

    def subscribe(self, first, second=None, *, session_id: str | None = None) -> None: ...

    def publish(self, event: Event) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


class EventBus:
    """内存事件通道（EventChannel 的内存 Adapter）。

    - 按 (event_type, session_key) 路由；session_key=None 通配，str 定向
    - publish 同步派发：该类型通配 + 该会话定向的所有 handler；async handler 用 create_task 后台调度
    - flush 等待所有后台 async handler 完成（持久化落库 / MQ 消费确认语义）
    - close 释放通道：flush + 清空订阅（停机语义）
    """

    def __init__(self) -> None:
        self._handlers: dict[tuple[EventType, str | None], list[Handler]] = {}
        self._pending: list[asyncio.Task] = []  # 跟踪后台 async handler，flush 用

    def subscribe(
        self,
        first: EventType | Handler,
        second: Handler | EventType | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        """订阅。兼容两种写法：
        - 旧式通配：subscribe(event_type, handler)
        - 新式定向/通配：subscribe(handler, event_type, session_id=session_id or None)
        """
        if isinstance(first, EventType):
            event_type, handler = first, second
        else:
            handler, event_type = first, second
        if not isinstance(event_type, EventType) or not callable(handler):
            raise TypeError("subscribe: 期望 (event_type, handler) 或 (handler, event_type)")
        key = (event_type, session_id)
        self._handlers.setdefault(key, []).append(handler)

    def unsubscribe(
        self,
        first: EventType | Handler,
        second: Handler | EventType | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        """取消订阅，签名同 subscribe。调用方需保证已订阅过，否则抛 ValueError"""
        if isinstance(first, EventType):
            event_type, handler = first, second
        else:
            handler, event_type = first, second
        if not isinstance(event_type, EventType) or not callable(handler):
            raise TypeError("unsubscribe: 期望 (event_type, handler) 或 (handler, event_type)")
        key = (event_type, session_id)
        self._handlers.get(key, []).remove(handler)

    def publish(self, event: Event) -> None:
        """同步派发：该类型通配 + 该会话定向 的所有 handler；async handler 后台 create_task（不阻塞流式 chunk）。"""
        wildcard = self._handlers.get((event.type, None), ())
        directed = self._handlers.get((event.type, event.session_id), ())
        for handler in (*wildcard, *directed):
            result = handler(event)
            if inspect.iscoroutine(result):
                task = asyncio.create_task(result)
                self._pending.append(task)
                task.add_done_callback(self._discard_pending)

    async def flush(self) -> None:
        """等待所有后台 async handler 完成。TurnComplete / close 时调用，保证落库后再继续。

        排除当前正在执行的 task：handler 内部（如 _handle_turn 收尾）再调 flush 时，
        若 gather 含自身 task 会自包含死锁（task await 自己）；排除后只等其子任务。
        """
        current = asyncio.current_task()
        pending = [t for t in self._pending if t is not current]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        """释放通道：等后台任务完成 + 清空订阅（停机语义）"""
        await self.flush()
        self._handlers.clear()

    def _discard_pending(self, task: asyncio.Task) -> None:
        """任务完成后的清理：从 _pending 移除，避免泄漏；并发移除时容忍 ValueError"""
        with contextlib.suppress(ValueError):
            self._pending.remove(task)
