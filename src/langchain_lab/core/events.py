"""事件模型：事件类型枚举 + 统一事件载体（两级粒度：过程事件给展示，完整事件给持久化）"""
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Any


class EventType(StrEnum):
    """全部事件类型。过程事件（给展示）与完整事件（给持久化）都在这"""

    # —— 会话生命周期 ——
    SESSION_CREATED = "session_created"
    SESSION_CLOSED = "session_closed"
    # —— 过程事件：逐片/即时，给展示（流式打字、工具进度）——
    HUMAN_MESSAGE_RECEIVED = "human_message_received"
    STREAM_CHUNK = "stream_chunk"
    TOOL_CALL_START = "tool_call_start"
    TOOL_RESULT = "tool_result"
    # —— 完整事件：累积完成，给持久化（落库）——
    ASSISTANT_MESSAGE_EMITTED = "assistant_message_emitted"
    TURN_COMPLETE = "turn_complete"
    # —— 上下文压缩：早期轮归档为摘要（持久化更新 summary + 展示提示）——
    CONTEXT_COMPRESSED = "context_compressed"


@dataclass(slots=True)
class Event:
    """统一事件载体。session_id 在事件里，订阅者按需过滤（总线全局，天然支持多会话隔离）"""

    type: EventType
    session_id: str
    payload: Any = None
    timestamp: float = field(default_factory=time)
