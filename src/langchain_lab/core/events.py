"""事件模型：事件类型枚举 + 统一事件载体（面向网络服务化）

事件分级：
- 输入驱动：外部适配器 publish，会话订阅触发一轮（USER_MESSAGE_SUBMITTED / COMPRESS_REQUESTED）
- 会话生命周期：SESSION_*（创建/恢复/关闭/系统提示）
- 轮次边界：AGENT_TURN_START / DONE（一次输入的处理开始/结束，供驱动等待+前端进度）
- 过程事件：逐片/即时，给展示（流式打字、工具进度、用户回显）
- 状态快照：SESSION_STATE_UPDATED 带完整消息快照（事件溯源，持久化订阅者落库）
- 压缩：COMPRESS_REQUESTED（触发，force/keep）+ CONTEXT_COMPRESSED（结果，forced 标记）
- 错误：ERROR_OCCURRED
"""
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Any


class EventType(StrEnum):
    """全部事件类型"""

    # —— 输入驱动（网络服务化：输入适配器发布，会话订阅触发）——
    USER_MESSAGE_SUBMITTED = "user_message_submitted"
    COMPRESS_REQUESTED = "compress_requested"  # 触发压缩：force=False 自动(每轮后置)，force=True 前端手动提前
    # —— 会话生命周期 ——
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    SESSION_CLOSED = "session_closed"
    SESSION_NOTICE = "session_notice"  # 会话内系统提示（分支提示等），展示层渲染
    # —— 轮次边界（一次 USER_MESSAGE 的处理开始/结束）——
    AGENT_TURN_START = "agent_turn_start"
    AGENT_TURN_DONE = "agent_turn_done"
    # —— 过程事件：逐片/即时，给展示（流式打字、工具进度）——
    HUMAN_MESSAGE_RECEIVED = "human_message_received"  # 会话追加用户消息后的回显
    STREAM_CHUNK = "stream_chunk"
    TOOL_CALL_START = "tool_call_start"
    TOOL_RESULT = "tool_result"
    # —— 完整事件：累积完成，给持久化/完整语义 ——
    ASSISTANT_MESSAGE_EMITTED = "assistant_message_emitted"
    TURN_COMPLETE = "turn_complete"
    # —— 状态快照（事件溯源核心：会话每轮发布完整快照，持久化订阅者落库）——
    SESSION_STATE_UPDATED = "session_state_updated"
    # —— 上下文压缩：触发请求 + 完成结果（forced 标记供前端区分手动/自动）——
    CONTEXT_COMPRESSED = "context_compressed"
    # —— 错误（运行时异常 publish，日志/前端/持久化处理）——
    ERROR_OCCURRED = "error_occurred"


@dataclass(slots=True)
class Event:
    """统一事件载体。session_id 在事件里，订阅者按需过滤（总线全局，天然支持多会话隔离）"""

    type: EventType
    session_id: str
    payload: Any = None
    timestamp: float = field(default_factory=time)
