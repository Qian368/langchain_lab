"""事件模型：事件类型枚举 + 统一事件载体（面向网络服务化 + MQ 序列化契约）

事件分级：
- 输入驱动：外部适配器 publish，会话订阅触发（USER_MESSAGE_SUBMITTED / COMPRESS_REQUESTED / SESSION_COMMAND）
- 会话生命周期：SESSION_*（创建/恢复/关闭/系统提示/删除）
- 轮次边界：AGENT_TURN_START / DONE（一次输入的处理开始/结束，供驱动等待+前端进度）
- 过程事件：逐片/即时，给展示（流式打字、工具进度、用户回显）
- 完整事件：累积完成，给持久化/完整语义（ASSISTANT_MESSAGE_EMITTED / TURN_COMPLETE）
- 归档：MESSAGES_ARCHIVED（每轮新增真实消息增量落库）+ SUMMARY_ARCHIVED（摘要按分支归档）
- 作废：MESSAGE_SUPERSEDED（重跑作废旧回答，store 软删 archive 中对应行）
- 状态快照：SESSION_STATE_UPDATED（当前视图快照，事件溯源）
- 压缩：COMPRESS_REQUESTED（触发，force/keep）+ CONTEXT_COMPRESSED（结果，forced 标记）
- 错误：ERROR_OCCURRED

面向 MQ 序列化契约（跨进程传递前提）：
- payload 必须 JSON 可序列化（dict/str/int/float/bool/None/list），禁止任意对象
- event_id：源头唯一标识，MQ 幂等去重（at-least-once 下重复投递去重，不乱序）
- schema_version：payload 结构版本，MQ 跨版本消费
"""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Any
from uuid import uuid4


class EventType(StrEnum):
    """全部事件类型"""

    # —— 输入驱动（网络服务化：输入适配器解析后发布，会话订阅响应）——
    USER_MESSAGE_SUBMITTED = "user_message_submitted"  # 普通用户消息（适配器已确认非指令）
    COMPRESS_REQUESTED = "compress_requested"  # 触发压缩：force=False 自动(每轮后置)，force=True 前端手动提前
    SESSION_COMMAND = "session_command"  # 控制指令（适配器解析+校验后发布；运行时按 action 响应，不解析原始文本）
    # —— 会话生命周期 ——
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    SESSION_CLOSED = "session_closed"
    SESSION_DELETED = "session_deleted"  # 会话（根分支）被删除，入口监听后退出循环
    SESSION_NOTICE = "session_notice"  # 会话内系统提示（分支提示等），展示层渲染
    # —— 轮次边界（一次输入的处理开始/结束）——
    AGENT_TURN_START = "agent_turn_start"
    AGENT_TURN_DONE = "agent_turn_done"
    # —— 过程事件：逐片/即时，给展示（流式打字、工具进度）——
    HUMAN_MESSAGE_RECEIVED = "human_message_received"  # 会话追加用户消息后的回显
    STREAM_CHUNK = "stream_chunk"
    FINISH_STREAM_CHUNK = "finish_stream_chunk"
    TOOL_CALL_START = "tool_call_start"
    TOOL_RESULT = "tool_result"
    # —— 完整事件：累积完成，给持久化/完整语义 ——
    ASSISTANT_MESSAGE_EMITTED = "assistant_message_emitted"
    TURN_COMPLETE = "turn_complete"
    # —— 归档：每轮增量落库 + 摘要消息按分支归档（append-only）——
    MESSAGES_ARCHIVED = "messages_archived"
    SUMMARY_ARCHIVED = "summary_archived"  # 摘要消息归档（带 branch_id，落 archive kind=summary；由 ChatSession 携带当前分支发出）
    # —— 作废：重跑作废旧回答（store 软删 archive 中该分支最后一条匹配的 AI 消息）——
    MESSAGE_SUPERSEDED = "message_superseded"  # payload {branch_id, superseded_content}
    # —— 状态快照（当前视图，事件溯源核心）——
    SESSION_STATE_UPDATED = "session_state_updated"
    # —— 上下文压缩：触发请求 + 完成结果（更新会话级 summary 文本）——
    CONTEXT_COMPRESSED = "context_compressed"
    # —— 错误（运行时异常 publish，日志/前端/持久化处理）——
    ERROR_OCCURRED = "error_occurred"


@dataclass(slots=True)
class Event:
    """统一事件载体。session_id 在事件里，通道据此定向投递（多会话隔离，O(1) 路由）

    MQ 序列化契约：payload 必须 JSON 可序列化；event_id 幂等去重；schema_version 跨版本消费。
    """

    type: EventType
    session_id: str
    payload: Any = None
    event_id: str = field(default_factory=lambda: uuid4().hex)  # 源头唯一标识，MQ 幂等去重基础
    timestamp: float = field(default_factory=time)
    schema_version: int = 1  # payload 结构版本（MQ 跨版本消费）

    def serialize(self) -> str:
        """序列化为 JSON（MQ 投递 / 持久化）。payload 非 JSON 可序列化时报错"""
        return json.dumps(
            {
                "event_id": self.event_id,
                "type": str(self.type),
                "session_id": self.session_id,
                "payload": self.payload,
                "timestamp": self.timestamp,
                "schema_version": self.schema_version,
            },
            ensure_ascii=False,
        )
