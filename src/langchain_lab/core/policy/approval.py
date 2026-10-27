"""approval 确认原语骨架（预留点位，暂不实现策略）

human-in-loop / 权限确认：Agent 过 policy 门命中"需确认"后，编排层 await ApprovalGate 的响应。
- approval_id 关联：前端 approve/reject 请求带 approval_id，编排层匹配 Future 完成等待。
- 不经 EventBus（避免把单向事件总线污染成 RPC）；确认状态/结果作为领域事件走 EventBus 通知+审计。
- 本文件只定原语边界，不接运行时（AgentRuntime 的 policy 门控后续在此之上落地）。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field


@dataclass
class ApprovalRequest:
    """一次待确认的请求（工具执行 / 敏感动作）"""

    approval_id: str
    action: str
    summary: str  # 展示给人类的确认摘要
    payload: dict = field(default_factory=dict)
    created_at: float = 0.0


class ApprovalGate:
    """确认原语：以 approval_id + asyncio.Future 提供"请求-确认"的阻塞等待与匹配。

    预留：具体策略（谁能批、超时、级联拒绝）后续在 policy/rules 落地。
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future] = {}
        self._requests: dict[str, ApprovalRequest] = {}

    async def request(self, *, action: str, summary: str, payload: dict | None = None) -> ApprovalRequest:
        """发起一次确认请求，登记 pending；返回可等待的请求句柄（等待 approve/reject 完成）"""
        req = ApprovalRequest(
            approval_id=str(uuid.uuid4()),
            action=action,
            summary=summary,
            payload=payload or {},
        )
        self._requests[req.approval_id] = req
        self._pending[req.approval_id] = asyncio.get_running_loop().create_future()
        return req

    async def wait(self, approval_id: str) -> dict:
        """阻塞等待该确认请求被 approve/reject；返回响应载荷"""
        future = self._pending[approval_id]
        return await future

    def respond(self, approval_id: str, *, approved: bool, note: str = "") -> None:
        """人类响应：approve / reject，完成对应 Future（幂等，重复响应忽略）"""
        future = self._pending.pop(approval_id, None)
        if future is not None and not future.done():
            future.set_result({"approval_id": approval_id, "approved": approved, "note": note})

    def cancel_all(self) -> None:
        """清空所有 pending（停机语义）"""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("ApprovalGate closed"))
        self._pending.clear()
