"""policy 域（预留点位，暂不实现具体规则）

规划（架构契约 docs/architecture.md §7）：
- 工具管理 / 权限管理 / human-in-loop / 沙箱确认 统一收编到本域。
- PolicyEngine 在编排层（AgentRuntime）工具执行前被调用：判定 allow / 需确认 / 拒绝。
- 命中"需确认" → 编排层 await ApprovalGate 等人类 approve/reject（见 approval.py）。
- 本文件只定边界（类型 / 接口），不实现规则；规则后续在 policy/rules 子域落地。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PolicyContext:
    """策略判定输入（工具调用、会话身份、动作类型）"""

    action: str  # 动作类型：tool_call / llm_probe / sandbox_write ...
    session_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """策略判定结果"""

    allow: bool = False  # 放行
    require_approval: bool = False  # 需人类确认（命中 HITL）
    reason: str = ""  # 判定依据（审计留痕）


class PolicyEngine(Protocol):
    """策略引擎接口：判定一次动作是否放行 / 需确认 / 拒绝。

    预留：由具体规则实现（工具白名单、权限等级、敏感操作、沙箱边界）。
    """

    def check(self, ctx: PolicyContext) -> PolicyDecision:
        ...
