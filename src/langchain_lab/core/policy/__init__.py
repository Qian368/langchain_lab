"""policy 域（预留点位）：策略引擎 + 确认原语骨架

规划：工具管理/权限/human-in-loop/沙箱确认统一收编于此；
编排层工具执行前调用 PolicyEngine.check → 命中需确认则 await ApprovalGate。
规则暂不实现，本域只定边界（见 docs/architecture.md §7）。
"""
from langchain_lab.core.policy.approval import ApprovalGate, ApprovalRequest
from langchain_lab.core.policy.policy_engine import PolicyContext, PolicyDecision, PolicyEngine

__all__ = [
    "ApprovalGate",
    "ApprovalRequest",
    "PolicyContext",
    "PolicyDecision",
    "PolicyEngine",
]
