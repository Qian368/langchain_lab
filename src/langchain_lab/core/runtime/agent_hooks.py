"""Agent 运行时钩子（控制型，同步可干预；默认 None，不侵入主逻辑）

与事件总线分工：
- 事件总线 = 观察（异步订阅者：前端/日志/持久化消费流式、快照、压缩、错误等事件）
- AgentHooks = 干预（校验/改写/拒绝，可同步拦截）

仅保留 3 个核心控制钩子，其余过滤/处理下放到各组件实现。

返回值约定（统一）：
- 返回 None     → 放行，原样继续
- 返回非 None   → 改写/替换后继续（如 pre_user_message 返回改写文本、post_tool 返回替换 result）
- 钩子内 raise  → 拒绝/中断：冒泡为 ERROR_OCCURRED 并终止本轮

钩子可以是同步 Callable 或 async Callable，运行时通过 ChatSession._run_hook 统一 await。
"""
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class AgentHooks:
    """Agent 运行时控制型钩子集（默认 None；后续加钩子只在此加字段，不平铺散落）

    - pre_user_message(content: str) -> str | None   输入合法性校验/脱敏/改写（追加消息前）
    - post_llm(message) -> message | None           LLM 合法输出校验/替换（累积消息写入前）
    - post_tool(result: dict) -> dict | None        工具执行结果校验/脱敏/替换（构造 ToolMessage 前）
    """

    pre_user_message: Callable | None = None
    post_llm: Callable | None = None
    post_tool: Callable | None = None
