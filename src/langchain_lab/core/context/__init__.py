"""core/context：会话上下文管理独立模块

- context_store：持久化订阅者（SQLite，事件溯源落库；入口用它加载历史/分支）
- compressor：上下文压缩摘要（投影策略，运行时调用）
- views：双视图投影纯函数（LLM 视图 / 前端视图）

职责边界：投影策略由运行时调用；持久化 I/O 由订阅者承担，运行时不引用 store。
"""

from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.context.views import build_display_context, build_llm_context

__all__ = [
    "ContextStore",
    "SessionCompressor",
    "build_display_context",
    "build_llm_context",
]
