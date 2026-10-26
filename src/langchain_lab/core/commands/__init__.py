"""控制面指令注册表：通信格式约定的统一编写处（全局单例字典，非配置文件）

- 指令集是通信格式约定，用 Python 常量承载，直接 import 复用，避免重复字面量
- 运行时、前端帮助、API 层都从 INSTRUCTION_SPECS 取定义
"""
from langchain_lab.core.commands.instruction_registry import (
    INSTRUCTION_SPECS,
    build_public_specs,
    get_spec,
    handler_for,
    is_known,
)

__all__ = [
    "INSTRUCTION_SPECS",
    "build_public_specs",
    "get_spec",
    "handler_for",
    "is_known",
]
