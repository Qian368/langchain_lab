"""
配置模型层
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级完全匹配 config/ 下的 yaml 配置文件结构
"""

from pathlib import Path

from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


# ===================== RAG 领域子模型 =====================
class IdentConfig(EventedModel):
    """用户配置"""

    ...
