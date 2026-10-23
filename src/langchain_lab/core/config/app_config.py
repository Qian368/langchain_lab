"""
配置模型层
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级完全匹配 config/ 下的 yaml 配置文件结构
"""

from pydantic import Field

from langchain_lab.core.config.app_schema.mcp_models import MCPSettings
from langchain_lab.core.config.app_schema.project_models import ProjectSettings
from langchain_lab.core.config.app_schema.rag_models import RagSettings
from langchain_lab.core.config.evented_model import EventedModel
from langchain_lab.core.llm.llm_schema import LLMConfig


# ===================== 全局根配置 =====================
class AppConfig(EventedModel):
    """
    全局根配置入口
    所有配置领域的统一入口，支持整体/字段级/子模型级订阅
    """

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    llm: LLMConfig = Field(default_factory=LLMConfig)
