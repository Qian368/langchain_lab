"""
配置模型层：运行时配置
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级匹配 config/runtime.yaml
"""

from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


class CompressionSettings(EventedModel):
    """上下文压缩配置"""

    enabled: bool = Field(default=True, description="是否启用上下文压缩")
    keep_recent_turns: int = Field(default=6, ge=1, description="压缩后保留的最近轮数")
    summarize_model: str | None = Field(default="default", description="压缩摘要用的 LLM 实例名")


class RuntimeSettings(EventedModel):
    """运行时配置（压缩等）"""

    compression: CompressionSettings = Field(default_factory=CompressionSettings)
