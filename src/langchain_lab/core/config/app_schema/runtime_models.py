"""
配置模型层：运行时配置
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级匹配 config/runtime.yaml
"""

from pathlib import Path

from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


class CompressionSettings(EventedModel):
    """上下文压缩配置"""

    enabled: bool = Field(default=True, description="是否启用上下文压缩")
    keep_recent_turns: int = Field(default=6, ge=1, description="压缩后保留的最近轮数")
    summarize_model: str | None = Field(default="default", description="压缩摘要用的 LLM 实例名")


class AgentSettings(EventedModel):
    """Agent 运行时行为配置（原 agent_session/exp9 硬编码，已收敛到配置）"""

    max_llm_loop: int = Field(default=3, ge=1, description="单轮对话 LLM 最大迭代次数")
    max_tool_calls: int = Field(default=4, ge=1, description="单轮对话最大工具调用次数")
    session_db_path: Path = Field(
        default_factory=lambda: Path("./data/sessions.db"), description="会话持久化 SQLite 路径"
    )
    default_system_prompt: str = Field(
        default="你是一个配合测试的助手，按照用户要求执行最小Agent循环任务", description="默认系统提示词"
    )


class RuntimeSettings(EventedModel):
    """运行时配置（压缩、Agent 行为等）"""

    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
