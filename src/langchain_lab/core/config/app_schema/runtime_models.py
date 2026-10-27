"""
配置模型层：运行时配置
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级匹配 config/runtime.yaml
（上下文压缩执行配置已迁移至 context_models.py / config/context.yaml）
"""

from pathlib import Path

from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


class AgentSettings(EventedModel):
    """Agent 运行时行为配置（原 agent_session/exp9 硬编码，已收敛到配置）"""

    max_llm_loop: int = Field(default=3, ge=1, description="单轮对话 LLM 最大迭代次数")
    max_tool_calls: int = Field(default=4, ge=1, description="单轮对话最大工具调用次数")
    session_db_path: Path = Field(default_factory=lambda: Path("./data/sessions.db"), description="会话持久化 SQLite 路径")
    default_system_prompt: str = Field(default="你是一个配合测试的助手，按照用户要求执行最小Agent循环任务", description="默认系统提示词")


class RuntimeSettings(EventedModel):
    """运行时配置（Agent 行为；上下文压缩已归 context 域）"""

    agent: AgentSettings = Field(default_factory=AgentSettings)
