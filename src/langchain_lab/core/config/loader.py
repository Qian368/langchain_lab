from collections.abc import Sequence

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from langchain_lab.core.config.app_config import AppConfig
from langchain_lab.core.config.app_schema.mcp_models import MCPSettings
from langchain_lab.core.config.app_schema.project_models import ProjectSettings
from langchain_lab.core.config.app_schema.rag_models import RagSettings
from langchain_lab.core.config.app_schema.runtime_models import RuntimeSettings
from langchain_lab.core.llm.llm_schema import LLMConfig


class ConfigLoader(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=[
            "config/rag.yaml",
            "config/project.yaml",
            "config/mcp.yaml",
            "config/llm.yaml",
            "config/runtime.yaml",
        ],
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        raw = settings_cls.model_config.get("yaml_file")

        # 👉 窄化类型：判断是不是序列（排除 str，因为 str 也是 Sequence，但是是单个文件名）
        if raw is None:
            file_list: list[str] = []
        elif isinstance(raw, str):
            file_list = [raw]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            file_list = [str(item) for item in raw]
        else:
            file_list = [str(raw)]

        yaml_source = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=file_list,
            yaml_file_encoding="utf-8",
        )

        # 优先级：init参数 > YAML > 系统环境变量 > .env > 密钥文件
        return (
            init_settings,
            yaml_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_config() -> AppConfig:
    config = ConfigLoader()
    return AppConfig(
        project=config.project,
        rag=config.rag,
        mcp=config.mcp,
        llm=config.llm,
        runtime=config.runtime,
    )
