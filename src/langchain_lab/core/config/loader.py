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


class ConfigLoader(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=[
            "config/rag.yaml",
            "config/project.yaml",
            "config/mcp.yaml",
        ],
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

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
            # 是列表/序列，转成字符串路径
            file_list = [str(item) for item in raw]
        else:
            # raw 是单独的 Path / Traversable 单个对象
            file_list = [str(raw)]

        yaml_source = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=file_list,
            yaml_file_encoding="utf-8",
        )

        # 优先级：init参数 > YAML > 系统环境变量 > .env > 密钥文件
        # 如要“yaml < .env”，就把 yaml 放 env/dotenv 前面
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
    )
