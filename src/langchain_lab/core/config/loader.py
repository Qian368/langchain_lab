from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from langchain_lab.core.config.app_config import AppConfig
from langchain_lab.core.config.app_schema.project_models import ProjectSettings
from langchain_lab.core.config.app_schema.rag_models import RagSettings


class ConfigLoader(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=[
            "config/rag.yaml",
            "config/project.yaml",
        ],
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    rag: RagSettings = Field(default_factory=RagSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 优先级：init参数 > YAML > 系统环境变量 > .env > 密钥文件
        # 你要“yaml < .env”，就把 yaml 放 env/dotenv 前面
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_config() -> AppConfig:
    config = ConfigLoader()
    return AppConfig(
        project=config.project,
        rag=config.rag,
    )
