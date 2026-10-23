"""
配置模型层
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级完全匹配 config/ 下的 yaml 配置文件结构
"""

from pathlib import Path

from pydantic import Field, model_validator

from langchain_lab.core.config.env_expand import expand_model_fields
from langchain_lab.core.config.evented_model import EventedModel


# ===================== RAG 领域子模型 =====================
class RagApiSettings(EventedModel):
    """RAG 服务接口配置"""

    api_base: str = "http://127.0.0.1:8000"


class RagMongoSettings(EventedModel):
    """RAG MongoDB 存储配置"""

    user_uri: str = ""
    admin_uri: str = ""
    db_name: str = "knowledge_base"
    collection: str = "kb_documents"

    @model_validator(mode="after")
    def expand_env(self):
        """加载后展开 ${VAR}：yaml 占位 → .env 实际值（连接串不落明文）"""
        expand_model_fields(self, ("user_uri", "admin_uri"))
        return self


class RagSqliteSettings(EventedModel):
    """RAG SQLite 元数据配置"""

    meta_db_path: Path = Field(default_factory=lambda: Path("./data/sqlite/meta.db"))


class RagChromaSettings(EventedModel):
    """RAG Chroma 向量库配置"""

    path: Path = Field(default_factory=lambda: Path("./data/chroma"))
    collection: str = "kb_chunks"


class RagFileSettings(EventedModel):
    """RAG 文件访问配置"""

    allowed_base_dirs: list[Path] = Field(default_factory=list)
    upload_dir: Path = Field(default_factory=lambda: Path("./data/uploads"))


class RagEmbeddingSettings(EventedModel):
    """RAG 嵌入模型配置"""

    model: str = Field(default="../models/bge-small-zh-v1.5", description="嵌入模型路径或名称")


class RagSettings(EventedModel):
    """RAG 领域根配置
    包含 RAG 全链路所有组件配置
    """

    api: RagApiSettings = Field(default_factory=RagApiSettings)
    mongo: RagMongoSettings = Field(default_factory=RagMongoSettings)
    sqlite: RagSqliteSettings = Field(default_factory=RagSqliteSettings)
    chroma: RagChromaSettings = Field(default_factory=RagChromaSettings)
    file: RagFileSettings = Field(default_factory=RagFileSettings)
    embedding: RagEmbeddingSettings = Field(default_factory=RagEmbeddingSettings)
