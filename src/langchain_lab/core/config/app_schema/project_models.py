"""
配置模型层
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级完全匹配 config/ 下的 yaml 配置文件结构
"""

from pathlib import Path

from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


# ===================== APP 全局子模型 =====================
class ProjectRootSettings(EventedModel):
    """应用项目路径配置"""

    project_root: Path = Field(default_factory=lambda: Path(".").resolve())
    data_dir: Path = Field(default_factory=lambda: Path("./data"))


class ProjectBackupSettings(EventedModel):
    """应用备份配置"""

    backup_dir: Path = Field(default_factory=lambda: Path("./data/backup"))


class ProjectSettings(EventedModel):
    """应用全局配置"""

    project: ProjectRootSettings = Field(default_factory=ProjectRootSettings)
    backup: ProjectBackupSettings = Field(default_factory=ProjectBackupSettings)
    session_ctx: dict = Field(default_factory=lambda: {"user_id": "u_001", "tenant_id": "t_001", "trace_id": "tr_abc123", "is_tenant_admin": False})
