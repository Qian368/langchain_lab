"""
配置模型层
基于 EventedModel 的强类型嵌套配置，支持字段级变更事件
层级完全匹配 config/ 下的 yaml 配置文件结构
"""

from typing import Any

from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


# ===================== MCP 领域子模型 =====================
class SessionCtxTemplate(EventedModel):
    """MCP 初始化上下文上下文模版"""

    tenant_id: str = "t_001"
    user_id: str = "u_001"
    is_tenant_admin: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)  # 存放所有动态扩展字段


class McpServerDefinition(EventedModel):
    """MCP 服务定义（全局唯一，描述服务本身）"""

    mcp_id: str = ""
    url: str = ""
    scope: str = "public"  # public=所有租户可见；tenant=需授权条目才能访问


class McpTenantGrant(EventedModel):
    """租户 MCP 授权条目（指定某个租户可访问某个 tenant 级服务）"""

    mcp_id: str = ""
    tenant_id: str = ""


class McpClientConfig(EventedModel):
    """MCP 客户端配置"""

    session_idle_timeout: float = 10 * 60  # 空闲超时：10分钟没被使用就断开
    session_check_interval: float = 60  # 巡检间隔：每分钟扫一次
    session_cache_max_size: int = 1000  # 会话缓存 LRU 上限，超过淘汰最久未访问的
    identity_cache_max_size: int = 5000  # 身份维度工具缓存 LRU 上限，超过淘汰最久未访问的


class MCPSettings(EventedModel):
    """MCP 全局配置"""

    ctx_template: SessionCtxTemplate = Field(default_factory=SessionCtxTemplate)
    servers: list[McpServerDefinition] = Field(default_factory=list)
    grants: list[McpTenantGrant] = Field(default_factory=list)
    client_cfg: McpClientConfig = Field(default_factory=lambda: shared_client_cfg)


# 全局唯一的客户端配置实例
shared_client_cfg = McpClientConfig(
    session_idle_timeout=10 * 60,
    session_check_interval=60,
    session_cache_max_size=1000,
    identity_cache_max_size=5000,
)
