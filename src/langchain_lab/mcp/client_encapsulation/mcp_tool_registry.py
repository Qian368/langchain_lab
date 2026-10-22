import asyncio
from typing import Any, cast

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession
from mcp.types import TextContent

from langchain_lab.mcp.client_encapsulation.mcp_session_manager import _MCP_SERVERS, MCPSessionManager, _session_mgr
from langchain_lab.mcp.client_encapsulation.mcp_streamable_session import MCPStreamableSession
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.MCP)


class MCPToolRegistry:
    """
    MCP 工具权限注册 + 按身份动态聚合
    - 服务定义与租户授权双层分离
    - 按当前身份动态拼装完整工具集
    - 按身份缓存工具列表，配置/权限变更时失效
    """

    def __init__(self, session_manager: MCPSessionManager):
        self._session_mgr = session_manager

        # 服务定义表：{mcp_id: {"url": str, "scope": "public"/"tenant"}}
        self._server_defs: dict[str, dict[str, str]] = {}
        # 租户授权表：{mcp_id: set(tenant_id, ...)}
        self._grants: dict[str, set[str]] = {}

        # 工具缓存：key = (tenant_id, user_id)
        self._tool_cache: dict[tuple, list[BaseTool]] = {}
        self._lock = asyncio.Lock()

        # 托管所有后台任务，防止 GC 回收，统一异常兜底
        self._background_tasks: set[asyncio.Task] = set()

    # ===================== 服务定义管理 =====================
    def register_server(self, mcp_id: str, url: str, scope: str = "public") -> None:
        """注册/更新一个 MCP 服务定义"""
        self._server_defs[mcp_id] = {"url": url, "scope": scope}
        self._invalidate_cache_by_mcp(mcp_id)

    async def unregister_server(self, mcp_id: str) -> int:
        """
        注销 MCP 服务（异步版，await 执行）
        :return: 关闭的会话数量
        """
        self._server_defs.pop(mcp_id, None)
        self._grants.pop(mcp_id, None)
        self._invalidate_cache_by_mcp(mcp_id)
        # 关闭该服务所有身份的连接
        removed = await self._session_mgr.remove_mcp(mcp_id)
        return removed

    def unregister_server_async(self, mcp_id: str) -> None:
        """后台异步注销服务（同步上下文用，如配置变更回调）"""
        self._run_background(self.unregister_server(mcp_id), f"unregister_{mcp_id}")

    # ===================== 租户授权管理 =====================
    def grant_tenant(self, mcp_id: str, tenant_id: str) -> None:
        """给指定租户授权访问某个 tenant 级服务"""
        if mcp_id not in self._server_defs:
            logger.warning(f"授权失败：MCP 服务 {mcp_id} 不存在")
            return
        if mcp_id not in self._grants:
            self._grants[mcp_id] = set()
        self._grants[mcp_id].add(tenant_id)
        # 失效该租户的工具缓存
        self._invalidate_cache_by_tenant(tenant_id)

    def revoke_tenant(self, mcp_id: str, tenant_id: str) -> None:
        """撤销指定租户的服务授权"""
        if mcp_id in self._grants:
            self._grants[mcp_id].discard(tenant_id)
        self._invalidate_cache_by_tenant(tenant_id)

    def grant_tenant_async(self, mcp_id: str, tenant_id: str) -> None:
        """后台异步授权"""

        async def _grant_task():
            self.grant_tenant(mcp_id, tenant_id)

        self._run_background(_grant_task(), f"grant_{mcp_id}_{tenant_id}")

    def revoke_tenant_async(self, mcp_id: str, tenant_id: str) -> None:
        """后台异步撤销授权"""

        async def _revoke_task():
            self.revoke_tenant(mcp_id, tenant_id)

        self._run_background(_revoke_task(), f"revoke_{mcp_id}_{tenant_id}")

    # ===================== 核心：按身份获取可用工具 =====================
    async def get_tools_for_identity(self, session_ctx: dict[str, Any]) -> list[BaseTool]:
        """
        获取当前身份可用的所有 MCP 工具
        内部自动合并权限筛选、连接复用、命名空间前缀
        """
        tenant_id = session_ctx.get("tenant_id", "t_001")
        user_id = session_ctx.get("user_id", "u_001")
        cache_key = (tenant_id, user_id)

        # 命中缓存直接返回
        async with self._lock:
            if cache_key in self._tool_cache:
                return self._tool_cache[cache_key]

        # 1. 筛选当前租户有权限的 MCP
        available_mcps = self._filter_available_mcps(tenant_id)

        # 2. 逐个拉取 MCP 工具
        all_mcp_tools = []
        for mcp_id in available_mcps:
            try:
                tools = await self._get_mcp_tools(mcp_id, session_ctx)
                all_mcp_tools.extend(tools)
                logger.debug(f"成功加载 MCP: {mcp_id} 的工具")
            except Exception as e:
                # 兜住 CancelledError（BaseException 子类）：单个 MCP 连接失败/超时只跳过，不拖垮整体
                logger.warning(f"加载 MCP {mcp_id} 工具失败：{e!s}")
                continue

        # 写入缓存
        async with self._lock:
            self._tool_cache[cache_key] = all_mcp_tools

        return all_mcp_tools

    def _filter_available_mcps(self, tenant_id: str) -> list[str]:
        """筛选当前租户有权限的 MCP ID 列表"""
        available = []
        for mcp_id, cfg in self._server_defs.items():
            if cfg["scope"] == "public" or (cfg["scope"] == "tenant" and tenant_id in self._grants.get(mcp_id, set())):
                available.append(mcp_id)
        return available

    async def _get_mcp_tools(self, mcp_id: str, session_ctx: dict[str, Any]) -> list[BaseTool]:
        """获取单个 MCP 的工具列表（复用会话管理器的连接）"""
        server_url = self._server_defs[mcp_id]["url"]
        sess = await self._session_mgr.get_session(mcp_id, server_url, session_ctx)

        # 不拿完整工具 raw_tools = await sess.get_tools()
        # 拿工具壳，不再获取完整 _MCPTool 对象
        raw_metas = await sess.get_tool_metas()

        wrapped: list[BaseTool] = []
        for meta in raw_metas:
            public_name = f"{mcp_id}__{meta.name}"
            wrapped.append(
                _make_mcp_tool(
                    mcp_id=mcp_id,
                    server_url=server_url,
                    original_name=meta.name,
                    public_name=public_name,
                    description=meta.description or "",
                    args_schema=meta.input_schema,
                )
            )
        return wrapped

    # ===================== 缓存管理 =====================
    def _invalidate_cache_by_mcp(self, mcp_id: str) -> None:
        """失效所有包含该 MCP 的工具缓存"""
        # 全量清空，简单可靠；量小无需精细化
        self._tool_cache.clear()

    def _invalidate_cache_by_tenant(self, tenant_id: str) -> None:
        """失效指定租户的所有工具缓存"""
        keys_to_remove = [k for k in self._tool_cache if k[0] == tenant_id]
        for k in keys_to_remove:
            self._tool_cache.pop(k, None)

    def invalidate_all_cache(self) -> None:
        """手动清空全部工具缓存"""
        self._tool_cache.clear()

    # ===================== 内部工具：后台任务托管 =====================
    def _run_background(self, coro, name: str) -> None:
        """统一托管后台任务，自动防 GC + 异常兜底"""

        async def _task():
            try:
                await coro
            except Exception as e:
                logger.warning(f"MCP 后台任务 {name} 失败：{e!s}")

        task = asyncio.create_task(_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


# 全局单例
_mcp_tool_registry = MCPToolRegistry(_session_mgr)

from langchain_lab.core.config.app_schema.mcp_models import MCPSettings
from langchain_lab.core.config.config_core import get_config


def init_mcp_from_config(mcp_settings: MCPSettings) -> None:
    """从配置批量初始化 MCP 服务 + 授权"""
    # 1. 注册所有服务定义
    for server_cfg in mcp_settings.servers:
        try:
            _mcp_tool_registry.register_server(
                mcp_id=server_cfg.mcp_id,
                url=server_cfg.url,
                scope=server_cfg.scope,
            )
        except Exception as e:
            logger.warning(f"MCP 服务 {server_cfg.mcp_id} 注册失败：{e!s}")

    # 2. 加载所有租户授权
    for grant_cfg in mcp_settings.grants:
        try:
            _mcp_tool_registry.grant_tenant(
                mcp_id=grant_cfg.mcp_id,
                tenant_id=grant_cfg.tenant_id,
            )
        except Exception as e:
            logger.warning(f"MCP 授权加载失败 {grant_cfg.mcp_id}/{grant_cfg.tenant_id}：{e!s}")

    logger.info("已完成 MCP 服务与授权初始化")


def bind_mcp_config_to_registry(mcp_settings: MCPSettings) -> None:
    """绑定配置变更事件，自动同步到运行时注册中心"""

    # ---------- 服务列表变更 ----------
    def on_servers_change(sender, old_val, new_val):
        old_ids = {s.mcp_id for s in old_val} if old_val else set()
        new_ids = {s.mcp_id for s in new_val} if new_val else set()

        # 移除已删除的服务
        for mcp_id in old_ids - new_ids:
            _mcp_tool_registry.unregister_server_async(mcp_id)

        # 注册新增/更新的服务
        for server_cfg in new_val:
            _mcp_tool_registry.register_server(
                mcp_id=server_cfg.mcp_id,
                url=server_cfg.url,
                scope=server_cfg.scope,
            )

    # ---------- 授权列表变更 ----------
    def on_grants_change(sender, old_val, new_val):
        old_grants = {(g.mcp_id, g.tenant_id) for g in old_val} if old_val else set()
        new_grants = {(g.mcp_id, g.tenant_id) for g in new_val} if new_val else set()

        # 撤销已删除的授权
        for mcp_id, tenant_id in old_grants - new_grants:
            _mcp_tool_registry.revoke_tenant_async(mcp_id, tenant_id)

        # 新增授权
        for mcp_id, tenant_id in new_grants - old_grants:
            _mcp_tool_registry.grant_tenant_async(mcp_id, tenant_id)

    # 订阅两个字段的独立变更事件
    mcp_settings.subscribe("servers", on_servers_change)
    mcp_settings.subscribe("grants", on_grants_change)

    logger.info("已绑定 MCP 配置变更事件，服务与授权自动同步全局配置")


# ========== 全局初始化 ==========
mcp_settings = get_config().mcp
init_mcp_from_config(mcp_settings)
bind_mcp_config_to_registry(mcp_settings)


# ========== 工具工厂 ==========
def _make_mcp_tool(
    mcp_id: str,
    server_url: str,
    original_name: str,
    public_name: str,
    description: str,
    args_schema: Any,
) -> BaseTool:
    """工厂函数：生成单个 MCP 包装工具"""

    async def _tool_runner(session_ctx: dict[str, Any] | None = None, **kwargs) -> str:
        """自定义 mcp 工具执行逻辑，只执行，不捕获异常，异常上浮至于工具中间件中捕获"""
        if session_ctx is None:
            session_ctx = {}
        sess = await _session_mgr.get_session(mcp_id, server_url, session_ctx)
        # 双重保险：调用前确保连接存活，异常断开自动重连
        await sess._ensure_session()
        client_session = cast(ClientSession, sess._session)
        result = await client_session.call_tool(name=original_name, arguments=kwargs)
        return "\n".join(c.text for c in result.content if isinstance(c, TextContent))

    # 正确调用方式：先传工具名 + 配置，返回装饰器，再装饰函数
    # 1. 第一个参数传字符串（工具名），后面全是关键字参数
    # 2. infer_schema=False：禁用自动推断，完全用传入的 args_schema，避免把 session_ctx 暴露给 LLM
    tool_decorator = tool(
        public_name,
        description=description,
        args_schema=args_schema,
        infer_schema=False,
    )
    # 3. 用装饰器包装执行函数，得到最终 BaseTool 实例
    wrapped = tool_decorator(_tool_runner)

    # 标记工具类型，供执行器识别
    wrapped._tool_type = "mcp"
    return wrapped


# ========== 核心：自动发现 + 批量包装 ==========
async def discover_mcp_tools(mcp_id: str) -> list[BaseTool]:
    """
    自动发现指定 MCP 服务的所有工具，批量生成标准 BaseTool
    - 自动加 {mcp_id}__ 命名空间前缀
    - 自动支持 session_ctx 参数注入身份 Header
    - 输出和本地工具完全兼容，可直接丢给 Agent
    - 连接失败时优雅降级，返回空列表，不抛出异常
    """
    server_url = _MCP_SERVERS.get(mcp_id)
    if not server_url:
        logger.warning(f"MCP 服务 {mcp_id} 未配置，跳过加载")
        return []

    try:
        # 临时会话拉取工具元数据
        async with MCPStreamableSession(server_url) as tmp_session:
            raw_tools = await tmp_session.get_tools()
    except BaseException as e:
        # 所有连接/初始化异常全部兜底，打警告返回空
        logger.warning(f"MCP 服务 {mcp_id} 连接失败，工具加载失败：{e!s}")
        return []

    wrapped_tools = []
    for raw_tool in raw_tools:
        public_name = f"{mcp_id}__{raw_tool.name}"
        wrapped_tool = _make_mcp_tool(
            mcp_id=mcp_id,
            server_url=server_url,
            original_name=raw_tool.name,
            public_name=public_name,
            description=raw_tool.description or "",
            args_schema=raw_tool.args_schema,
        )
        wrapped_tools.append(wrapped_tool)

    return wrapped_tools
