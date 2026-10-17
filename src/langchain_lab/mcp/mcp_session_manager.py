import asyncio
import contextlib
from typing import Any, cast

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession
from mcp.types import TextContent

from langchain_lab.mcp.mcp_client_session import MCPStreamableSession
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.MCP)

# MCP 服务地址配置
_MCP_SERVERS = {
    "rag": "http://127.0.0.1:8000/mcp",
}


class MCPSessionManager:
    """按身份缓存 MCP 会话，调用时自动注入 Header 上下文
    - 支持按身份缓存复用长连接
    - 支持单会话移除、按服务批量移除、全量清空
    - 移除时自动关闭连接，异常兜底压制
    """

    def __init__(self):
        self._cache_sess: dict[tuple, MCPStreamableSession] = {}
        self._lock = asyncio.Lock()

    async def get_session(
        self,
        mcp_id: str,
        server_url: str,
        session_ctx: dict[str, Any],
    ) -> MCPStreamableSession:
        tenant_id = session_ctx.get("tenant_id", "t_001")
        user_id = session_ctx.get("user_id", "u_001")
        cache_key = (mcp_id, tenant_id, user_id)

        async with self._lock:
            if cache_key in self._cache_sess:
                return self._cache_sess[cache_key]

            session = MCPStreamableSession(server_url=server_url)
            await session.set_context_meta(
                override={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "is_tenant_admin": str(session_ctx.get("is_tenant_admin", False)).lower(),
                },
                extra={k: v for k, v in session_ctx.items() if k not in {"tenant_id", "user_id", "is_tenant_admin"}},
            )
            # 设置完 Header 后主动建连，确保 _session 可用
            await session._ensure_session()
            self._cache_sess[cache_key] = session
            return session

    async def remove_session(self, mcp_id: str, tenant_id: str, user_id: str) -> bool:
        """
        移除指定身份的单个会话并关闭连接
        :return: 是否存在并移除了该会话
        """
        cache_key = (mcp_id, tenant_id, user_id)
        async with self._lock:
            session = self._cache_sess.pop(cache_key, None)

        if session is not None:
            # 关闭连接，所有清理异常全部压制
            with contextlib.suppress(BaseException):
                await session.close()
            return True
        return False

    async def remove_mcp(self, mcp_id: str) -> int:
        """
        移除指定 MCP 服务下的所有会话（服务下线/注销时调用）
        :return: 移除的会话数量
        """
        removed_keys = []
        async with self._lock:
            # 先在锁内收集所有匹配的 key，避免长时间占锁
            for key in list(self._cache_sess.keys()):
                if key[0] == mcp_id:
                    removed_keys.append(key)
            # 批量删除缓存引用
            sessions = [self._cache_sess.pop(k) for k in removed_keys]

        # 锁外执行 IO 关闭，不阻塞其他缓存操作
        for sess in sessions:
            with contextlib.suppress(BaseException):
                await sess.close()

        return len(sessions)

    async def clear_all(self) -> None:
        """清空所有缓存会话，服务停机时调用"""
        async with self._lock:
            sessions = list(self._cache_sess.values())
            self._cache_sess.clear()

        for sess in sessions:
            with contextlib.suppress(BaseException):
                await sess.close()


# 全局单例
_session_mgr = MCPSessionManager()


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
