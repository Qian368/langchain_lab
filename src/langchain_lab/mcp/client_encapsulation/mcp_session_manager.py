import asyncio
import contextlib
from typing import Any

from langchain_lab.mcp.client_encapsulation.mcp_streamable_session import MCPStreamableSession
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

        # 锁外：创建+初始化，允许并发，这里会产生完整连接的session
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

        # 再次上锁二次校验
        async with self._lock:
            if cache_key in self._cache_sess:
                # 别的协程抢先完成初始化，缓存已经存在
                # 👉绝对不能直接丢弃session！必须手动close，在当前协程关闭异步生成器，交给GC必炸
                await session.close()
                return self._cache_sess[cache_key]
            # 当前session有效，存入缓存持有引用，不会被GC回收
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
