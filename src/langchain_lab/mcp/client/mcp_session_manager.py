import asyncio
import contextlib
import time
from collections import OrderedDict
from typing import Any

from langchain_lab.mcp.client.mcp_streamable_session import MCPStreamableSession
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.MCP)

from langchain_lab.core.config.config_core import get_config

mcp_settings = get_config().mcp


class MCPSessionManager:
    """按身份缓存 MCP 会话，调用时自动注入 Header 上下文
    - 支持按身份缓存复用长连接
    - 支持单会话移除、按服务批量移除、全量清空
    - 移除时自动关闭连接，异常兜底压制
    """

    def __init__(self):
        # 会话缓存：OrderedDict 实现 LRU，key = (mcp_id, tenant_id, user_id)
        self._cache_sess: OrderedDict[tuple, MCPStreamableSession] = OrderedDict()
        self._last_used: dict[tuple, float] = {}  # 埋点：每个会话的最后使用时间戳
        self._lock = asyncio.Lock()

        # 空闲超时回收配置
        self.idle_timeout: float = mcp_settings.client_cfg.session_idle_timeout  # 空闲超时：没被使用就断开
        self._check_interval: float = mcp_settings.client_cfg.session_check_interval  # 巡检间隔：定时扫一次
        # LRU 上限配置
        self._session_cache_max_size: int = mcp_settings.client_cfg.session_cache_max_size

        # 后台清理任务托管
        self._cleaner_task: asyncio.Task | None = None
        self._cleaner_started = False

    async def get_session(
        self,
        mcp_id: str,
        server_url: str,
        session_ctx: dict[str, Any],
    ) -> MCPStreamableSession:
        tenant_id = session_ctx.get("tenant_id")
        user_id = session_ctx.get("user_id")
        cache_key = (mcp_id, tenant_id, user_id)

        # 确保清理后台任务已启动（懒启动，第一次用会话才启动）
        await self._ensure_cleaner()

        async with self._lock:
            if cache_key in self._cache_sess:
                # LRU：命中标记为最近使用
                self._cache_sess.move_to_end(cache_key)
                self._last_used[cache_key] = time.time()
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
        await session.ensure_session()

        # 再次上锁二次校验 + LRU 淘汰
        evicted_session: MCPStreamableSession | None = None
        async with self._lock:
            if cache_key in self._cache_sess:
                # 别的协程抢先完成初始化，缓存已经存在
                await session.close()
                self._cache_sess.move_to_end(cache_key)
                self._last_used[cache_key] = time.time()
                return self._cache_sess[cache_key]
            # 当前session有效，存入缓存持有引用
            self._cache_sess[cache_key] = session
            self._cache_sess.move_to_end(cache_key)
            self._last_used[cache_key] = time.time()
            # LRU 淘汰：超过上限时弹出最久未访问的会话
            while len(self._cache_sess) > self._session_cache_max_size:
                evicted_key, evicted_sess = self._cache_sess.popitem(last=False)
                self._last_used.pop(evicted_key, None)
                evicted_session = evicted_sess

        # 锁外关闭被淘汰的会话（不能在锁里做 IO）
        if evicted_session is not None:
            with contextlib.suppress(BaseException):
                await evicted_session.close()

        return session

    async def remove_session(self, mcp_id: str, tenant_id: str, user_id: str) -> bool:
        """
        移除指定身份的单个会话并关闭连接
        :return: 是否存在并移除了该会话
        """
        cache_key = (mcp_id, tenant_id, user_id)
        async with self._lock:
            session = self._cache_sess.pop(cache_key, None)
            self._last_used.pop(cache_key, None)  # 同步移除时间记录
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
            # 同步清理时间记录
            for k in removed_keys:
                self._last_used.pop(k, None)

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
            self._last_used.clear()  # 清空时间记录
            # 停机时取消巡检任务
            if self._cleaner_task and not self._cleaner_task.done():
                self._cleaner_task.cancel()
        for sess in sessions:
            with contextlib.suppress(BaseException):
                await sess.close()

    async def _ensure_cleaner(self) -> None:
        """懒启动后台巡检任务，保证只启动一次"""
        if self._cleaner_started:
            return
        async with self._lock:
            if self._cleaner_started:
                return
            self._cleaner_task = asyncio.create_task(self._clean_idle_loop())
            logger.debug(
                f"MCP 后台巡检清理超时会话任务已启动：每隔 {int(self._check_interval)} 秒清理超过 {int(self.idle_timeout)} 秒未使用的 MCP 会话"
            )
            self._cleaner_started = True

    async def _clean_idle_loop(self) -> None:
        """后台巡检循环：定期清理超时空闲会话"""
        while True:
            try:
                await asyncio.sleep(self._check_interval)
                await self._clean_expired_sessions()
            except asyncio.CancelledError:
                # 正常取消（停机），直接退出
                break
            except Exception as e:
                logger.warning(f"MCP会话巡检任务异常，已忽略：{e!s}")
                continue

    async def _clean_expired_sessions(self) -> int:
        """执行一次超时清理，返回被清理的会话数量"""
        now = time.time()
        expired_keys = []

        # 锁内只做判断和收集，不做IO关闭
        async with self._lock:
            for key, last_used in self._last_used.items():
                if now - last_used > self.idle_timeout:
                    expired_keys.append(key)

            # 先从缓存和时间记录移除
            sessions_to_close = []
            for key in expired_keys:
                sess = self._cache_sess.pop(key, None)
                if sess is not None:
                    sessions_to_close.append(sess)
                self._last_used.pop(key, None)

        # 锁外执行关闭IO，不阻塞其他缓存操作
        for sess in sessions_to_close:
            with contextlib.suppress(BaseException):
                await sess.close()

        if expired_keys:
            logger.debug(f"已自动清理 {len(expired_keys)} 个超时空闲的MCP会话")
        return len(expired_keys)


# 全局单例
_session_mgr = MCPSessionManager()
