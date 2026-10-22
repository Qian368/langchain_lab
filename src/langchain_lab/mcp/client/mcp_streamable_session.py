import asyncio
import contextlib
from typing import Any, cast

import httpx2
from langchain_core.tools import ArgsSchema, BaseTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent, Tool
from pydantic import BaseModel, create_model


def _schema_to_pydantic(schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")
        py_type = type_mapping.get(field_type, str)

        if field_name in required_fields:
            fields[field_name] = (py_type, ...)
        else:
            fields[field_name] = (py_type, None)

    return create_model("MCPToolArgs", **fields)


class _MCPTool(BaseTool):
    tool_name: str
    _session_ref: Any

    def _run(self, **kwargs: Any) -> str:
        raise RuntimeError("This tool only supports async invocation via _arun")

    async def _arun(self, **kwargs: Any) -> str:
        session = self._session_ref()
        result = await session.call_tool(name=self.tool_name, arguments=kwargs)

        text_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
        return "\n".join(text_parts)


# ========== 工具转换层（接口不变） ==========
async def load_mcp_tools(session: ClientSession) -> list[BaseTool]:
    list_result = await session.list_tools()
    tools: list[BaseTool] = []

    for tool in list_result.tools:
        args_schema = cast(ArgsSchema, _schema_to_pydantic(tool.input_schema))
        tools.append(
            _MCPTool(
                name=tool.name,
                description=tool.description or "",
                args_schema=args_schema,
                tool_name=tool.name,
                _session_ref=lambda: session,
            )
        )
    return tools


# 单工具版：签名和官方 as_langchain_tool 完全对齐
# 现在是我们自己实现，未来内部换成官方 as_langchain_tool 即可，外部调用零改动
# > 预留，暂时不用
def _as_langchain_tool(meta: Tool, session: ClientSession) -> BaseTool:
    args_schema = cast(ArgsSchema, _schema_to_pydantic(meta.input_schema))
    return _MCPTool(
        name=meta.name,
        description=meta.description or "",
        args_schema=args_schema,
        tool_name=meta.name,
        _session_ref=lambda: session,
    )


# 批量版（上层调用）
# > 预留，暂时不用
def _build_tools_from_mcp_metas(metas: list[Tool], session: ClientSession) -> list[BaseTool]:
    return [_as_langchain_tool(meta, session) for meta in metas]


# ========== 通用会话类 ==========
class MCPStreamableSession:
    """
    Streamable HTTP 流式会话类
    - 支持任意上下文元数据透传，不限制字段
    - 对齐服务端 override / extra 双语义
    - 保留 set_identity 便捷方法，兼容现有调用
    - 独立后台任务创建会话，提供拿工具，调用工具接口
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._override_meta: dict[str, str] = {}
        self._extra_meta: dict[str, str] = {}

        self._session: ClientSession | None = None

        # ========== 独立宿主任务相关状态 ==========
        # 核心：streamable_http_client + ClientSession 两层 async with
        # 全部封闭在 _host_coro 这个独立后台任务里，不污染主任务上下文
        self._host_task: asyncio.Task | None = None
        self._stream_ready: asyncio.Future[ClientSession] | None = None  # 宿主建好流+初始化完 session 后通知外部
        self._shutdown_event: asyncio.Event | None = None  # 外部发关闭信号给宿主
        self._http_client: httpx2.AsyncClient | None = None

        self._tool_cache: list[BaseTool] | None = None
        self._tool_meta_cache: list[Tool] | None = None

    # ===== 通用元数据方法（推荐生产用） =====
    async def set_context_meta(
        self,
        override: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """
        通用设置上下文元数据
        :param override: 覆盖模式，存在即替换（身份类字段用这个）
        :param extra: 增补模式，仅新增不覆盖（追踪类字段用这个）
        """
        if override is not None:
            self._override_meta = {str(k): str(v) for k, v in override.items()}
        if extra is not None:
            self._extra_meta = {str(k): str(v) for k, v in extra.items()}

        self._tool_cache = None
        self._tool_meta_cache = None
        # 直接 await 重置，不用后台任务，消除 anyio API 报错
        if self._session is not None:
            await self._reset_session()

    # ===== 便捷方法（兼容现有代码，内部调用通用方法） =====
    async def set_identity(self, tenant_id: str, user_id: str, is_admin: bool = False):
        """便捷设置身份三要素，内部走通用元数据通道"""
        await self.set_context_meta(
            override={
                "tenant_id": tenant_id,
                "user_id": user_id,
                "is_tenant_admin": str(is_admin).lower(),
            }
        )

    async def ensure_session(self):
        # 如果已有会话且宿主任务还活着，直接复用
        if self._session is not None and self._host_task is not None and not self._host_task.done():
            return

        # 有残留状态（宿主异常退出等），先清理干净
        if self._session is not None or self._host_task is not None:
            await self._reset_session()

        # 初始化同步原语
        self._stream_ready = asyncio.Future()
        self._shutdown_event = asyncio.Event()

        # 启动宿主任务（CancelScope 封闭在这个任务内部，不泄漏到当前调用方）
        self._host_task = asyncio.create_task(self._host_coro())

        try:
            self._session = await self._stream_ready
        except (asyncio.CancelledError, Exception) as e:
            # 建连失败，关闭清理所有资源
            await self._reset_session()
            raise RuntimeError(f"MCP 会话连接失败，已关闭会话。底层异常： {e!s}") from e

    # 只拿壳，缓存原生元数据，注册器走这个
    async def get_tool_metas(self) -> list[Tool]:
        """只拿工具壳，只缓存原生工具元数据，减轻缓存负担，注册器走这个"""
        if self._tool_meta_cache is not None:
            return self._tool_meta_cache
        await self.ensure_session()
        # 底层仍然是官方标准 API list_tools
        assert self._session is not None
        resp = await self._session.list_tools()
        self._tool_meta_cache = resp.tools
        return self._tool_meta_cache

    async def get_tools(self) -> list[BaseTool]:
        """生成绑定当前会话的可执行 _MCPTool，供直接使用session的场景（discover/调试）"""
        if self._tool_cache is not None:
            return self._tool_cache
        await self.ensure_session()
        assert self._session is not None
        self._tool_cache = await load_mcp_tools(self._session)  # 独立缓存，不连带缓存壳。
        return self._tool_cache

    # ========== 统一工具调用入口，外部不用再摸 _session 私有属性 ==========
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        统一工具调用：自动确保连接存活 + 调用 MCP + 提取文本结果
        外部（registry 的 _tool_runner）直接调这个，不用再碰 _session / TextContent
        """
        await self.ensure_session()
        assert self._session is not None
        result = await self._session.call_tool(name=tool_name, arguments=arguments)
        return "\n".join(c.text for c in result.content if isinstance(c, TextContent))

    async def close(self):
        await self._reset_session()

    # ========== 宿主任务协程——唯一持有 streamable_http_client + ClientSession 的地方 ==========
    async def _host_coro(self):
        """
        宿主任务：
        两层 async with（streamable_http_client + ClientSession）全部封闭在这个任务内部。
        两层自带的 anyio TaskGroup/CancelScope 都绑定宿主任务，不泄漏到主任务上下文。
        """
        # 组装 Header
        headers = {}
        for k, v in self._override_meta.items():
            headers[f"X-Meta-Override-{k}"] = v
        for k, v in self._extra_meta.items():
            headers[f"X-Meta-Extra-{k}"] = v

        self._http_client = httpx2.AsyncClient(headers=headers)
        try:
            # ClientSession.__aenter__() 内部会创建自己的 anyio TaskGroup，
            # 这个 TaskGroup 现在绑定宿主任务，不再绑定主任务。
            async with (
                streamable_http_client(self.server_url, http_client=self._http_client) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as mcp_session,
            ):
                await mcp_session.initialize()

                # 初始化完成，把 session 抛给外部任务
                if self._stream_ready and not self._stream_ready.done():
                    self._stream_ready.set_result(mcp_session)

                # 【核心阻塞点】等待关闭信号，维持两层 async with 存活 = 保持长连接
                assert self._shutdown_event is not None
                await self._shutdown_event.wait()
                # event 被 set 后：
                # 1. 先退出 ClientSession async with → 它内部 cancel_scope.cancel() → 只取消宿主任务
                # 2. 再退出 streamable_http_client async with → 它内部 cancel_scope.cancel() → 只取消宿主任务
        except BaseException as exc:
            # 建连阶段异常，通过 future 传递给等待中的外部任务
            if self._stream_ready and not self._stream_ready.done():
                self._stream_ready.set_exception(exc)
            # 运行中异常直接抛出，宿主任务结束
            raise
        finally:
            # 关闭 http 客户端
            if self._http_client:
                with contextlib.suppress(BaseException):
                    await self._http_client.aclose()
                self._http_client = None

    async def _reset_session(self):
        # ClientSession 的 async with 现在在宿主任务内部，
        # 退出时由宿主任务自己执行 __aexit__，它的 cancel_scope.cancel() 只作用宿主任务。
        # 这里只做：发关闭信号给宿主 → 等宿主收尾 → 清空状态引用

        # 1. 通知宿主任务退出两层 async with（幂等，多次 set 无害）
        if self._shutdown_event is not None and not self._shutdown_event.is_set():
            self._shutdown_event.set()

        # 2. 等待宿主任务完成收尾（宿主自己依次退出 ClientSession 和 streamable_http_client）
        if self._host_task is not None:
            with contextlib.suppress(BaseException):
                await self._host_task
            self._host_task = None

        # 3. 清空所有状态引用
        self._session = None
        self._stream_ready = None
        self._shutdown_event = None

        self._tool_meta_cache = None
        self._tool_cache = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._reset_session()
        return False


# ========== 使用示例 ==========
async def main():
    async with MCPStreamableSession("http://127.0.0.1:8000/mcp") as mcp:
        # 方式1：便捷方法，兼容旧代码
        await mcp.set_identity(tenant_id="t_001", user_id="u_001", is_admin=False)

        # 方式2：通用元数据，生产扩展用
        await mcp.set_context_meta(
            override={"tenant_id": "t_001", "user_id": "u_001", "role": "admin"},
            extra={"trace_id": "abc123", "request_id": "req-001"},
        )

        tools = await mcp.get_tools()
        print("MCP 工具列表：")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            if tool.args_schema is not None:
                # ✅ 强转后再调用，类型检查就认了
                schema_cls = cast(type[BaseModel], tool.args_schema)
                print(f"      └─ 参数: {schema_cls.model_json_schema()}")
            else:
                print("      └─ 参数: 无参数")
            print()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
