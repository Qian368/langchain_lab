import asyncio
import contextlib
from typing import Any, cast

import httpx2
from langchain_core.tools import ArgsSchema, BaseTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
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


# ========== 通用会话类 ==========
class MCPStreamableSession:
    """
    Streamable HTTP 会话管理类（通用元数据版）
    - 支持任意上下文元数据透传，不限制字段
    - 对齐服务端 override / extra 双语义
    - 保留 set_identity 便捷方法，兼容现有调用
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8000/mcp"):
        self.server_url = server_url
        self._override_meta: dict[str, str] = {}
        self._extra_meta: dict[str, str] = {}
        self._session: ClientSession | None = None
        self._tool_cache: list[BaseTool] | None = None
        self._stream_ctx = None
        self._session_ctx = None

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
        # ✅ 修复：直接 await 重置，不用后台任务，消除 anyio API 报错
        if self._session is not None:
            await self._reset_session()
            # ✅ 关键：重置后让出事件循环，等后台清理任务收尾， 试了，没用
            # await asyncio.sleep(1)

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

    async def get_tools(self) -> list[BaseTool]:
        if self._tool_cache is not None:
            return self._tool_cache
        await self._ensure_session()
        assert self._session is not None
        self._tool_cache = await load_mcp_tools(self._session)
        return self._tool_cache

    async def close(self):
        await self._reset_session()

    async def _reset_session(self):
        if self._session_ctx:
            with contextlib.suppress(BaseException):
                await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
        if self._stream_ctx:
            with contextlib.suppress(BaseException):
                await self._stream_ctx.__aexit__(None, None, None)
            self._stream_ctx = None
        # ✅ 重置后让出事件循环，等后台任务收尾
        await asyncio.sleep(0)

    async def _ensure_session(self):
        if self._session is not None:
            return

        # 组装 Header：override 走 X-Meta-Override- 前缀，extra 走 X-Meta-Extra- 前缀
        headers = {}
        for k, v in self._override_meta.items():
            headers[f"X-Meta-Override-{k}"] = v
        for k, v in self._extra_meta.items():
            headers[f"X-Meta-Extra-{k}"] = v

        http_client = httpx2.AsyncClient(headers=headers)
        self._stream_ctx = streamable_http_client(self.server_url, http_client=http_client)
        try:
            read_stream, write_stream = await self._stream_ctx.__aenter__()
            self._session_ctx = ClientSession(read_stream, write_stream)
            session = await self._session_ctx.__aenter__()

            await session.initialize()
            self._session = session
        except (asyncio.CancelledError, Exception) as e:
            await self.close()
            raise RuntimeError(f"MCP 会话连接失败，会话已关闭底层异常： {e!s}") from e

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
