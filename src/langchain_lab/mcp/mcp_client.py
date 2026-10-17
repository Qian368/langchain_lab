# import anyio

# from langchain_mcp_adapters.tools import load_mcp_tools
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# async def connect_rag():
#     params = StdioServerParameters(
#         command="python",
#         args=["-m", "langchain_lab.mcp.mcp_servers.rag_mcp_server"],
#     )
#     read, write = await stdio_client(params).__aenter__()
#     session = await ClientSession(read, write).__aenter__()
#     await session.initialize()
#     tools = await load_mcp_tools(session)
#     return tools, session


# async def main():
#     tools, _ = await connect_rag()
#     print("MCP 工具列表：")
#     print(tools)


# if __name__ == "__main__":
#     anyio.run(main())

from typing import Any

import anyio
from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from pydantic import BaseModel, create_model


def _schema_to_pydantic(schema: dict[str, Any]) -> type[BaseModel]:
    """
    简易版 JSON Schema → Pydantic 模型转换（测试够用）
    支持基础类型：string / integer / number / boolean / array / object
    """
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
    """MCP v2 工具的 LangChain 异步包装器"""

    mcp_session: ClientSession
    tool_name: str

    # 必须实现同步抽象方法，否则无法实例化
    def _run(self, **kwargs: Any) -> str:
        raise RuntimeError("This tool only supports async invocation via _arun")

    async def _arun(self, **kwargs: Any) -> str:
        result = await self.mcp_session.call_tool(name=self.tool_name, arguments=kwargs)
        # 只提取 TextContent 的文本，跳过图片/音频/资源链接
        text_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
        return "\n".join(text_parts)


async def load_mcp_tools(session: ClientSession) -> list[BaseTool]:
    """
    替代官方 langchain_mcp_adapters 的 load_mcp_tools
    适配 MCP v2，接口签名与原版完全兼容
    """
    list_result = await session.list_tools()
    tools: list[BaseTool] = []

    for tool in list_result.tools:
        args_schema = _schema_to_pydantic(tool.input_schema)
        tools.append(
            _MCPTool(
                name=tool.name,
                description=tool.description or "",
                args_schema=args_schema,  # type: ignore[arg-type]
                mcp_session=session,
                tool_name=tool.name,
            )
        )

    return tools


async def connect_rag_stdio():
    params = StdioServerParameters(
        command="python",
        args=["-m", "langchain_lab.mcp.mcp_servers.rag_mcp_server"],
    )

    async with stdio_client(params) as (read_stream, write_stream), ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        return tools, session


async def connect_rag_http(server_url: str = "http://127.0.0.1:8000/mcp"):
    # Streamable HTTP，自动内部 httpx2
    async with streamable_http_client(server_url) as (read_stream, write_stream), ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        return tools, session


async def main():
    # 切换这里：
    tools, _ = await connect_rag_http()
    # tools, _ = await connect_rag_stdio()
    print("MCP 工具列表：")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")


if __name__ == "__main__":
    anyio.run(main)
