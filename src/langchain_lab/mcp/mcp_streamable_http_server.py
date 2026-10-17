# mcp_streamable_http_server.py
import anyio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from langchain_lab.mcp.context_inject import bind_exec_context, setup_exec_context
from langchain_lab.mcp.mcp_servers.rag_mcp_server import mcp as mcp_server


# ========== 上下文注入中间件 ==========
class ExecContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 批量读取 override 元数据
        override_ctx = {}
        for header_name, value in request.headers.items():
            if header_name.lower().startswith("x-meta-override-"):
                key = header_name[len("X-Meta-Override-") :].lower()
                override_ctx[key] = value

        # 批量读取 extra 元数据
        extra_ctx = {}
        for header_name, value in request.headers.items():
            if header_name.lower().startswith("x-meta-extra-"):
                key = header_name[len("X-Meta-Extra-") :].lower()
                extra_ctx[key] = value

        # 统一注入上下文，和 bind_exec_context 原生能力对齐
        token_wrapper = bind_exec_context(
            override_session_ctx=override_ctx if override_ctx else None,
            extra_session_ctx=extra_ctx if extra_ctx else None,
        )
        try:
            return await call_next(request)
        finally:
            token_wrapper.reset()


# ========== 生成 MCP 应用，端点就是 /mcp ==========
app = mcp_server.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=False,
)
# 给 MCP 应用加中间件
app.add_middleware(ExecContextMiddleware)


async def main():
    # 服务启动：全局模板初始化 + 配置订阅
    setup_exec_context()

    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    anyio.run(main)
