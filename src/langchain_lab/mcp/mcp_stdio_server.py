# mcp_stdio_server.py
import asyncio

from langchain_lab.mcp.context_inject import bind_exec_context, setup_exec_context
from langchain_lab.mcp.mcp_servers.rag_mcp_server import mcp as mcp_server


async def run_single_mcp_session():
    token_wrapper = bind_exec_context()
    try:
        # 直接调用底层async方法，不再走 mcp.run()（它会新建anyio循环）
        await mcp_server.run_stdio_async()
    finally:
        token_wrapper.reset()


async def main():
    # ========== 服务全局模板初始化（整个进程仅一次） ==========
    setup_exec_context()
    await run_single_mcp_session()


if __name__ == "__main__":
    asyncio.run(main())
