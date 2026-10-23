"""实验9：事件驱动 Agent 运行时入口（独立运行时，不动 exp3）

- 组装：配置注入 session_ctx / LLM / 全量工具 / EventBus / TerminalView / SessionStore / 压缩器
- 会话选择：列出已有（恢复）或新建（uuid4）；支持 :branch/:branches/:switch 分支命令
- 压缩：按 config/runtime.yaml 的 compression 配置构建 SessionCompressor（可选）
- 运行上限 / 会话库路径 / 默认系统提示词：按 config/runtime.yaml 的 agent 配置注入
- ChatSession 只做消息会话；MCP 连接 clear_all() 在此 finally（服务停机语义）
"""
import asyncio
import uuid

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.config.config_core import get_config
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.session_compressor import SessionCompressor
from langchain_lab.core.session_store import SessionStore
from langchain_lab.core.terminal_view import TerminalView
from langchain_lab.core.tools.tool_disclosure import get_init_tools
from langchain_lab.core.llm.llm import build_llm, get_llm
from langchain_lab.mcp.client.mcp_session_manager import _session_mgr
from langchain_lab.util.prompt_style import Printer


async def _pick_session(printer: Printer, view: TerminalView, store: SessionStore, session_ctx: dict) -> str:
    """会话选择：列出已有会话 → 选编号恢复，否则回车新建"""
    sessions = await store.list_sessions(session_ctx.get("user_id", ""), session_ctx.get("tenant_id", ""))
    if sessions:
        printer.info(f"已有 {len(sessions)} 个会话：")
        for i, s in enumerate(sessions, 1):
            printer.output(f"  [{i}] {s['title'] or '(无标题)'}  · {s['session_id'][:8]}  · {s['status']}")
        choice = (await view.prompt_async("输入编号恢复会话（回车新建）：\n> ")).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]["session_id"]
    return str(uuid.uuid4())


def _build_compressor(bus: EventBus, session_id: str) -> SessionCompressor | None:
    """按 runtime 配置构建压缩器；未启用返回 None（R3：配置注入）"""
    comp_cfg = get_config().runtime.compression
    if not comp_cfg.enabled:
        return None
    llm_cfg = get_config().llm
    inst_name = comp_cfg.summarize_model or llm_cfg.default_instance
    if not inst_name or inst_name not in llm_cfg.instances:
        raise ValueError(f"压缩摘要模型实例 '{inst_name}' 未在 config/llm.yaml 中配置")
    summarize_llm = build_llm(llm_cfg.instances[inst_name])
    return SessionCompressor(
        summarize_llm=summarize_llm,
        keep_recent_turns=comp_cfg.keep_recent_turns,
        bus=bus,
        session_id=session_id,
    )


async def main() -> None:
    # 0. 运行时 agent 配置（上限/库路径/默认提示词，原硬编码已收敛）
    agent_cfg = get_config().runtime.agent

    # 1. 组装（全部构造注入，无模块级全局单例依赖）
    session_ctx = get_config().project.session_ctx
    printer = Printer()
    printer.reset_all_state()

    llm = await asyncio.to_thread(get_llm)
    all_tools = await get_init_tools(session_ctx)

    bus = EventBus()
    view = TerminalView()
    store = SessionStore(db_path=str(agent_cfg.session_db_path))

    # 2. 会话选择：恢复 or 新建
    session_id = await _pick_session(printer, view, store, session_ctx)

    # 3. system prompt（可回车跳过默认，默认值来自 runtime.yaml）
    system_prompt = (await view.prompt_async("请输入你的系统提示词(shift + enter 换行，回车用默认)：\n> ")) or agent_cfg.default_system_prompt

    # 4. 组装 ChatSession + 压缩器 + 绑定订阅者
    compressor = _build_compressor(bus, session_id)
    chat = ChatSession(
        session_id=session_id,
        session_ctx=session_ctx,
        bus=bus,
        llm=llm,
        view=view,
        tools=all_tools,
        store=store,
        compressor=compressor,
        system_prompt=system_prompt,
        max_llm_loop=agent_cfg.max_llm_loop,
        max_tool_calls=agent_cfg.max_tool_calls,
    )
    view.bind(bus, session_id)
    view.info("提示：可用分支命令  :branch <N>（从第N条对话分叉）/ :branches / :switch <branch_id>")

    try:
        await chat.resume()  # 恢复历史（含摘要）
        await chat.run()  # 主循环
    finally:
        await chat.close()  # flush + save + SESSION_CLOSED
        # 服务停机语义：清空所有 MCP 长连接（只在此处，不放在 ChatSession 内）
        print("程序停机，开始关闭所有 MCP 长连接")
        await _session_mgr.clear_all()


if __name__ == "__main__":
    asyncio.run(main())
