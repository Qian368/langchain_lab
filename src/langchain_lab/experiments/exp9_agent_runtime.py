"""实验9：事件驱动 Agent 运行时入口（消息总线化，面向网络服务化，不动 exp3）

架构边界（事件驱动 / 订阅者化 / 解析前移 / 单一真相源 / 树形会话）：
- 输入事件化：终端适配器读输入 → 【适配器层解析+校验】→ 转事件（指令或普通消息）→ publish → ChatSession 订阅响应
- 指令解析不放在运行时：适配器把 :branch/:branches/:switch/:compress/:delete/:rename/:rerun 分类校验后转为
  SESSION_COMMAND / COMPRESS_REQUESTED 事件，普通输入转 USER_MESSAGE_SUBMITTED
- 输出事件化：ChatSession 发布 STREAM/TOOL/ASSISTANT/SESSION 等事件
- 持久化订阅者：ContextStore.subscribe_to_bus 订阅增量归档/压缩结果/摘要进归档/重跑作废落库（archive 单一真相源）
- 展示订阅者：终端渲染订阅者【内联本入口】（不复用独立 view 模块，exp9 即展示层）
- 恢复：入口用 store.load_view（回溯完整历史 + 最近摘要 → build_llm_context）注入 ChatSession（KV 逐 token 一致）
- 分支：store.create_branch 返回 (new_id, 拐点视图)（branch_history 回溯，支持压缩后/继承历史任意位置分叉）
- 删除：store.delete_branch（级联软删子树）；删 root=删整个会话 → SESSION_DELETED 驱动退出；gc_purge 留手动/后台
- 重跑：:rerun 重跑当前最后一条人类消息轮（复用该轮工具结果、条件化注解、新回答归档、旧回答作废）
- 命名：新建会话/分支带 name（默认值）；:rename 改当前会话或分支名
- 运行上限/库路径/默认提示词：按 config/runtime.yaml agent 配置注入
- MCP 连接 clear_all() 在此 finally（服务停机语义，不放在 ChatSession 内）
"""
import asyncio
import uuid

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.config.config_core import get_config
from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType
from langchain_lab.core.llm.llm import build_llm, get_llm
from langchain_lab.core.tools.tool_disclosure import get_init_tools
from langchain_lab.mcp.client.mcp_session_manager import _session_mgr
from langchain_lab.util.prompt_style import Printer

_DEFAULT_SESSION_NAME = "新会话"


def _parse_command(text: str) -> tuple[EventType | None, dict | None]:
    """终端适配器解析输入（指令解析在适配器层，运行时只响应事件）

    返回 (event_type, payload)：
    - 识别到指令 → (对应事件类型, payload)
    - 像指令但非法（以 : 或 / 开头、参数不合法）→ (None, {"error": 提示})
    - 普通消息 → (None, None)
    """
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in (":branch", "/branch"):
        if arg.isdigit() and int(arg) >= 1:
            return EventType.SESSION_COMMAND, {"action": "branch_create", "point": int(arg)}
        return None, {"error": "branch 用法：:branch <N>（N≥1 的对话消息序号）"}
    if cmd in (":branches", "/branches"):
        return EventType.SESSION_COMMAND, {"action": "branch_list"}
    if cmd in (":switch", "/switch"):
        if arg:
            return EventType.SESSION_COMMAND, {"action": "branch_switch", "branch_id": arg}
        return None, {"error": "switch 用法：:switch <branch_id>"}
    if cmd in (":delete", "/delete"):
        if arg:
            return EventType.SESSION_COMMAND, {"action": "branch_delete", "branch_id": arg}
        return None, {"error": "delete 用法：:delete <branch_id>（root=删除整个会话）"}
    if cmd in (":rename", "/rename"):
        if arg:
            return EventType.SESSION_COMMAND, {"action": "rename", "name": arg}
        return None, {"error": "rename 用法：:rename <新名字>（当前分支 root 改会话名，否则改分支名）"}
    if cmd in (":compress", "/compress"):
        keep = int(arg) if arg.isdigit() and int(arg) >= 1 else None
        return EventType.COMPRESS_REQUESTED, {"force": True, "keep": keep}
    if cmd in (":rerun", "/rerun", ":regenerate"):
        # 重跑当前最后一条人类消息轮（复用工具结果；适配器解析即触发）
        return EventType.SESSION_COMMAND, {"action": "regenerate"}
    return None, None  # 普通消息


class _TerminalRenderer:
    """终端渲染订阅者（内联 exp9：展示层归入口，不复用独立 view 模块）

    - 订阅展示类事件，按 session_id 过滤渲染（多会话隔离基础）
    - 输入交互收进终端（prompt_async）；会话内系统提示经 SESSION_NOTICE 渲染
    - 行为对拍：核心内容（流式/工具/终止/回显）与 exp3 一致
    """

    _DISPLAY_EVENTS = (
        EventType.SESSION_CREATED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_CLOSED,
        EventType.SESSION_DELETED,
        EventType.SESSION_NOTICE,
        EventType.HUMAN_MESSAGE_RECEIVED,
        EventType.STREAM_CHUNK,
        EventType.ASSISTANT_MESSAGE_EMITTED,  # 仅重置状态，不渲染内容（避免与流式重复）
        EventType.TOOL_CALL_START,
        EventType.TOOL_RESULT,
        EventType.CONTEXT_COMPRESSED,
        EventType.ERROR_OCCURRED,
    )

    def __init__(self) -> None:
        self._printer = Printer()
        self._printer.reset_all_state()
        self._session_id = ""
        self._expect_reply_start = True  # 状态机：下一轮答复前先打"模型答复>"标签

    def bind(self, bus: EventBus, session_id: str) -> None:
        """注册为订阅者；只渲染本会话的事件（多会话隔离）"""
        self._session_id = session_id
        for et in self._DISPLAY_EVENTS:
            bus.subscribe(et, self._on_event)

    async def prompt_async(self, text: str) -> str:
        return await self._printer.prompt_async(text, style_name=self._printer.style_name.MESSAGE)

    def info(self, text: str) -> None:
        self._printer.info(text)

    def output(self, text: str) -> None:
        self._printer.output(text)

    def _on_event(self, event: Event) -> None:
        if event.session_id != self._session_id:
            return
        handler = {
            EventType.SESSION_CREATED: self._on_session_created,
            EventType.SESSION_RESUMED: self._on_session_resumed,
            EventType.SESSION_CLOSED: self._on_session_closed,
            EventType.SESSION_DELETED: self._on_session_deleted,
            EventType.SESSION_NOTICE: self._on_notice,
            EventType.HUMAN_MESSAGE_RECEIVED: self._on_human_message,
            EventType.STREAM_CHUNK: self._on_stream_chunk,
            EventType.ASSISTANT_MESSAGE_EMITTED: self._on_assistant_emitted,
            EventType.TOOL_CALL_START: self._on_tool_call_start,
            EventType.TOOL_RESULT: self._on_tool_result,
            EventType.CONTEXT_COMPRESSED: self._on_context_compressed,
            EventType.ERROR_OCCURRED: self._on_error,
        }.get(event.type)
        if handler is not None:
            handler(event)

    def _on_session_created(self, event: Event) -> None:
        self._printer.begin("X" * 20 + " 开始你的多轮对话 " + "X" * 20)

    def _on_session_resumed(self, event: Event) -> None:
        self._printer.warning("X" * 20 + " 会话已恢复 " + "X" * 20)

    def _on_session_closed(self, event: Event) -> None:
        self._printer.warning("X" * 20 + " 用户选择终止 " + "X" * 20 + "\n")

    def _on_session_deleted(self, event: Event) -> None:
        self._printer.warning("X" * 20 + " 会话已删除 " + "X" * 20 + "\n")

    def _on_notice(self, event: Event) -> None:
        self._printer.info(str(event.payload))

    def _on_human_message(self, event: Event) -> None:
        self._printer.label_blue("你 > ")
        self._printer.blue(str(event.payload))

    def _on_stream_chunk(self, event: Event) -> None:
        if self._expect_reply_start:
            self._printer.label_blue("模型答复> ")
            self._expect_reply_start = False
        self._printer.stream_chunk(str(event.payload), style_name=self._printer.style_name.MAGENTA)

    def _on_assistant_emitted(self, event: Event) -> None:
        self._expect_reply_start = True

    def _on_tool_call_start(self, event: Event) -> None:
        p = event.payload
        self._printer.label_yellow(f"有工具执行> 工具[{p['index']}]：{p['name']}")
        self._printer.output(f"        └─ 参数：{p['args']}")

    def _on_tool_result(self, event: Event) -> None:
        content = event.payload.get("content")
        self._printer.output(f"        └─ 执行结果：{content}")

    def _on_context_compressed(self, event: Event) -> None:
        p = event.payload
        forced = "[手动]" if p.get("forced") else "[自动]"
        self._printer.label_yellow(f"{forced}上下文压缩] 早期 {p.get('compressed_turns', '?')} 轮已折叠为摘要")

    def _on_error(self, event: Event) -> None:
        p = event.payload or {}
        self._printer.warning(f"[运行时错误 @{p.get('phase', '?')}] {p.get('error', '')}")


async def _pick_session(
    printer: Printer, renderer: _TerminalRenderer, store: ContextStore, session_ctx: dict,
    keep_recent_turns: int,
) -> tuple[str, str, list | None, str | None, str]:
    """选会话：列出已有会话，编号恢复（回溯视图）或新建；返回 (session_id, branch_id, history, summary, name)"""
    sessions = await store.list_sessions(
        session_ctx.get("user_id", ""), session_ctx.get("tenant_id", "")
    )
    if sessions:
        printer.info(f"已有 {len(sessions)} 个会话：")
        for i, s in enumerate(sessions, 1):
            printer.output(f"  [{i}] {s['name'] or '(未命名)'}  · {s['session_id'][:8]}  · {s['status']}")
        choice = (await renderer.prompt_async("输入编号恢复会话（回车新建）：\n> ")).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                sid = sessions[idx - 1]["session_id"]
                history = await store.load_view(sid, "root", keep_recent_turns)  # 回溯构建 llm 视图
                summary = await store.load_summary(sid)
                return sid, "root", history, summary, sessions[idx - 1]["name"]
    name = (await renderer.prompt_async(f"输入会话名（回车默认「{_DEFAULT_SESSION_NAME}」）：\n> ")).strip()
    return str(uuid.uuid4()), "root", None, None, name or _DEFAULT_SESSION_NAME


def _build_compressor(bus: EventBus, session_id: str) -> SessionCompressor | None:
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
        summarize_prompt=comp_cfg.summarize_prompt,
        bus=bus,
        session_id=session_id,
    )


async def main() -> None:
    agent_cfg = get_config().runtime.agent
    session_ctx = get_config().project.session_ctx
    keep_recent_turns = get_config().runtime.compression.keep_recent_turns
    printer = Printer()
    printer.reset_all_state()

    llm = await asyncio.to_thread(get_llm)
    all_tools = await get_init_tools(session_ctx)

    bus = EventBus()
    store = ContextStore(db_path=str(agent_cfg.session_db_path))
    store.subscribe_to_bus(bus)  # 持久化订阅者（增量归档/压缩结果/摘要进归档/重跑作废）
    renderer = _TerminalRenderer()  # 展示订阅者（内联 exp9）

    session_id, branch_id, history, summary, session_name = await _pick_session(
        printer, renderer, store, session_ctx, keep_recent_turns
    )
    renderer.bind(bus, session_id)

    system_prompt = agent_cfg.default_system_prompt
    is_new = history is None
    if is_new:
        system_prompt = (
            await renderer.prompt_async("请输入你的系统提示词(shift + enter 换行，回车用默认)：\n> ")
        ) or agent_cfg.default_system_prompt
    # 注册会话：新建写入原始 system_prompt + name；恢复传 None 保留 store 已有（KV 前缀一致，不覆盖）
    await store.ensure_session(
        session_id, session_ctx,
        None if not is_new else system_prompt,
        session_name if is_new else None,
    )

    compressor = _build_compressor(bus, session_id)
    chat = ChatSession(
        session_id=session_id, session_ctx=session_ctx, bus=bus, llm=llm, tools=all_tools,
        compressor=compressor, branch_id=branch_id, system_prompt=system_prompt,
        max_llm_loop=agent_cfg.max_llm_loop, max_tool_calls=agent_cfg.max_tool_calls,
        history=history, summary=summary,
        branch_loader=lambda bid: store.load_view(session_id, bid, keep_recent_turns),
        branch_lister=lambda: store.list_branches(session_id),
        branch_creator=lambda parent, point: store.create_branch(session_id, parent, point),
        branch_deleter=lambda bid: store.delete_branch(session_id, bid),
        session_renamer=lambda n: store.rename_session(session_id, n),
        branch_renamer=lambda bid, n: store.rename_branch(session_id, bid, n),
    )

    done = asyncio.Event()
    compress_done = asyncio.Event()
    session_deleted = asyncio.Event()

    def _on_turn_done(e: Event) -> None:
        if e.session_id == session_id:
            done.set()

    def _on_compressed(e: Event) -> None:
        if e.session_id == session_id:
            compress_done.set()

    def _on_deleted(e: Event) -> None:
        if e.session_id == session_id:
            session_deleted.set()

    bus.subscribe(EventType.AGENT_TURN_DONE, _on_turn_done)
    bus.subscribe(EventType.CONTEXT_COMPRESSED, _on_compressed)
    bus.subscribe(EventType.SESSION_DELETED, _on_deleted)
    renderer.info(
        "提示：分支  :branch <N> / :branches / :switch <id>；删除  :delete <id>（root=整会话）；"
        "重命名  :rename <名>；重跑当前轮  :rerun；手动压缩  :compress [保留轮数]"
    )

    try:
        await chat.start(resumed=history is not None)
        while True:
            human_prompt = await renderer.prompt_async("请向你的助手发送消息(shift + enter 换行)：\n> ")
            if human_prompt == "0":
                break
            etype, payload = _parse_command(human_prompt)
            if etype is not None:
                if etype == EventType.COMPRESS_REQUESTED:
                    compress_done.clear()
                    bus.publish(Event(etype, session_id, payload))
                    try:
                        await asyncio.wait_for(compress_done.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        renderer.info("无可压早期轮或压缩未完成")
                else:  # SESSION_COMMAND（含 :rerun 重跑，走 _do_regenerate，自带 AGENT_TURN_DONE）
                    done.clear()
                    bus.publish(Event(etype, session_id, payload))
                    await done.wait()
                    if session_deleted.is_set():  # 删了 root（整个会话）→ 退出
                        break
                continue
            if payload is not None:  # 像指令但非法 → 适配器层提示
                renderer.info(payload.get("error", "无法识别的指令"))
                continue
            done.clear()
            bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, session_id, {"content": human_prompt}))
            await done.wait()
    finally:
        await chat.close()
        print("程序停机，开始关闭所有 MCP 长连接")
        await _session_mgr.clear_all()


if __name__ == "__main__":
    asyncio.run(main())
