"""实验9：事件驱动 Agent 运行时终端接入层（伪前端，不动 exp3）

三层边界（接入层 / AgentRuntime / ChatSession）：
- 【接入层职责】身份与会话选择(调 runtime) · 指令面/消息面分类(终端解析) · 订阅事件流渲染 · 输入交互
- 不 new ChatSession · 不组装 store/llm/compressor · 不读全局配置（全部由 AgentRuntime 宿主承担）
- 指令解析在接入层：:branch/:branches/:switch/:compress/:delete/:rename/:rerun 分类后转事件（走 runtime.handle_*）
- 持久化订阅者 ContextStore 由 AgentRuntime 组装并 subscribe，接入层只从 runtime 拿 bus 订阅事件流展示
- 事件订阅全部【定向投递】session_id（只收当前会话事件，O(1) 不放大）；接入层订阅自己绑定会话的事件
- MCP 连接 clear_all 在 AgentRuntime.close()（服务停机语义，不在 ChatSession 内）
- 系统提示词不再前端自定义，由后端全局配置加载（runtime 注入）
"""

import asyncio

from langchain_lab.core.agent_runtime import AgentRuntime
from langchain_lab.core.commands import INSTRUCTION_SPECS, get_spec
from langchain_lab.core.event_bus import EventChannel
from langchain_lab.core.events import Event, EventType
from langchain_lab.util.prompt_style import Printer

_DEFAULT_SESSION_NAME = "新会话"


def _build_alias_map() -> dict[str, str]:
    """从指令注册表构建 终端别名(alias) → action(name) 反查表（单一真相源）"""
    return {
        alias: name
        for name, spec in INSTRUCTION_SPECS.items()
        for alias in spec.get("aliases", [])
    }


_ALIAS_TO_ACTION = _build_alias_map()


def _parse_command(text: str) -> tuple[EventType | None, dict | None]:
    """终端接入层解析输入（指令面/消息面分类；指令解析在接入层，运行时只响应事件）

    命令表由指令注册表派生（aliases → action，payload 按注册表 schema 填充必填字段），
    不再散落硬编码别名/参数。返回 (event_type, payload)：
    - 识别到注册表指令 → (SESSION_COMMAND, {"action":..., 载荷字段})
    - 独立触发面（:compress，非注册表 SESSION_COMMAND 指令）→ (COMPRESS_REQUESTED, payload)
    - 像指令但非法（别名存在但参数缺失/类型错/不满足约束）→ (None, {"error": 提示})
    - 普通消息 → (None, None)
    """
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    action = _ALIAS_TO_ACTION.get(cmd)
    if action is None:
        # 独立触发面（非 SESSION_COMMAND 指令）：压缩触发，保留特殊处理
        if cmd in (":compress", "/compress"):
            keep = int(arg) if arg.isdigit() and int(arg) >= 1 else None
            return EventType.COMPRESS_REQUESTED, {"force": True, "keep": keep}
        return None, None  # 普通消息

    spec = get_spec(action)
    assert spec is not None, f"未注册指令 action={action}"
    payload_fields = spec.get("payload") or {}
    payload: dict = {}
    # 按注册表 schema 填充必填字段（终端单参数：取第一个 required 字段，类型转换 + 约束校验）
    required = [f for f, m in payload_fields.items() if m.get("required")]
    if required:
        fname = required[0]
        fmeta = payload_fields[fname]
        if not arg:
            return None, {"error": f"{cmd} 用法：{cmd} <{fname}>（{fmeta.get('description') or ''}）"}
        try:
            value: int | str = int(arg) if fmeta.get("type") == "integer" else arg
        except ValueError:
            return None, {"error": f"{cmd} 参数 {fname} 需为 {fmeta.get('type')}"}
        if fmeta.get("minimum") is not None and isinstance(value, int) and value < fmeta["minimum"]:
            return None, {"error": f"{cmd} 参数 {fname} 需 >= {fmeta['minimum']}"}
        payload[fname] = value

    return EventType.SESSION_COMMAND, {"action": action, **payload}


def _build_help_text() -> str:
    """终端帮助：从指令注册表生成（别名 + 必填参数 + 描述），单一真相源"""
    lines = ["控制面指令（来源：指令注册表）："]
    for name, spec in INSTRUCTION_SPECS.items():
        aliases = " / ".join(spec.get("aliases", []))
        required = [f for f, m in (spec.get("payload") or {}).items() if m.get("required")]
        usage = aliases + (f" <{' '.join(required)}>" if required else "")
        lines.append(f"  {usage}  — {spec.get('description')}")
    lines.append("  手动压缩：:compress [保留轮数]（独立触发面，非注册表指令）")
    return "\n".join(lines)


class _TerminalRenderer:
    """终端渲染订阅者（内联 exp9：展示层归接入层，不复用独立 view 模块）

    - 定向订阅绑定会话的事件（session_id），不被其他会话事件触发；多会话隔离基础
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

    def bind(self, bus: EventChannel, session_id: str) -> None:
        """注册为订阅者：定向订阅本会话事件（O(1) 投递，不被其他会话触发）"""
        self._session_id = session_id
        for et in self._DISPLAY_EVENTS:
            bus.subscribe(self._on_event, et, session_id=session_id)

    async def prompt_async(self, text: str) -> str:
        return await self._printer.prompt_async(text, style_name=self._printer.style_name.MESSAGE)

    def info(self, text: str) -> None:
        self._printer.info(text)

    def output(self, text: str) -> None:
        self._printer.output(text)

    def _on_event(self, event: Event) -> None:
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


async def main() -> None:
    runtime = AgentRuntime()  # 后端宿主：身份/配置/依赖组装/store 全在内部，接入层不碰
    await runtime.ainit()
    bus = runtime.bus
    renderer = _TerminalRenderer()

    # —— 身份/会话选择（调 runtime，不 new ChatSession）——
    sessions = await runtime.list_sessions()
    chat = None
    session_name = _DEFAULT_SESSION_NAME
    resumed = False
    if sessions:
        renderer.info(f"已有 {len(sessions)} 个会话：")
        for i, s in enumerate(sessions, 1):
            renderer.output(f"  [{i}] {s['name'] or '(未命名)'}  · {s['session_id'][:8]}  · {s['status']}")
        choice = (await renderer.prompt_async("输入编号恢复会话（回车新建）：\n> ")).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                sid = sessions[idx - 1]["session_id"]
                session_name = sessions[idx - 1]["name"] or session_name
                chat = await runtime.resume_session(sid)
                resumed = chat is not None
    if chat is None:
        name = (await renderer.prompt_async(f"输入会话名（回车默认「{_DEFAULT_SESSION_NAME}」）：\n> ")).strip()
        session_name = name or _DEFAULT_SESSION_NAME
        chat = await runtime.create_session(name=session_name)

    session_id = chat._session_id
    renderer.bind(bus, session_id)

    done = asyncio.Event()
    compress_done = asyncio.Event()
    session_deleted = asyncio.Event()

    def _on_turn_done(e: Event) -> None:
        done.set()

    def _on_compressed(e: Event) -> None:
        compress_done.set()

    def _on_deleted(e: Event) -> None:
        session_deleted.set()

    # 定向订阅当前会话（O(1) 投递，不放大）
    bus.subscribe(_on_turn_done, EventType.AGENT_TURN_DONE, session_id=session_id)
    bus.subscribe(_on_compressed, EventType.CONTEXT_COMPRESSED, session_id=session_id)
    bus.subscribe(_on_deleted, EventType.SESSION_DELETED, session_id=session_id)
    renderer.info(_build_help_text())

    try:
        await chat.start(resumed=resumed)
        while True:
            human_prompt = await renderer.prompt_async("请向你的助手发送消息(shift + enter 换行)：\n> ")
            if human_prompt == "0":
                break
            if human_prompt.strip().lower() in (":help", "/help", "help"):
                renderer.info(_build_help_text())
                continue
            etype, payload = _parse_command(human_prompt)
            if etype is not None and payload is not None:
                if etype == EventType.COMPRESS_REQUESTED:
                    compress_done.clear()
                    runtime.request_compress(session_id, force=True, keep=payload.get("keep"))
                    try:
                        await asyncio.wait_for(compress_done.wait(), timeout=10)
                    except TimeoutError:
                        renderer.info("无可压早期轮或压缩未完成")
                else:  # SESSION_COMMAND（含 :rerun，走 _do_regenerate，自带 AGENT_TURN_DONE）
                    done.clear()
                    runtime.handle_command(session_id, payload["action"], payload)
                    await done.wait()
                    if session_deleted.is_set():  # 删了 root（整个会话）→ 退出
                        break
                continue
            if etype is not None and payload is None:
                renderer.info("输入指令缺乏参数，请输入正确指令")
            if payload is not None:  # 像指令但非法 → 接入层提示
                renderer.info(payload.get("error", "无法识别的指令"))
                continue
            done.clear()
            runtime.send_message(session_id, human_prompt)
            await done.wait()
    finally:
        await runtime.close()  # 停机语义：关全部会话 + channel.close + 关 MCP 长连接


if __name__ == "__main__":
    asyncio.run(main())
