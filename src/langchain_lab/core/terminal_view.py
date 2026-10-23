"""TerminalView：终端交互 + 展示订阅者（复用 Printer 样式，不改接口）

- 自持 Printer 实例（不复用 prompt_style 模块级单例 printer，守 R1：无全局单例）
- 订阅展示类事件，按 session_id 过滤渲染（多会话隔离基础）
- 输入交互归 view（prompt_async）
- HUMAN_MESSAGE_RECEIVED 回显用户输入：用蓝色系渲染，区别于模型输出的 MAGENTA
- 行为对拍：核心内容（流式/工具/终止）与 exp3 一致；装饰标签近似复刻
"""
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType
from langchain_lab.util.prompt_style import Printer

# 订阅的展示类事件
_DISPLAY_EVENTS = (
    EventType.SESSION_CREATED,
    EventType.SESSION_CLOSED,
    EventType.HUMAN_MESSAGE_RECEIVED,
    EventType.STREAM_CHUNK,
    EventType.ASSISTANT_MESSAGE_EMITTED,  # 仅用于重置状态，不渲染内容（避免与流式重复）
    EventType.TOOL_CALL_START,
    EventType.TOOL_RESULT,
    EventType.CONTEXT_COMPRESSED,
)


class TerminalView:
    """终端订阅者：渲染展示事件 + 提供输入交互"""

    def __init__(self) -> None:
        self._printer = Printer()
        self._printer.reset_all_state()
        self._session_id = ""
        self._expect_reply_start = True  # 状态机：下一轮答复前先打"模型答复>"标签

    def bind(self, bus: EventBus, session_id: str) -> None:
        """注册为订阅者；只渲染本会话的事件（多会话隔离）"""
        self._session_id = session_id
        for et in _DISPLAY_EVENTS:
            bus.subscribe(et, self._on_event)

    # —— 输入交互（归 view）——
    async def prompt_async(self, text: str) -> str:
        return await self._printer.prompt_async(text, style_name=self._printer.style_name.MESSAGE)

    # —— 事件渲染 ——
    def _on_event(self, event: Event) -> None:
        if event.session_id != self._session_id:
            return
        handler = {
            EventType.SESSION_CREATED: self._on_session_created,
            EventType.SESSION_CLOSED: self._on_session_closed,
            EventType.HUMAN_MESSAGE_RECEIVED: self._on_human_message,
            EventType.STREAM_CHUNK: self._on_stream_chunk,
            EventType.ASSISTANT_MESSAGE_EMITTED: self._on_assistant_emitted,
            EventType.TOOL_CALL_START: self._on_tool_call_start,
            EventType.TOOL_RESULT: self._on_tool_result,
            EventType.CONTEXT_COMPRESSED: self._on_context_compressed,
        }.get(event.type)
        if handler is not None:
            handler(event)

    def _on_session_created(self, event: Event) -> None:
        self._printer.begin("X" * 20 + " 开始你的多轮对话 " + "X" * 20)

    def _on_session_closed(self, event: Event) -> None:
        self._printer.warning("X" * 20 + " 用户选择终止 " + "X" * 20 + "\n")

    def _on_human_message(self, event: Event) -> None:
        # 回显用户输入：蓝色系（区别于模型 MAGENTA），为前端对话框结构化交互预留
        self._printer.label_blue("你 > ")
        self._printer.blue(str(event.payload))

    def _on_stream_chunk(self, event: Event) -> None:
        # 每轮答复前先打"模型答复>"标签（近似复刻 exp3 观感）
        if self._expect_reply_start:
            self._printer.label_blue("模型答复> ")
            self._expect_reply_start = False
        self._printer.stream_chunk(str(event.payload), style_name=self._printer.style_name.MAGENTA)

    def _on_assistant_emitted(self, event: Event) -> None:
        # 整段答复已累积完成：不渲染内容（已流式打过），只重置状态，下一轮答复前再打标签
        self._expect_reply_start = True

    def _on_tool_call_start(self, event: Event) -> None:
        p = event.payload
        self._printer.label_yellow(f"有工具执行> 工具[{p['index']}]：{p['name']}")
        self._printer.output(f"        └─ 参数：{p['args']}")

    def _on_tool_result(self, event: Event) -> None:
        content = event.payload.get("content")
        self._printer.output(f"        └─ 执行结果：{content}")

    def _on_context_compressed(self, event: Event) -> None:
        # 压缩归档提示：早期轮已折叠为摘要（前端视图的"早期历史已折叠"占位对应）
        p = event.payload
        self._printer.label_yellow(f"[上下文压缩] 早期 {p.get('compressed_turns', '?')} 轮已折叠为摘要")
