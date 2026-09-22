"""
总体设计：主题 → 样式字典 → 样式组成要素（颜色、效果） → 附加底层指令控制
1. 单一数据源：StyleName 枚举 + Theme.get_style_dict() 统一定义所有样式
2. 双输出路径：print_formatted_text（非流式）和 output.write（流式）共享同一份样式定义
3. API 设计：set_style / reset_style / stream_style / stream_out 覆盖了所有流式场景
"""

import sys
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
import threading
from typing import ClassVar

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

# region 支持 Shift+Enter 换行输入
bindings = KeyBindings()


# Enter = 提交
@bindings.add(Keys.Enter)
def _(event):
    event.current_buffer.validate_and_handle()


# Shift+Enter = 换行
@bindings.add(Keys.Escape, Keys.Enter)  # Shift+Tab 的底层键名
def _(event):
    event.current_buffer.insert_text("\n")


# endregion


# region 样式的组成成分：颜色、效果枚举
class Color(StrEnum):
    """前景色（文字颜色）"""

    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"

    # ANSI 亮色
    ANSI_BLACK = "ansiblack"
    ANSI_RED = "ansired"
    ANSI_GREEN = "ansigreen"
    ANSI_YELLOW = "ansiyellow"
    ANSI_BLUE = "ansiblue"
    ANSI_MAGENTA = "ansimagenta"
    ANSI_CYAN = "ansicyan"
    ANSI_WHITE = "ansigray"

    # ANSI 亮色（bright 系列）
    ANSI_BRIGHT_BLACK = "ansibrightblack"
    ANSI_BRIGHT_RED = "ansibrightred"
    ANSI_BRIGHT_GREEN = "ansibrightgreen"
    ANSI_BRIGHT_YELLOW = "ansibrightyellow"
    ANSI_BRIGHT_BLUE = "ansibrightblue"
    ANSI_BRIGHT_MAGENTA = "ansibrightmagenta"
    ANSI_BRIGHT_CYAN = "ansibrightcyan"
    ANSI_BRIGHT_WHITE = "ansibrightwhite"


class Effect(StrEnum):
    """所有支持的样式效果"""

    BOLD = "bold"
    DIM = "dim"
    ITALIC = "italic"
    UNDERLINE = "underline"
    BLINK = "blink"
    REVERSE = "reverse"
    HIDDEN = "hidden"
    STRIKE = "strike"


class Background(StrEnum):
    """背景色"""

    BG_BLACK = "bg:black"
    BG_RED = "bg:red"
    BG_GREEN = "bg:green"
    BG_YELLOW = "bg:yellow"
    BG_BLUE = "bg:blue"
    BG_MAGENTA = "bg:magenta"
    BG_CYAN = "bg:cyan"
    BG_WHITE = "bg:white"

    # ANSI 背景色
    BG_ANSI_BLACK = "bg:ansiblack"
    BG_ANSI_RED = "bg:ansired"
    BG_ANSI_GREEN = "bg:ansigreen"
    BG_ANSI_YELLOW = "bg:ansiyellow"
    BG_ANSI_BLUE = "bg:ansiblue"
    BG_ANSI_MAGENTA = "bg:ansimagenta"
    BG_ANSI_CYAN = "bg:ansicyan"
    BG_ANSI_WHITE = "bg:ansigray"


class Highlight(StrEnum):
    """高亮色（fg: 前缀）"""

    FG_BLACK = "fg:black"
    FG_RED = "fg:red"
    FG_GREEN = "fg:green"
    FG_YELLOW = "fg:yellow"
    FG_BLUE = "fg:blue"
    FG_MAGENTA = "fg:magenta"
    FG_CYAN = "fg:cyan"
    FG_WHITE = "fg:white"

    FG_ANSI_BLACK = "fg:ansiblack"
    FG_ANSI_RED = "fg:ansired"
    FG_ANSI_GREEN = "fg:ansigreen"
    FG_ANSI_YELLOW = "fg:ansiyellow"
    FG_ANSI_BLUE = "fg:ansiblue"
    FG_ANSI_MAGENTA = "fg:ansimagenta"
    FG_ANSI_CYAN = "fg:ansicyan"
    FG_ANSI_WHITE = "fg:ansigray"


# endregion


# region 样式枚举、主题、ANSI码指令
"""
定义样式枚举 → 主题（样式字典封装） → ANSI（根据样式枚举生成ANSI码/指令）
"""


class StyleName(StrEnum):
    """所有样式名，改样式时只需改这里和 Theme.get_style_dict"""

    OUTPUT = "output"
    MESSAGE = "message"

    # 纯色区
    MAGENTA = "magenta"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

    # 状态区
    BEGIN = "begin"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    HIGHLIGHT = "highlight"

    # 标签区
    LABEL_BLUE = "label_blue"
    LABEL_GREEN = "label_green"
    LABEL_YELLOW = "label_yellow"
    LABEL_RED = "label_red"


class Theme:
    """
    主题配置类，返回通用 style：

    用法示例：
    style = Theme.get_style()
    """

    @staticmethod
    def build(
        color: Color | None = None,  # 前景色
        effect: Effect | None = None,  # 效果（加粗等）
        background: Background | None = None,  # 背景色
        highlight: Highlight | None = None,  # 文字高亮
    ) -> str:
        """构建样式字符串"""
        parts = []
        if background:
            parts.append(background.value)
        if highlight:
            parts.append(highlight.value)
        if color:
            parts.append(color.value)
        if effect:
            parts.append(effect.value)
        return " ".join(parts)

    @classmethod
    def get_style_map(cls) -> dict[str, str]:
        style_map: dict[str, str] = {
            # 标准区
            StyleName.OUTPUT: cls.build(None),
            StyleName.MESSAGE: cls.build(Color.ANSI_BLUE, Effect.BOLD),
            # 纯色区
            StyleName.MAGENTA: cls.build(Color.MAGENTA),
            StyleName.BLUE: cls.build(Color.ANSI_BLUE),
            StyleName.GREEN: cls.build(Color.GREEN),
            StyleName.YELLOW: cls.build(Color.YELLOW),
            StyleName.RED: cls.build(Color.RED),
            # 状态区
            StyleName.BEGIN: cls.build(Color.ANSI_BLUE, Effect.BOLD),
            StyleName.SUCCESS: cls.build(Color.ANSI_GREEN, Effect.BOLD),
            StyleName.WARNING: cls.build(Color.ANSI_YELLOW, Effect.BOLD),
            StyleName.ERROR: cls.build(Color.ANSI_RED, Effect.BOLD),
            StyleName.INFO: cls.build(Color.ANSI_CYAN),
            StyleName.HIGHLIGHT: cls.build(highlight=Highlight.FG_ANSI_YELLOW),
            # 标签区
            StyleName.LABEL_BLUE: cls.build(background=Background.BG_BLUE, effect=Effect.BOLD),
            StyleName.LABEL_GREEN: cls.build(background=Background.BG_GREEN, effect=Effect.BOLD),
            StyleName.LABEL_YELLOW: cls.build(background=Background.BG_YELLOW, effect=Effect.BOLD),
            StyleName.LABEL_RED: cls.build(background=Background.BG_RED, effect=Effect.BOLD),
        }
        return style_map

    @classmethod
    def get_style(cls) -> Style:
        """获取完整样式"""
        return Style.from_dict(cls.get_style_map())


class ANSI:
    """从 StyleName 枚举动态生成 ANSI 码"""

    RESET = "\033[0m"

    _FG_MAP: ClassVar[dict] = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "ansiblack": 30,
        "ansired": 31,
        "ansigreen": 32,
        "ansiyellow": 33,
        "ansiblue": 34,
        "ansimagenta": 35,
        "ansicyan": 36,
        "ansigray": 37,
        "ansibrightblack": 90,
        "ansibrightred": 91,
        "ansibrightgreen": 92,
        "ansibrightyellow": 93,
        "ansibrightblue": 94,
        "ansibrightmagenta": 95,
        "ansibrightcyan": 96,
        "ansibrightwhite": 97,
    }

    _EFFECT_MAP: ClassVar[dict] = {
        "bold": 1,
        "dim": 2,
        "italic": 3,
        "underline": 4,
        "blink": 5,
        "reverse": 7,
        "hidden": 8,
        "strike": 9,
    }

    # 普通色名 → 高亮 RGB（参考各终端 bright 色，按喜好调整）
    _BRIGHT_RGB: ClassVar[dict[str, tuple[int, int, int]]] = {
        "black": (128, 128, 128),
        "red": (255, 85, 85),
        "green": (85, 255, 85),
        "yellow": (255, 255, 85),
        "blue": (85, 85, 255),
        "magenta": (255, 85, 255),
        "cyan": (85, 255, 255),
        "white": (255, 255, 255),
    }

    @staticmethod
    def from_style_name(style_name: StyleName) -> str:
        """根据 StyleName 从 Theme 的样式配置生成 ANSI 码"""
        style_str = Theme.get_style_map()[style_name]
        codes = []

        for part in style_str.split():
            if part.startswith("bg:"):
                bg_name = part[3:]
                codes.append(str(ANSI._FG_MAP.get(bg_name, 0) + 10))
            elif part.startswith("fg:"):
                fg_name = part[3:]
                codes.append(str(ANSI._FG_MAP.get(fg_name, 39)))
            elif part in ANSI._EFFECT_MAP:
                codes.append(str(ANSI._EFFECT_MAP[part]))
            elif part in ANSI._FG_MAP:
                codes.append(str(ANSI._FG_MAP[part]))

        return f"\033[{';'.join(codes)}m" if codes else ""

    @staticmethod
    def from_style_name_bright(style_name: StyleName) -> str:
        """生成高亮版 ANSI 码，直接输出真彩色 RGB，保证每种颜色都有明显区别"""
        style_str = Theme.get_style_map()[style_name]
        codes = []

        for part in style_str.split():
            if part.startswith("bg:"):
                bg_name = part[3:]
                # 背景真彩色（可选，标签区用）
                if bg_name in ANSI._BRIGHT_RGB:
                    r, g, b = ANSI._BRIGHT_RGB[bg_name]
                    codes.append(f"48;2;{r};{g};{b}")
                else:
                    codes.append(str(ANSI._FG_MAP.get(bg_name, 0) + 10))
            elif part.startswith("fg:"):
                fg_name = part[3:]
                if fg_name in ANSI._BRIGHT_RGB:
                    r, g, b = ANSI._BRIGHT_RGB[fg_name]
                    codes.append(f"38;2;{r};{g};{b}")
                else:
                    codes.append(str(ANSI._FG_MAP.get(fg_name, 39)))
            elif part in ANSI._EFFECT_MAP:
                codes.append(str(ANSI._EFFECT_MAP[part]))
            elif part in ANSI._FG_MAP:
                # 把普通名/ansi 系列名归一化后查 RGB 表
                color = part
                base = color.removeprefix("ansi").removesuffix("bright")
                if base == "gray":
                    base = "white"
                if base in ANSI._BRIGHT_RGB:
                    r, g, b = ANSI._BRIGHT_RGB[base]
                    codes.append(f"38;2;{r};{g};{b}")
                else:
                    codes.append(str(ANSI._FG_MAP[color]))

        return f"\033[{';'.join(codes)}m" if codes else ""


# endregion


# region 自定义打印器
class Printer:
    """
    自定义打印器：
    统一封装不同样式：标准输入输出类、颜色类、状态类、标签类
    支持流式：流式、非流式打印

    Typical usage：

    printer = Printer()
    printer.output("模型答复> ")
    printer.success("操作成功")
    ...
    """

    style_name = StyleName  # 外部不需要额外导入 StyleName，这里用类属性，提示更便捷

    def __init__(self):
        self.style = Theme.get_style()
        self._current_style = ""  # 当前生效的 ANSI 码
        self._stdout_stack = []  # 保存原始 stdout（最多一个）
        self._style_stack = []  # 保存每层的外层 ANSI 码
        self._session = PromptSession(
            style=self.style,
            key_bindings=bindings,
            multiline=True,
        )
        # self._out = create_output()  # 保留给 prompt_toolkit 用

    # region
    # 打印到终端的内部方法
    # def _format(self, text: str, style_name: StyleName):  # 部分自动转化为亮色版，不可控，此api不采用
    #     """统一处理，完全避免 XML 解析"""
    # print_formatted_text(FormattedText([(f"class:{style_name.value}", text)]), style=self.style)
    def _format(self, text: str, style_name: StyleName, *, bright: bool = False, end: str = "\n"):
        """非流式单行输出
        - 默认走原生 ANSI；bright=True 时输出自定义真彩色高亮。
        """
        outer_style = self._current_style

        inner_code = ANSI.from_style_name_bright(style_name) if bright else ANSI.from_style_name(style_name)

        # 完整拼好再一次性写出，避免prompt_toolkit hook中间重置颜色
        # 有外层样式：先重置 → 写本行样式 → 文本 → 重置 → 恢复外层
        output = f"{ANSI.RESET}{inner_code}{text}{ANSI.RESET}{outer_style}{end}" if outer_style else f"{inner_code}{text}{ANSI.RESET}{end}"

        sys.stdout.write(output)
        sys.stdout.flush()

    # 标准区
    def output(self, text: str):
        self._format(text, StyleName.OUTPUT)

    def message(self, text: str, style_name: StyleName = StyleName.MESSAGE):
        """
        带样式的输入提示：用于传给 prompt_toolkit 的 prompt方法。

        Typical usage:

        printer.message("请向你的助手发送消息> ", StyleName.MESSAGE)
        """
        return FormattedText([(f"class:{style_name.value}", text)])

    def prompt(
        self,
        text: str = "",
        style_name: StyleName = StyleName.MESSAGE,
        **kwargs,
    ) -> str:
        """获取用户输入，返回输入文本"""
        message = self.message(text, style_name)
        prompt_continuation = self.message("> ", style_name=style_name)
        return self._session.prompt(
            message,
            prompt_continuation=prompt_continuation,
            **kwargs,
        )

    async def prompt_async(
        self,
        text: str = "",
        style_name: StyleName = StyleName.MESSAGE,
        **kwargs,
    ) -> str:
        """异步获取用户输入，返回输入文本，调用方需要自身异步等待"""
        message = self.message(text, style_name)
        prompt_continuation = self.message("> ", style_name=style_name)
        return await self._session.prompt_async(
            message,
            prompt_continuation=prompt_continuation,
            **kwargs,
        )

    # 纯色区
    def magenta(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.MAGENTA, bright=bright)

    def blue(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.BLUE, bright=bright)

    def green(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.GREEN, bright=bright)

    def yellow(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.YELLOW, bright=bright)

    def red(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.RED, bright=bright)

    # 状态区
    def begin(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.BEGIN, bright=bright)

    def success(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.SUCCESS, bright=bright)

    def warning(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.WARNING, bright=bright)

    def error(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.ERROR, bright=bright)

    def info(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.INFO, bright=bright)

    def highlight(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.HIGHLIGHT, bright=bright)

    # 标签区
    def label_blue(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.LABEL_BLUE, bright=bright)

    def label_green(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.LABEL_GREEN, bright=bright)

    def label_yellow(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.LABEL_YELLOW, bright=bright)

    def label_red(self, text: str, *, bright: bool = False):
        self._format(text, StyleName.LABEL_RED, bright=bright)

    # endregion
    # region 单次样式设置，流式输出样式配置
    def set_style(self, style_name: StyleName):
        """不推荐直接使用：设置当前输出样式（支持嵌套），兼容 init(autoreset=True)"""
        # 第一次进入样式区时，保存原始 stdout 并替换为 sys.__stdout__
        if not self._stdout_stack:
            self._stdout_stack.append(sys.stdout)
            raw_stdout = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
            sys.stdout = raw_stdout

        # 压栈“当前生效的样式”，作为将来 reset 时恢复的外层样式
        self._style_stack.append(self._current_style)

        # 设置新样式
        self._current_style = ANSI.from_style_name(style_name)
        sys.stdout.write(self._current_style)
        sys.stdout.flush()

    def recover_style(self):
        """不推荐直接使用：重置为外层样式，如果已退出所有样式则恢复默认"""
        # 清掉当前所有样式
        sys.stdout.write(ANSI.RESET)
        sys.stdout.flush()

        if self._style_stack:
            # 恢复外层样式
            self._current_style = self._style_stack.pop()
            if self._current_style:
                sys.stdout.write(self._current_style)
                sys.stdout.flush()
        else:
            self._current_style = ""

        # 如果所有样式已退出，恢复原始 stdout
        if not self._style_stack and self._stdout_stack:
            sys.stdout = self._stdout_stack.pop()

    def reset_all_state(self):
        """备用方法：强制清空内存样式栈，恢复终端默认状态"""
        self._style_stack.clear()
        self._current_style = ""
        sys.stdout.write(ANSI.RESET)
        sys.stdout.flush()

    def sync_terminal_from_stack(self):
        """备用方法：强制对齐终端状态到栈记录的状态"""
        sys.stdout.write(ANSI.RESET)
        if self._current_style:
            sys.stdout.write(self._current_style)
        sys.stdout.flush()

    def stream_chunk(
        self,
        chunk: str,
        style_name: StyleName = StyleName.OUTPUT,
        end: str = "",
        *,
        bright: bool = False,
    ):
        """
        输出文本块，使用当前生效的样式
        """
        self._format(chunk, style_name, bright=bright, end=end)

    @contextmanager
    def stream_style(self, style_name: StyleName) -> Generator[None]:
        """
        上下文管理器，进入时设置样式，退出时自动重置 **（不推荐异步场景使用）**

        Typical usage:

        printer = Printer()

        with printer.stream_style(StyleName.SUCCESS):

            for chunk in stream_chunks:

                printer.stream_chunk(chunk) # 或者其他流式打印方法
        """
        self.set_style(style_name)
        try:
            yield
        finally:
            self.recover_style()

    # endregion


# endregion
