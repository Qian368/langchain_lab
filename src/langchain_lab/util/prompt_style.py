from enum import StrEnum

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style


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


class Theme:
    """主题配置类"""

    @staticmethod
    def build(
        color: Color | None = None,
        effect: Effect | None = None,
        background: Background | None = None,
        highlight: Highlight | None = None,
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
    def get_style(cls) -> Style:
        """获取完整样式"""
        return Style.from_dict(
            {
                "prompt": cls.build(Color.ANSI_BLUE, Effect.BOLD),
                "output": cls.build(Color.ANSI_BLUE, Effect.BOLD),
                "output1": cls.build(background=Background.BG_BLUE, effect=Effect.BOLD),
                "output2": cls.build(background=Background.BG_YELLOW, effect=Effect.BOLD),
                "output3": cls.build(Color.ANSI_BLUE, Effect.BOLD),
                "output4": cls.build(Color.ANSI_BLUE, Effect.BOLD),
                "output5": cls.build(Color.ANSI_BLUE, Effect.BOLD),
                "error": cls.build(Color.ANSI_RED, Effect.BOLD),
                "success": cls.build(Color.ANSI_GREEN),
                "warning": cls.build(Color.ANSI_YELLOW, Effect.BOLD),
                "info": cls.build(Color.ANSI_CYAN),
                "highlight": cls.build(highlight=Highlight.FG_ANSI_YELLOW),
            }
        )


# 使用
# style = Theme.get_style()


# 封装成更简洁的 API


class Printer:
    def __init__(self):
        self.style = Theme.get_style()

    def _format(self, style_name: str, text: str):
        """统一处理，完全避免 XML 解析"""
        print_formatted_text(FormattedText([(f"class:{style_name}", text)]), style=self.style)

    # 用法示例：
    # printer.prompt("请向你的助手发送消息> ", "output")
    def prompt(self, text: str, style_name: str = "prompt"):
        """带样式的输入提示"""
        return FormattedText([(f"class:{style_name}", text)])

    def output(self, text: str):
        self._format("output", text)

    def key_label(self, text: str):
        self._format("output1", text)

    def tool_label(self, text: str):
        self._format("output2", text)

    def output3(self, text: str):
        self._format("output3", text)

    def output4(self, text: str):
        self._format("output4", text)

    def output5(self, text: str):
        self._format("output5", text)

    def error(self, text: str):
        self._format("error", text)

    def success(self, text: str):
        self._format("success", text)

    def warning(self, text: str):
        self._format("warning", text)


# 使用
# printer = Printer()
# printer.output("模型答复> ")
# printer.error("出错了！")
# printer.success("成功了！")
