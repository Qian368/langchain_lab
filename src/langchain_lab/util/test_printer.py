"""
Printer 流式样式 + 嵌套样式测试脚本

用法：
    1. 将本文件放在 printer.py 的同级目录
    2. 如果你的模块名不是 printer.py，请修改下方导入
    3. 运行：uv run python test_printer.py
"""

import sys
import traceback

from langchain_lab.util.prompt_style import Printer


def separator(title: str):
    print("=" * 60)
    print(title)
    print("=" * 60)


def check(name: str, condition: bool) -> bool:
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def run_tests() -> bool:
    printer = Printer()
    original_stdout = sys.stdout
    all_ok = True

    try:
        # ---------- 测试 1：基础 set/reset ----------
        separator("测试 1：基础 set_style → stream_out → reset_style")
        printer.set_style(printer.style_name.SUCCESS)
        printer.stream_chunk("① 这段是 SUCCESS 样式（绿色加粗）\n")
        print("② 这段 print 应该也保持绿色加粗")
        printer.reset_style()
        print("③ 这段已恢复正常")
        all_ok &= check("_current_style 已清空", printer._current_style == "")
        all_ok &= check("_stdout_stack 已清空", len(printer._stdout_stack) == 0)
        all_ok &= check("_style_stack 已清空", len(printer._style_stack) == 0)
        all_ok &= check("sys.stdout 已恢复原引用", sys.stdout is original_stdout)

        # ---------- 测试 2：嵌套 set/reset ----------
        separator("测试 2：嵌套 set_style，reset 后恢复外层样式")
        printer.set_style(printer.style_name.BLUE)
        printer.stream_chunk("① 外层蓝色\n")
        printer.set_style(printer.style_name.RED)
        printer.stream_chunk("② 内层红色（覆盖蓝色）\n")
        printer.reset_style()
        printer.stream_chunk("③ 应该恢复蓝色\n")
        printer.reset_style()
        printer.stream_chunk("④ 恢复默认\n")
        all_ok &= check("_current_style 已清空", printer._current_style == "")
        all_ok &= check("_stdout_stack 已清空", len(printer._stdout_stack) == 0)
        all_ok &= check("_style_stack 已清空", len(printer._style_stack) == 0)
        all_ok &= check("sys.stdout 已恢复原引用", sys.stdout is original_stdout)

        # ---------- 测试 3：stream_style 上下文管理器 ----------
        separator("测试 3：stream_style 上下文管理器")
        with printer.stream_style(printer.style_name.GREEN):
            printer.stream_chunk("① 绿色（stream_style 内）\n")
            print("② 这段 print 也应该是绿色")
        print("③ 退出后恢复默认")
        all_ok &= check("_current_style 已清空", printer._current_style == "")
        all_ok &= check("_stdout_stack 已清空", len(printer._stdout_stack) == 0)
        all_ok &= check("_style_stack 已清空", len(printer._style_stack) == 0)
        all_ok &= check("sys.stdout 已恢复原引用", sys.stdout is original_stdout)

        # ---------- 测试 4：stream_style 嵌套 ----------
        separator("测试 4：stream_style 嵌套，内层退出后恢复外层")
        with printer.stream_style(printer.style_name.GREEN):
            printer.stream_chunk("① 绿色外层\n")
            with printer.stream_style(printer.style_name.YELLOW):
                printer.stream_chunk("② 黄色内层\n")
            printer.stream_chunk("③ 退出内层，应恢复绿色\n")
        print("④ 退出外层，恢复默认")
        all_ok &= check("_current_style 已清空", printer._current_style == "")
        all_ok &= check("_stdout_stack 已清空", len(printer._stdout_stack) == 0)
        all_ok &= check("_style_stack 已清空", len(printer._style_stack) == 0)
        all_ok &= check("sys.stdout 已恢复原引用", sys.stdout is original_stdout)

        # ---------- 测试 5：样式区间内外部 print 混用 ----------
        separator("测试 5：set_style 区间内 print 保持样式")
        printer.set_style(printer.style_name.HIGHLIGHT)
        print("① 这段 print 应该是高亮色（亮黄色前景）")
        print("② 第二段 print 仍保持高亮")
        printer.reset_style()
        print("③ 恢复默认")
        all_ok &= check("_current_style 已清空", printer._current_style == "")
        all_ok &= check("_stdout_stack 已清空", len(printer._stdout_stack) == 0)
        all_ok &= check("_style_stack 已清空", len(printer._style_stack) == 0)

        # ---------- 测试 6：边界情况 ----------
        separator("测试 6：空状态调用 reset_style / 连续 reset")
        printer.reset_style()  # 无样式时调用，不应报错
        printer.stream_chunk("① 默认样式输出\n")
        printer.reset_style()  # 连续第二次 reset
        printer.stream_chunk("② 再 reset 一次，也不应报错\n")
        check("无异常发生", True)

        # ---------- 测试 7：colorama 兼容 ----------
        separator("测试 7：与 colorama autoreset=True 兼容")
        try:
            import colorama
        except ImportError:
            print("⚠️ 未安装 colorama，跳过该测试")
        else:
            colorama.init(autoreset=True)
            try:
                print("已执行 colorama.init(autoreset=True)")
                printer.set_style(printer.style_name.WARNING)
                print("① 警告色（autoreset=True 下 print 仍保持样式）")
                print("② 再次 print 仍是警告色")
                printer.reset_style()
                print("③ 恢复默认")
            finally:
                colorama.deinit()
            check("sys.stdout 最终恢复原引用", sys.stdout is original_stdout)

        # ---------- 最终状态检查 ----------
        separator("最终状态检查")
        final_ok = True
        final_ok &= check("sys.stdout 已恢复原引用", sys.stdout is original_stdout)
        final_ok &= check("_current_style 为空", printer._current_style == "")
        final_ok &= check("_stdout_stack 为空", len(printer._stdout_stack) == 0)
        final_ok &= check("_style_stack 为空", len(printer._style_stack) == 0)

        # 确保终端样式复位
        sys.stdout.write("\033[0m")
        sys.stdout.flush()

        print()
        if final_ok and all_ok:
            print("🎉 所有结构测试通过！请视觉确认上方颜色输出是否正确。")
        else:
            print("⚠️ 存在异常，请检查上方 ❌ 提示。")
        return final_ok and all_ok

    except Exception:
        # 异常时兜底恢复 stdout，避免污染调用方环境
        if printer._stdout_stack:
            sys.stdout = printer._stdout_stack.pop()
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
