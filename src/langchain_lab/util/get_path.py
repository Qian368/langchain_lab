
import sys

from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget

app = QApplication(sys.argv)

def get_files(parent: QWidget | None = None) -> list[str]:
    """
    选择多个文件，返回路径列表。
    默认过滤器为 Markdown/文本文件，可切换为所有文件。
    """
    files, _ = QFileDialog.getOpenFileNames(
        parent,
        "选择文件",
        "",
        "文档文件 (*.md *.txt *.markdown);;所有文件 (*.*)"
    )
    return files


def get_folder(parent: QWidget | None = None) -> str:
    """
    选择文件夹，返回路径字符串。
    用于后续加载文件夹中的所有笔记文档。
    """
    folder = QFileDialog.getExistingDirectory(
        parent,
        "选择文件夹"
    )
    return folder
