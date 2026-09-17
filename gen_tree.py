# /// script
# requires-python = ">=3.10"
# dependencies = ["pathspec>=0.12"]
# ///
"""根据 .gitignore 规则生成项目目录树文档。

脚本必须放置在项目根目录，与 .gitignore 同级。
若缺少 .gitignore 则自动提示并退出。
输出文件为 docs/project_tree.md，若已存在则覆盖。
"""

import sys
from datetime import datetime
from pathlib import Path

import pathspec

# 无论 .gitignore 如何配置，都强制忽略的目录 / 文件 / （完整相对路径）
ALWAYS_SKIP_PATHS: set[str] = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".env",
    "docs/project_tree.md",
}


def get_project_root() -> Path:
    """获取项目根目录。

    Returns:
        Path: 脚本所在目录，即项目根目录。

    Raises:
        SystemExit: 如果目录中不存在 .gitignore 文件。
    """
    root = Path(__file__).resolve().parent
    if not (root / ".gitignore").is_file():
        print(f"错误：脚本所在目录 {root} 下未找到 .gitignore 文件。")
        print("请在项目根目录创建 .gitignore 后再运行本脚本。")
        sys.exit(1)
    return root


def load_ignore_spec(root: Path) -> pathspec.PathSpec:
    """加载 .gitignore 规则。

    Args:
        root: 项目根目录。

    Returns:
        pathspec.PathSpec: 编译后的忽略规则对象。
    """
    gitignore_path = root / ".gitignore"
    with open(gitignore_path, "r", encoding="utf-8") as f:  # noqa: UP015
        return pathspec.PathSpec.from_lines("gitwildmatch", f)


def _build_tree(root: Path, node: Path, prefix: str, lines: list[str], spec: pathspec.PathSpec) -> None:
    """递归构建树形目录结构。

    Args:
        root: 项目根目录。
        node: 当前遍历的目录节点。
        prefix: 当前节点路径的前缀，用于绘制树形连接线。
        lines: 输出行列表。
        spec: .gitignore 忽略规则。
    """

    entries: list[Path] = []
    for entry in sorted(node.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        rel = entry.relative_to(root)
        rel_str = rel.as_posix()

        # 额外忽略规则：完整路径匹配
        if rel_str in ALWAYS_SKIP_PATHS:
            continue

        pattern = rel_str + ("/" if entry.is_dir() else "")  # 目录规则需补斜杠
        if spec.match_file(pattern):
            continue
        entries.append(entry)

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            _build_tree(root, entry, next_prefix, lines, spec)


def generate_tree_doc(root: Path) -> None:
    """遍历项目目录树，生成带树形连接符的 Markdown 文档。

    Args:
        root: 项目根目录。
    """
    spec = load_ignore_spec(root)
    output_file = root / "docs" / "project_tree.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# 项目目录结构",
        "",
        f"- 生成时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 项目根目录：`{root}`",
        "",
        "```text",
        f"{root.name}/",
    ]

    # 直接把根目录作为起点，递归打印所有子项
    _build_tree(root, root, "", lines, spec)

    lines.append("```")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"目录树已生成：{output_file}")


def main() -> None:
    """主入口函数。"""
    root = get_project_root()
    generate_tree_doc(root)


if __name__ == "__main__":
    main()
