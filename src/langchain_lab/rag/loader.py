"""
文件加载器：
负责从磁盘读取文件，提取原始文本和文件级元数据。
支持单个文件加载和目录批量加载。
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.RAG)


@dataclass
class RawDocument:
    """
    ### 原生文件对象：真正的文件
    尚未切分的原始文档，包含文本和文件级元数据
    """

    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    # 例：{"source": "社保笔记.md", "file_path": "D:/notes/社保笔记.md", "modified": "2026-09-18 10:30"}


class FileLoader:
    def __init__(self, base_dir: str | Path = ".") -> None:
        """
        - 支持 .md 和 .txt 文件，按 UTF-8 读取
        - base_dir: 笔记根目录，用于显示相对路径（可选）
        """
        self.SUPPORTED_EXTS = {".md", ".txt", ".markdown"}
        self.base_dir = Path(base_dir)

    def _calc_file_sha256(self, file_path: Path) -> str:
        """计算文件sha256哈希"""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def load_file(self, file_path: str | Path) -> RawDocument:
        """加载单个文件，返回 RawDocument"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTS:
            logger.warning(f"不支持的文件类型：{path.suffix},跳过：{path}")
            raise ValueError(f"不支持的文件类型：{path.suffix}")  # 让调用方尝试打开，捕获并处理异常

        with open(path, "r", encoding="utf-8") as f:  # noqa: UP015
            text = f.read()

        content_hash = self._calc_file_sha256(path)

        metadata = {
            "content_hash": content_hash,
            "source": path.name,  # 先用文件名保底
            "file_path": str(path.absolute()),
        }
        # 尝试获取相对路径（如果 base_dir 匹配）
        try:
            rel_path = path.relative_to(self.base_dir)
            metadata["source"] = str(rel_path)
        except ValueError:
            pass

        # 获取修改时间
        stat = path.stat()
        metadata["modified"] = str(stat.st_mtime)

        logger.info(f"加载文件：{path}")
        return RawDocument(text=text, metadata=metadata)

    def load_directory(self, dir_path: str | Path) -> list[RawDocument]:
        """
        加载目录下所有支持的文件（递归）
        返回 RawDocument 列表
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"不是目录：{dir_path}")

        raw_docs = []

        for path in sorted(dir_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTS:
                try:
                    raw_docs.append(self.load_file(path))
                except Exception as e:
                    logger.error(f"加载失败{path}: {e}")
                    continue
        logger.info(f"共加载 {len(raw_docs)} 个文件")
        return raw_docs
