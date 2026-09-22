"""
文本切分器：
将 RawDocument 切分为多个 Document（chunk）。
1. 按 Markdown 标题分节（如果有）
2. 按段落分块
3. 超过 chunk_size 的块按固定大小+重叠滑动切分
切分后的每个 Document 会继承文件级元数据，并添加 section 和 chunk_index 等块级元数据。
"""

import re
from dataclasses import dataclass

from langchain_lab.rag.loader import RawDocument
from langchain_lab.rag.retriever import Document
from langchain_lab.util.logger import logger


@dataclass
class SplitConfig:
    """文本分段配置参数

    Attributes:
        chunk_size: int
            目标块大小，单位：字符数。单个分片尽量不超过该长度。
        overlap: int
            分片重叠字符数。相邻分片尾部/头部保留重叠文本，用于减少上下文断裂损失。
        separator: str
            主分隔符，优先使用该分隔符切分文本，默认两个换行（段落分隔）。
        secondary_separator: str
            备用分隔符。使用主分隔符切分后，分片依然超过 chunk_size 时，使用该分隔符继续切分。
    """

    chunk_size: int = 500  # 目标块大小（字符数）
    overlap: int = 50  # 重叠字符数
    separator: str = "\n\n"  # 主分隔符（段落）
    secondary_separator: str = "\n"  # 备用分隔符（句子/行），分段后仍然字数超限时，优先分隔


class TextSplitter:
    """
    - 通用文本分片器
    - 基于分隔符的层级文本切分：优先按主分隔符拆分；块过大时自动降级使用备用分隔符。
    - 支持设置分片重叠，适用于 RAG 文档预处理场景。

    Args:
        config: SplitConfig | None
            分片配置对象，不传则使用 SplitConfig 默认参数。
    """

    def __init__(self, config: SplitConfig | None = None) -> None:
        self.config = config or SplitConfig()

    def split_raw_document(self, raw_doc: RawDocument) -> list[Document]:
        """
        入口：将 RawDocument 切分为 Document 列表
        自动继承 raw_doc.metadata 到每个 chunk 的 metadata
        """
        # 文件级元数据：source, file_path, modified...
        base_meta = raw_doc.metadata.copy()  # 这里 copy 隔离，避免后续操作修改元数据影响文件级元数据
        return self._split_file_text(raw_doc.text, base_meta)

    def _split_file_text(self, text: str, base_meta: dict) -> list[Document]:
        """
        核心切分逻辑：
        1. 按 Markdown 标题切分为多个 section
        2. 每个 section 内按段落切分
        3. 超长段落按固定大小+重叠切分
        """
        sections = self._split_by_markdowm_heading(text)

        chunks: list[tuple[str, dict]] = []  # 存储每个待生成 chunk 的 text 和 metadata
        for section_title, section_text in sections:
            # 每个 section 有自己的 metadata （继承 base_meta， 加上 section 标题）
            section_meta = base_meta.copy()
            if section_title:
                section_meta["section"] = section_title

            # 段落切分
            if len(section_text) <= self.config.chunk_size:
                # 整个 section_text 短，直接作为一个块（保留结构完整性）
                chunks.append((section_text, section_meta))
            else:
                # section_text 长，按段落切分，再处理超长段落
                paragraphs = self._split_by_paragraphs(section_text)
                for para in paragraphs:
                    if len(para) <= self.config.chunk_size:
                        chunks.append((para, section_meta.copy()))
                    else:
                        # 长段落滑动切分
                        sub_chunks = self._split_by_fixed_size(para)
                        for sub in sub_chunks:
                            chunks.append((sub, section_meta.copy()))

        # 转换为 Document 对象，并添加 chunk_index
        final_docs: list[Document] = []

        for idx, (text, meta) in enumerate(chunks):
            doc = Document(
                text=text,
                metadata={
                    **meta,
                    "chunk_index": str(idx),  # 文本块于对于单个文件内的序号
                },
            )
            final_docs.append(doc)

        # 1. 按 Markdowm 标题分节（如果存在 # 标题）
        logger.info(f"切分完成：{len(final_docs)}")
        return final_docs

    def _split_by_markdowm_heading(self, text: str) -> list[tuple[str, str]]:
        """
        按 Markdown 标题切分，返回 [(标题, 内容)]
        没有标题时返回 [("", 全文)]
        标题行（#、##、###...）不会被包含在内容中，而是作为 section 元数据。
        """
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [("", text)]

        sections = []

        for i, match in enumerate(matches):
            level = len(match.group(1))  # 标题级别，可留作扩展  # noqa: F841
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            sections.append((title, section_text))
        return sections

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """按连续空行（或指定分隔符）切分为段落，去除空段落"""
        paragraphs = [p.strip() for p in text.split(self.config.separator) if p.strip()]
        return paragraphs

    def _split_by_fixed_size(self, text: str) -> list[str]:
        """
        固定大小 + 重叠滑动切分。
        尽量在句号、换行等自然断点处断开，避免切断语义。
        """
        chunk_size = self.config.chunk_size
        secondary_separator = self.config.secondary_separator
        overlap = self.config.overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # 还没到末尾时，尝试在最近的断点处结束
            if end < len(text):
                for sep in [secondary_separator, "。", "！", "？", "\n", "；", "，", " "]:
                    pos = text.rfind(sep, start + chunk_size // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            # 下一次起点 = 结束位置 - 重叠，至少前进一个字符，避免死循环
            start = max(end - overlap, start + 1)
        return chunks
