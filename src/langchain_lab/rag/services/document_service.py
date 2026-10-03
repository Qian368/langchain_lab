"""
文档服务：协调 MongoDB + SQLite + Chroma + splitter 完成导入、删除、重建索引。
"""

from pathlib import Path

from langchain_lab.config import ALLOWED_BASE_DIRS
from langchain_lab.rag.loader import FileLoader, RawDocument
from langchain_lab.rag.splitter import TextSplitter
from langchain_lab.rag.storage import MetaDB, MongoDocumentStore, VectorStore
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.RAG)


class DocumentService:
    """文档管理的业务逻辑"""

    def __init__(self):
        self.mongo = MongoDocumentStore()
        self.meta = MetaDB()
        self.vector = VectorStore()
        self.loader = FileLoader()
        self.splitter = TextSplitter()

    def store_file(
        self,
        file_path: str | Path,
        tenant_id: str = "unknown_id",
        user_id: str = "unknown_id",
    ) -> str:
        """
        第一步：读取文件并存入 MongoDB。

        返回 doc_id，此时状态为 pending。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        raw_doc = self.loader.load_file(file_path)
        doc_id = self.mongo.insert_document(
            raw_doc.metadata["source"],
            raw_doc.text,
            content_hash=raw_doc.metadata["content_hash"],
            tenant_id=tenant_id,
            user_id=user_id,
        )
        logger.info(f"文档已入库: {doc_id} | {raw_doc.metadata['source']}")
        return doc_id

    def index_document(self, doc_id: str) -> int:
        """
        第二步：从 MongoDB 取原文 → 切分 → 写 SQLite + Chroma → 更新状态。
        返回 chunk 数。
        """
        doc = self.mongo.get_document(doc_id)
        if not doc:
            raise ValueError(f"文档不存在: {doc_id}")

        raw_doc = RawDocument(
            text=doc["content"],
            metadata={"source": doc["filename"], "tenant_id": doc["tenant_id"], "user_id": doc["user_id"]},
        )
        chunks = self.splitter.split_raw_document(raw_doc)

        if not chunks:
            self.mongo.update_document_status(doc_id, "failed", error_msg="切分结果为 0")
            logger.warning(f"{doc_id} 切分为 0 块，标记 failed")
            return 0

        # 写 SQLite：chunk 元数据与文本
        records = []
        for chunk in chunks:
            records.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": doc_id,
                    "tenant_id": chunk.metadata.get("tenant_id", "unknown_id"),
                    "user_id": chunk.metadata.get("user_id", "unknown_id"),
                    "content": chunk.text,
                    "source": chunk.metadata.get("source", ""),
                    "section": chunk.metadata.get("section", ""),
                    "chunk_index": int(chunk.metadata.get("chunk_index", 0)),
                }
            )
        self.meta.insert_chunks(records)

        # 写 Chroma：向量
        from langchain_lab.rag.embedding import get_embedding_model

        model = get_embedding_model()
        embeddings = (
            model.encode(
                [c.text for c in chunks],
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            .astype("float32")
            .tolist()
        )
        self.vector.add_vectors(
            chunk_ids=[c.id for c in chunks],
            embeddings=embeddings,
            doc_ids=[doc_id] * len(chunks),
        )

        self.mongo.update_document_status(doc_id, "indexed", chunk_count=len(chunks))
        logger.info(f"{doc_id} 索引完成，共 {len(chunks)} 块")
        return len(chunks)

    def import_file(self, file_path: str | Path) -> str:
        """同步完整导入：入库 + 切分 + 索引。返回 doc_id"""
        doc_id = self.store_file(file_path)
        self.index_document(doc_id)
        return doc_id

    def import_folder(self, folder_path: str | Path) -> list[str]:
        """
        批量导入文件夹下所有 md/txt，返回成功导入的 doc_id 列表。
        只允许访问 ALLOWED_BASE_DIRS 内的目录。
        """
        folder_path = Path(folder_path).resolve()

        allowed = any(folder_path == base.resolve() or folder_path.is_relative_to(base.resolve()) for base in ALLOWED_BASE_DIRS)
        if not allowed:
            raise PermissionError(f"禁止访问该目录: {folder_path}，请先加入 ALLOWED_BASE_DIRS")

        files = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown"}]
        logger.info(f"发现 {len(files)} 个文档文件")

        doc_ids = []
        for fp in files:
            try:
                doc_id = self.import_file(fp)
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"导入 {fp} 失败: {e}")
        return doc_ids

    def delete_document(self, doc_id: str) -> bool:
        """删除文档：Chroma 删向量 → SQLite 删 chunks → MongoDB 删文档"""
        self.vector.delete_by_doc_ids([doc_id])
        deleted = self.meta.delete_chunks_by_doc_id(doc_id)
        logger.info(f"删除 {doc_id} 的 {deleted} 条 chunks")
        ok = self.mongo.delete_document(doc_id)
        if not ok:
            logger.warning(f"MongoDB 中不存在: {doc_id}")
        return ok

    def delete_documents(self, doc_ids: list[str]) -> int:
        """
        批量删除多篇文档：Chroma批量删向量 → SQLite【分批】删chunks → MongoDB批量删文档

        :param doc_ids: 文档_id字符串列表
        :return: 返回MongoDB成功删除的文档数量（对齐mongo侧返回语义）
        """
        # 过滤空字符串
        valid_doc_ids = [did for did in doc_ids if did]
        if not valid_doc_ids:
            return 0

        # 1. Chroma 删除向量
        self.vector.delete_by_doc_ids(valid_doc_ids)
        # 2. SQLite 【分批】删除chunks，调用的batch方法
        deleted_chunk_count = self.meta.batch_delete_by_doc_ids(valid_doc_ids)
        logger.info(f"批量删除文档 {valid_doc_ids}，共删除 {deleted_chunk_count} 条 chunks")
        # 3. MongoDB 批量删除文档
        deleted_doc_count = self.mongo.delete_documents(valid_doc_ids)
        if deleted_doc_count < len(valid_doc_ids):
            logger.warning(f"MongoDB 部分文档不存在，预期删除{len(valid_doc_ids)}，实际删除{deleted_doc_count}")
        return deleted_doc_count

    def reindex_document(self, doc_id: str) -> str:
        """重建索引：先删旧索引，再重新走 index_document"""
        self.vector.delete_by_doc_ids([doc_id])
        self.meta.delete_chunks_by_doc_id(doc_id)
        self.mongo.update_document_status(doc_id, "pending")
        self.index_document(doc_id)
        return doc_id
