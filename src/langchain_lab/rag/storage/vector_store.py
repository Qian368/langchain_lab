"""
Chroma 向量存储层：只负责向量的增删查。
metadata 增加 tenant_id、user_id，用于租户隔离；不存业务文本。
职责：内置底层租户隔离兜底约束；业务权限、角色逻辑上移至service。
"""

from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.base_types import Metadata

from langchain_lab.core.config.config_core import get_config
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.RAG)
type EmbeddingVec = Sequence[float]


class VectorStore:
    """Chroma 封装，对齐Mongo、MetaDB的仓储范式"""

    def __init__(self, path: str | Path | None = None, collection_name: str | None = None):
        # 未显式传入则读统一配置 rag.chroma.*（原硬编码 CHROMA_PATH/CHROMA_COLLECTION 已收敛）
        chroma_cfg = get_config().rag.chroma
        self.client = chromadb.PersistentClient(path=str(path) if path is not None else str(chroma_cfg.path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name or chroma_cfg.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def ping(self) -> bool:
        """仅健康检查，探测Chroma连接，不做count全库扫描"""
        try:
            # 简单调用验证客户端/集合可用，轻量操作
            self.collection.peek(limit=1)
            return True
        except Exception:
            return False

    def add_vectors(
        self,
        chunk_ids: list[str],
        embeddings: list[EmbeddingVec],
        doc_ids: list[str],
        tenant_id: str,
        user_id: str,
    ) -> None:
        """
        添加向量。metadata 存入 doc_id、tenant_id、user_id。
        documents 为空字符串，文本由 SQLite MetaDB 管理。
        tenant_id / user_id 必传，由上层service传入。
        """
        metadatas: list[Metadata] = [{"doc_id": doc_id, "tenant_id": tenant_id, "user_id": user_id} for doc_id in doc_ids]
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[""] * len(chunk_ids),
        )
        logger.info(f"已写入 {len(chunk_ids)} 条向量到 Chroma, tenant={tenant_id}")

    def query(self, query_embedding: EmbeddingVec, tenant_id: str, top_k: int) -> list[dict]:
        """
        单个向量检索入口，底层强制增加tenant过滤兜底
        """
        groups = self.multi_query([query_embedding], tenant_id, top_k)
        if not groups:
            return []
        return groups[0]

    def multi_query(self, query_embeddings: list[EmbeddingVec], tenant_id: str, top_k: int) -> list[list[dict]]:
        """
        批量向量查询。强制带上tenant_id作为where过滤，底层租户隔离兜底。
        返回：外层列表长度等于输入向量数量，每组是对应召回结果。
        """
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={"tenant_id": tenant_id},  # 底层硬隔离：只召回当前租户向量
            include=["metadatas", "distances"],
        )
        ids = result["ids"]
        distances = result["distances"]
        metadatas = result["metadatas"]
        if ids is None or distances is None or metadatas is None:
            return []

        all_groups: list[list[dict]] = []
        for one_ids, one_dist, one_meta in zip(ids, distances, metadatas):
            group = []
            for cid, distance, meta in zip(one_ids, one_dist, one_meta):
                if meta is None:
                    continue
                group.append(
                    {
                        "chunk_id": cid,
                        "doc_id": meta.get("doc_id", ""),
                        "user_id": meta.get("user_id", ""),
                        "score": round(1 - distance, 4),
                    }
                )
            all_groups.append(group)
        return all_groups

    def query_with_filter(self, query_embedding: EmbeddingVec, filter: dict, top_k: int) -> list[dict]:
        """
        通用向量检索接口，filter由service组装（动态业务视图）。
        用于：管理员全租户内检索 / 普通用户只检索自己的向量。
        filter 必须由service内部硬编码构造，禁止直接透传前端参数。
        示例filter: {"tenant_id":"tid", "user_id":"uid"}
        """
        groups = self.multi_query_with_filter([query_embedding], filter, top_k)
        return groups[0] if groups else []

    def multi_query_with_filter(self, query_embeddings: list[EmbeddingVec], filter: dict, top_k: int) -> list[list[dict]]:
        """批量通用向量检索，接收service组装的where filter"""
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=filter,
            include=["metadatas", "distances"],
        )
        ids = result["ids"]
        distances = result["distances"]
        metadatas = result["metadatas"]
        if ids is None or distances is None or metadatas is None:
            return []

        all_groups: list[list[dict]] = []
        for one_ids, one_dist, one_meta in zip(ids, distances, metadatas):
            group = []
            for cid, distance, meta in zip(one_ids, one_dist, one_meta):
                if meta is None:
                    continue
                group.append(
                    {
                        "chunk_id": cid,
                        "doc_id": meta.get("doc_id", ""),
                        "user_id": meta.get("user_id", ""),
                        "score": round(1 - distance, 4),
                    }
                )
            all_groups.append(group)
        return all_groups

    def delete_by_doc_ids(self, doc_ids: list[str], tenant_id: str) -> None:
        """按doc_id + tenant_id 删除向量，底层租户兜底防护"""
        if not doc_ids:
            return
        for doc_id in doc_ids:
            self.collection.delete(where={"doc_id": doc_id, "tenant_id": tenant_id})
        logger.info(f"已删除 {len(doc_ids)} 个文档的向量，tenant={tenant_id}")

    def count(self, filter: dict | None = None) -> int:
        """
        统计向量数量。
        filter: 上层service组装过滤条件，不传则统计全部（运维场景）。
        """
        if filter is None or len(filter) == 0:
            return self.collection.count()
        res = self.collection.query(where=filter)
        return len(res)
