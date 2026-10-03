"""
Chroma 向量存储层：只负责向量的增删查。
不存业务文本，metadata 只保留 doc_id 用于按文档删除。
"""

from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from langchain_lab.config import CHROMA_COLLECTION, CHROMA_PATH
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.RAG)

type EmbeddingVec = Sequence[float]


class VectorStore:
    """Chroma 封装"""

    def __init__(self, path: str | Path = CHROMA_PATH, collection_name: str = CHROMA_COLLECTION):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_vectors(self, chunk_ids: list[str], embeddings: list[EmbeddingVec], doc_ids: list[str]) -> None:
        """
        添加向量。只存 chunk_id、向量、doc_id。
        documents 为空字符串，因为文本由 SQLite 管理。
        """

        # embeddings_casted = cast(list[Sequence], embeddings)
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=[{"doc_id": doc_id} for doc_id in doc_ids],
            documents=[""] * len(chunk_ids),
        )
        logger.info(f"已写入 {len(chunk_ids)} 条向量到 Chroma")

    def query(self, query_embedding: EmbeddingVec, top_k: int) -> list[dict]:
        """
        单个向量检索（上层业务简易入口）
        内部复用 multi_query，仅做参数包装，底层只调用一次 chroma 查询。

        Args:
            query_embedding: 单个查询向量
            top_k: 每个查询向量召回的结果数量

        Returns:
            list[dict]: 当前查询向量召回的chunk列表，统一结构：
                [{"chunk_id": str, "doc_id": str, "score": float}, ...]
                score: 余弦相似度，值域 [0,1]，数值越高代表相似度越高
        """
        # 包装单个向量为列表，复用批量查询逻辑，取第 [0] 组结果
        groups = self.multi_query([query_embedding], top_k)
        if not groups:
            return []
        return groups[0]

    def multi_query(self, query_embeddings: list[EmbeddingVec], top_k: int) -> list[list[dict]]:
        """
        批量向量查询，直接对接Chroma底层接口，一次请求支持多个查询向量。
        充分利用Chroma原生批量检索能力，只执行一次collection.query，性能最优。

        Args:
            query_embeddings: 多个查询向量组成的列表
            top_k: 每一个查询向量各自召回的结果数量

        Returns:
            list[list[dict]]: 外层列表长度 = 输入查询向量数量
                每一个外层元素，对应一个查询向量的召回结果 list[dict]
                单条结果结构：{"chunk_id": str, "doc_id": str, "score": float}
                score: 余弦相似度，值域 [0,1]，数值越高代表相似度越高

        Notes:
            1. Chroma原始返回为列式双层嵌套数组，本方法完成扁平化转换，统一输出业务结构
            2. 会过滤掉meta为None的脏数据
            3. 距离转换规则：score = round(1 - distance,4)，Chroma返回的是cosine距离
        """
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        ids = result["ids"]
        distances = result["distances"]
        metadatas = result["metadatas"]
        # 空值守卫，消除Pylance reportOptionalSubscript警告
        if ids is None or distances is None or metadatas is None:
            return []

        all_groups: list[list[dict]] = []
        # 遍历每一组：一组对应一个查询向量的全部召回结果
        for one_ids, one_dist, one_meta in zip(ids, distances, metadatas):
            group = []
            for cid, distance, meta in zip(one_ids, one_dist, one_meta):
                # 元数据为空时跳过本条记录
                if meta is None:
                    continue
                group.append(
                    {
                        "chunk_id": cid,
                        "doc_id": meta.get("doc_id", ""),
                        "score": round(1 - distance, 4),
                    }
                )
            all_groups.append(group)
        return all_groups

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """按 doc_id 删除对应向量"""
        if not doc_ids:
            return
        for doc_id in doc_ids:
            self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"已删除 {len(doc_ids)} 个文档的向量")

    def count(self) -> int:
        """向量总数"""
        return self.collection.count()
