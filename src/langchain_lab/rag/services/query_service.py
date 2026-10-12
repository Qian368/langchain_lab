"""
查询服务：
1. retrieve()：纯向量检索。
2. hybrid_retrieve()：向量 + 关键字混合检索，RRF（Reciprocal Rank Fusion）融合。
不接 LLM，只返回检索结果，方便下一步自行组装 Prompt。
业务规则：由上层传入租户/用户信息。
- _build_vector_filter：产出 Chroma where 语法
- _build_meta_filter：产出 SQLite 扁平KV过滤条件
两套存储的过滤构造分开，不混用。
"""

import re
from typing import Any

from langchain_lab.rag.storage import MetaDB, VectorStore
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.RAG)


def simple_tokenize(text: str) -> list[str]:
    """
    简易分词：
    - 英文/数字按单词提取
    - 中文按连续串做 Bigram 切分
    示例："Python装饰器怎么用"
      → ["Python", "装饰", "饰器", "怎么", "么用"]
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    chinese_parts = re.findall(r"[\u4e00-\u9fff]+", text)
    for part in chinese_parts:
        if len(part) <= 1:
            tokens.append(part)
        else:
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    return tokens


class QueryService:
    """RAG 检索入口，编排向量检索、关键词检索、RRF融合、结果组装"""

    def __init__(self):
        self.meta = MetaDB()
        self.vector = VectorStore()
        from langchain_lab.rag.embedding import get_embedding_model

        self.model = get_embedding_model()

    def _build_vector_filter(self, tenant_id: str, user_id: str, is_tenant_admin: bool) -> dict[str, Any]:
        """【向量库Chroma专用】构造where过滤条件，支持$and操作符"""
        if is_tenant_admin:
            return {"tenant_id": tenant_id}
        return {
            "$and": [
                {"tenant_id": tenant_id},
                {"user_id": user_id},
            ]
        }

    def _build_meta_filter(self, tenant_id: str, user_id: str, is_tenant_admin: bool) -> dict[str, Any]:
        """【元数据库SQLite专用】扁平KV过滤，无Chroma操作符，用于关键词检索"""
        if is_tenant_admin:
            return {"tenant_id": tenant_id}
        return {"tenant_id": tenant_id, "user_id": user_id}

    def _vector_search(self, question: str, filter: dict, limit: int) -> list[dict]:
        """纯向量检索，使用业务filter前置过滤，返回 [{"chunk_id": str, "score": float}]"""
        embedding = (
            self.model.encode(
                [question],
                normalize_embeddings=True,
            )
            .astype("float32")
            .tolist()
        )
        hits = self.vector.query_with_filter(embedding[0], filter=filter, top_k=limit)
        return [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]

    def _keyword_search(self, question: str, meta_filter: dict, limit: int) -> list[dict]:
        """关键字检索，使用meta过滤条件，返回 [{"chunk_id": str}]"""
        tokens = simple_tokenize(question)
        if not tokens:
            return []
        # 把meta_filter传给meta层，后续meta层可以基于这个字典拼接参数化SQL
        tenant_id = meta_filter["tenant_id"]
        rows = self.meta.keyword_search(tokens, tenant_id=tenant_id, limit=limit)
        return [{"chunk_id": r["chunk_id"]} for r in rows]

    def _rrf_fusion(
        self,
        vector_hits: list[dict],
        keyword_hits: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion：
        score = sum(1 / (k + rank))，rank 从 1 开始。
        k 是平滑常数，默认 60。
        k 越小侧重“偏科”，偏科更好。k越大侧重“均衡”，均衡更好。
        向量结果权重1.3，关键词1.0，优先语义。
        """
        fused: dict[str, dict] = {}
        for rank, hit in enumerate(vector_hits):
            cid = hit["chunk_id"]
            fused[cid] = {
                "chunk_id": cid,
                "score": 1.3 / (k + rank + 1),
            }
        for rank, hit in enumerate(keyword_hits):
            cid = hit["chunk_id"]
            if cid in fused:
                fused[cid]["score"] += 1.0 / (k + rank + 1)
            else:
                fused[cid] = {
                    "chunk_id": cid,
                    "score": 1.0 / (k + rank + 1),
                }
        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _assemble(self, hits: list[dict], tenant_id: str) -> list[dict]:
        """按命中顺序从 SQLite 查询完整信息并组装结果；带上租户过滤兜底"""
        if not hits:
            return []
        chunk_ids = [h["chunk_id"] for h in hits]
        rows = self.meta.get_chunks_by_ids(chunk_ids, tenant_id=tenant_id)
        row_map = {r["chunk_id"]: r for r in rows}
        results = []
        for hit in hits:
            cid = hit["chunk_id"]
            if cid not in row_map:
                continue
            row = row_map[cid]
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "score": round(hit["score"], 4),
                    "source": row["source"],
                    "section": row["section"],
                    "content": row["content"],
                }
            )
        return results

    def retrieve(self, question: str, tenant_id: str, user_id: str, is_tenant_admin: bool, top_k: int = 4) -> list[dict]:
        """纯向量检索。"""
        vector_filter = self._build_vector_filter(tenant_id, user_id, is_tenant_admin)
        hits = self._vector_search(question, filter=vector_filter, limit=top_k)
        results = self._assemble(hits, tenant_id=tenant_id)
        logger.info(f"向量检索: {len(results)} 条, tenant={tenant_id}")
        return results

    def hybrid_retrieve(self, question: str, tenant_id: str, user_id: str, is_tenant_admin: bool, top_k: int = 4) -> list[dict]:
        """混合检索：向量 + 关键字，RRF 融合。"""
        vector_filter = self._build_vector_filter(tenant_id, user_id, is_tenant_admin)
        meta_filter = self._build_meta_filter(tenant_id, user_id, is_tenant_admin)

        fetch_limit = top_k * 2
        vector_hits = self._vector_search(question, filter=vector_filter, limit=fetch_limit)
        keyword_hits = self._keyword_search(question, meta_filter=meta_filter, limit=fetch_limit)

        logger.info(f"混合检索：向量命中 {len(vector_hits)} 条，关键字命中 {len(keyword_hits)} 条, tenant={tenant_id}")
        fused = self._rrf_fusion(vector_hits, keyword_hits, top_k=top_k)
        results = self._assemble(fused, tenant_id=tenant_id)
        logger.info(f"混合检索融合后: {len(results)} 条")
        return results
