"""
查询服务：
1. retrieve()：纯向量检索。
2. hybrid_retrieve()：向量 + 关键字混合检索，RRF（Reciprocal Rank Fusion）融合。
不接 LLM，只返回检索结果，方便下一步自行组装 Prompt。
"""

import re

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
    """RAG 检索入口"""

    def __init__(self):
        self.meta = MetaDB()
        self.vector = VectorStore()
        from langchain_lab.rag.embedding import get_embedding_model

        self.model = get_embedding_model()

    def _vector_search(self, question: str, limit: int) -> list[dict]:
        """纯向量检索，返回 [{"chunk_id": str}]，顺序即相关性排名"""
        embedding = (
            self.model.encode(
                [question],
                normalize_embeddings=True,
            )
            .astype("float32")
            .tolist()
        )  # embedding 类型：list[list[float]] = list[EmbeddingVec]
        hits = self.vector.query(embedding[0], top_k=limit)
        return [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]

    def _keyword_search(self, question: str, limit: int) -> list[dict]:
        """关键字检索，返回 [{"chunk_id": str}]，顺序即相关性排名"""
        tokens = simple_tokenize(question)
        if not tokens:
            return []
        rows = self.meta.keyword_search(tokens, limit=limit)
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
        """
        fused: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits):
            # key 也用chunk_id 的作用：下面可以直接 if cid in fused 去重，并直接叠加结果
            cid = hit["chunk_id"]
            fused[cid] = {
                "chunk_id": cid,
                "score": 1.3 / (k + rank + 1),  # 正常权重取1.0，这里取1.3侧重语义
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

    def _assemble(self, hits: list[dict]) -> list[dict]:
        """按命中顺序从 SQLite 查询完整信息并组装结果"""
        if not hits:
            return []

        chunk_ids = [h["chunk_id"] for h in hits]
        rows = self.meta.get_chunks_by_ids(chunk_ids)

        # 构建 id -> 行数据映射
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

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        """
        纯向量检索。

        返回格式：
        [{
            "chunk_id": str,
            "doc_id": str,
            "score": float,
            "source": str,
            "section": str,
            "content": str
        }]
        """
        hits = self._vector_search(question, limit=top_k)
        results = self._assemble(hits)
        logger.info(f"向量检索: {len(results)} 条")
        return results

    def hybrid_retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        """
        混合检索：向量 + 关键字，RRF 融合。

        返回格式与 retrieve() 一致。
        """
        vector_hits = self._vector_search(question, limit=top_k * 2)
        keyword_hits = self._keyword_search(question, limit=top_k * 2)
        logger.info(f"向量命中 {len(vector_hits)} 条，关键字命中 {len(keyword_hits)} 条")

        fused = self._rrf_fusion(vector_hits, keyword_hits, top_k=top_k)
        results = self._assemble(fused)
        logger.info(f"混合检索融合后: {len(results)} 条")
        return results
