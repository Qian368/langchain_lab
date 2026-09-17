"""
最小 RAG 检索器：
1. 文档向量化
2. 查询向量化
3. 相似度排序，返回 TopK
4. 拼装 Prompt
"""

import numpy as np

from langchain_lab.rag.embedding import get_embedding_model
from langchain_lab.util.logger import logger


class SimpleRetriever:
    def __init__(self, documents: list[str], top_k: int = 3, least_score: float = 0.7) -> None:
        """
        document：知识库文档（已切分好的文本块）
        top_k：每次检索返回几条最相关的
        least_score：最低相似度分数过滤
        """

        self.documents = documents
        self.top_k = top_k
        self.least_score = least_score
        self.model = get_embedding_model()
        # 文档向量化一次，之后复用
        self.doc_vectors = self.model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )

    def retrieve(self, query: str) -> list[tuple[float, str]]:
        """输入问题，返回 [(相似度，文档片段), ...] 按相似度降序或者空列表"""
        query_vec = self.model.encode(query, normalize_embeddings=True)
        scores = self.doc_vectors @ query_vec

        mask = scores > self.least_score
        candidate_indices = np.flatnonzero(mask)

        if len(candidate_indices) == 0:
            # 返回空列表 + 打印提示，方便调试
            logger.warning(f"[Retriever] 未找到相似度超过 {self.least_score} 的文档")
            return []

        order = np.argsort(-scores[candidate_indices])[: self.top_k]
        top_indices = candidate_indices[order]

        # 组装结果
        result = [(float(scores[i]), self.documents[i]) for i in top_indices]
        return result

    def build_prompt(self, query: str) -> str:
        """检索 + 拼装 Prompt，直接给 LLM 用"""
        result = self.retrieve(query=query)
        context = "\n\n".join([doc for _, doc in result]) if result else "抱歉，知识库中暂无相关信息"
        prompt = f"""请根据以下资料回答问题：
资料：
{context}

问题：{query}"""
        return prompt
