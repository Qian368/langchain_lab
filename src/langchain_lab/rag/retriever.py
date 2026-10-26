"""
最小 RAG 检索器（Document 三元组版）：
1. 文档向量化
2. 查询向量化
3. faiss 索引检索，返回 TopK
4. 拼装 Prompt
"""

import json
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from langchain_lab.core.config.config_core import get_config
from langchain_lab.rag.embedding import get_embedding_model
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.RAG)


class SimpleRetriever:
    def __init__(self, documents: list[str], top_k: int = 3, least_score: float = 0.7) -> None:
        """
        document：知识库文档（已切分好的文本块）
        top_k：每次检索返回几条最相关的
        least_score：最小相似度阈值，低于此值的文档不返回
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


@dataclass
class Document:
    """
    ## 切记：看着是文件，本质是Chunk对象，这个跟从社区的傻逼命名
    知识库文档三元组：ID + 文本 + 元数据
    如没有初始化id：则自动生成 uuid
    """

    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()).replace("-", ""))  # 自动生成ID
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """持久化用：转成可序列化的字典"""
        return {
            "text": self.text,
            "id": self.id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """持久化用：从字典恢复"""
        return cls(
            text=data["text"],
            id=data["id"],
            metadata=data.get("metadata", {}),
        )


class Retriever:
    def __init__(
        self,
        documents: list[Document],
        top_k: int = 3,
        least_score: float = 0.6,
    ) -> None:
        """
        - documents: 知识库文档列表（Document 三元组）
        - top_k: 每次检索返回几条最相关的
        - least_score: 最小相似度阈值，低于此值的文档不返回
        """

        # 校验：ID 不能重复
        ids = [doc.id for doc in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("文档 ID 存在重复，请检查后重新构建")

        self.documents = documents
        self.id_to_idx = {doc.id: i for i, doc in enumerate(documents)}
        self.top_k = top_k
        self.least_score = least_score
        self.model = get_embedding_model()
        self._build_index()

    def _build_index(self):
        """文档向量化 + 构建 faiss 索引"""
        texts = [doc.text for doc in self.documents]

        # 检查文本 token 长度是否超过模型限制
        count_error = self._check_token_lengths(texts)
        if (count_error is not None and count_error > 0) and input("请问还要继续执行向量化索引构建吗？（y/n）> ") == "n":
            raise ValueError("分割 chunk 部分文字过长，且用户取消 Retriever 初始化及构建")

        self.doc_vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        ).astype("float32")

        # 根据文档量选择 faiss 索引
        dim = self.doc_vectors.shape[1]
        if len(self.documents) < 10000:
            # 小规模：暴力检索，100%准确
            self.index = faiss.IndexFlatIP(dim)
        # 注意：以下两种情况：这里每次都训练策略，实践得定期或者到达阈值才训练或重建图
        elif len(self.documents) < 100000:
            # 中等规模：IVF，先聚类再搜索，速度提升但精度略降
            nlist = 100  # 聚类中心数，一般取 sqrt(文档数)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.doc_vectors)  # IVF 需要训练
        else:
            # 大规模：HNSW，图索引，速度最快但内存占用高
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 是每个节点的连接数

        self.index.add(self.doc_vectors)

    def save(self, dir_path: str | Path = "."):
        """保存索引和文档到磁盘， 默认根目录"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 1.保存 faiss 索引集合对象
        faiss.write_index(self.index, str(dir_path / "index.faiss"))

        # 2.保存文档和配置（用 pickle）
        meta = {
            "documents": [doc.to_dict() for doc in self.documents],
            "top_k": self.top_k,
            "least_score": self.least_score,
        }
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"索引已保存到 {dir_path}")

    @classmethod
    def load(cls, dir_path: str | Path, model_name: str | None = None) -> "Retriever":
        """从磁盘加载索引和文档（模型名未传则用统一配置 rag.embedding.model）"""
        if model_name is None:
            model_name = get_config().rag.embedding.model
        dir_path = Path(dir_path)

        # 加载文档和配置
        with open(dir_path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        # 1.创建实例但不构建索引
        obj = cls.__new__(cls)
        obj.documents = [Document.from_dict(doc) for doc in meta["documents"]]
        obj.id_to_idx = {doc.id: i for i, doc in enumerate(obj.documents)}
        obj.top_k = meta["top_k"]
        obj.least_score = meta["least_score"]
        obj.model = get_embedding_model(model_name)

        # 2.加载 faiss 索引集合对象
        obj.index = faiss.read_index(str(dir_path / "index.faiss"))

        logger.info(f"已从 {dir_path} 加载索引，共 {len(obj.documents)} 条文档")
        return obj

    # 这里只实现增，删改查再说吧~
    def add_documents(self, new_documents: list[Document]):
        """增量添加文档，只向量化新文档，不重新构建整个索引"""
        if not new_documents:
            return

        # 过滤已存在的 ID
        existing_ids = set(self.id_to_idx.keys())
        filtered_docs = []
        for doc in new_documents:
            if doc.id in existing_ids:
                logger.debug(f"文档 {doc.id} 已存在，跳过")
            else:
                filtered_docs.append(doc)

        if not filtered_docs:
            logger.warning("所有文档已存在，跳过")
            return

        # 向量化新文档
        new_texts = [doc.text for doc in filtered_docs]
        new_vectors = self.model.encode(
            new_texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        ).astype("float32")

        # 添加到 faiss 索引（faiss 会自动分配新 ID）
        self.index.add(new_vectors)

        # 更新文档列表和 ID 映射
        start_idx = len(self.documents)
        self.documents.extend(filtered_docs)
        for i, doc in enumerate(filtered_docs):
            self.id_to_idx[doc.id] = start_idx + i

        logger.info(f"已添加 {len(filtered_docs)} 条新文档，共 {len(self.documents)} 条")

    def retrieve(self, query: str) -> list[tuple[float, Document]]:
        """输入问题，返回 [(相似度，文档片段), ...] 按相似度降序或者空列表"""
        query_vec = self.model.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)

        # faiss 检索 TopK
        scores, indices = self.index.search(query_vec, self.top_k)

        # 过滤低于阈值的
        result = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # faiss 返回 -1 表示没有足够的结果
                break
            if score >= self.least_score:
                result.append((float(score), self.documents[idx]))

        if not result:
            logger.warning(f"[Retriever] 未找到相似度超过 {self.least_score} 的文档")
            return []

        return result

    def build_content(self, query: str) -> str:
        """检索 + 拼装 content，直接给 LLM 用"""
        result = self.retrieve(query=query)

        if not result:
            return """知识库中没有找到匹配的相关资料"""

        # 带元数据标注的上下文
        context_list = []
        for i, (score, doc) in enumerate(result, 1):
            # 如果元数据里有来源文件名，展示出来
            source = doc.metadata.get("source", "")
            if source:
                context_list.append(f"[{i}] 资料来源：{source}\n资料内容如下：\n{doc.text}")
            else:
                context_list.append(f"[{i}] 资料内容如下：\n{doc.text}")
        context = "\n\n".join(context_list)

        content = f"""
===== 检索参考资料开始 =====
{context}
===== 检索参考资料结束 =====
"""
        return content

    def build_json_content(self, query: str) -> str:
        """检索 + 拼装 json 格式的 content 文本，直接给 LLM 用"""
        result = self.retrieve(query=query)

        if not result:
            return """知识库中没有找到匹配的相关资料"""

        # 带元数据标注的上下文
        json_context_list = [
            {
                "doc_id": i,
                "source": doc.metadata.get("source", "未知来源"),
                "content": doc.text,
            }
            for i, (_, doc) in enumerate(result, 1)
        ]

        json_content = json.dumps(json_context_list, ensure_ascii=False, indent=2)
        return json_content

    def _check_token_lengths(self, texts: list[str]) -> int | None:
        """
        检查文本 token 长度是否超过模型最大限制。
        超长文本会被 SentenceTransformer 截断，导致向量信息缺失，
        这里提前告警，提示调整切分参数。
        """
        try:
            tokenizer = self.model.tokenizer
        except AttributeError:
            logger.warning("无法获取 tokenizer，跳过 token 长度检查")
            return

        # 从模型获取最大序列长度，默认 512
        max_length = getattr(self.model, "max_seq_length", 512)

        overlong = []
        for doc, text in zip(self.documents, texts):
            token_count = len(tokenizer.encode(text))
            if token_count > max_length:
                overlong.append((doc.id, token_count, doc.metadata.get("source", "")))

        if overlong:
            logger.warning(f"发现 {len(overlong)} 条文档超过模型最大 token 数 {max_length}，将被截断。建议调小 SplitConfig.chunk_size 或 overlap。")
            for doc_id, token_count, source in overlong[:5]:
                logger.warning(f" - 文档id:{doc_id} tokens:{token_count} sourse:{source}")
        else:
            logger.info(f"所有文档均在模型最大 token 限制 {max_length} 内")
        return len(overlong)


def serialize_ret_result(retriever_result: list[tuple[float, Document]]):
    """
    - 格式化 retrieve 返回结果，便于打印和传输
    - 可直接json.dumps(返回结果)
    """
    json_result = [{"score": f"{s:.6f}", "doc": doc.to_dict()} for s, doc in retriever_result]
    return json_result
