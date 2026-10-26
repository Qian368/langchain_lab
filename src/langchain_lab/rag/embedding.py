"""编码器组件：初始化编码器
模型路径/名称收敛到统一配置 rag.embedding.model（原硬编码已收敛）
"""

from sentence_transformers import SentenceTransformer

from langchain_lab.core.config.config_core import get_config

# 模块级缓存，避免重复加载
_model = None


def get_embedding_model(model_name: str | None = None):
    """
    返回编码器模型实例。
    第一次调用时加载模型，之后直接返回缓存实例
    """
    global _model

    if model_name is None:
        model_name = get_config().rag.embedding.model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_documents(text: list[str], model_name: str | None = None):
    """批量编码文档，返回向量列表"""
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=True)


def embed_query(text: str, model_name: str | None = None):
    """编码单条查询，返回向量"""
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=True)
