"""编码器组件：初始化编码器"""

from sentence_transformers import SentenceTransformer

# 模块级缓存，避免重复加载
_model = None


def get_embedding_model(model_name: str = "../models/bge-small-zh-v1.5"):
    """
    返回编码器模型实例。
    第一次调用时加载模型，之后直接返回缓存实例
    """
    global _model

    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_documents(text: list[str], model_name: str = "../models/bge-small-zh-v1.5"):
    """批量编码文档，返回向量列表"""
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=True)


def embed_query(text: str, model_name: str = "BAAI/bge-small-zh-v1.5"):
    """编码单条查询，返回向量"""
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=True)
