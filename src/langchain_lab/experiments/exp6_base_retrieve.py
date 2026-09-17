"""实验6：文本向量化于检索"""

from langchain_lab.rag.embedding import embed_documents, embed_query, get_embedding_model

# 1. 直接拿模型实例自己操作
model = get_embedding_model()
docs = ["养老金申领需要达到法定退休年龄", "申领时需要连续缴费满15年", "我9月1号才开学"]
docs_vecs = model.encode(docs, normalize_embeddings=True)

# 2. 用封装好的函数
docs_vecs = embed_documents(docs)
query_vec = embed_query("我缴费多少年才能领养老金？")

scores = docs_vecs @ query_vec  # 求相似度
ranked = sorted(zip(scores, docs), reverse=True)  # 排序
print(ranked)
