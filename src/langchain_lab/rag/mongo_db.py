"""
MongoDB 文档库封装：
- 存储文档原始内容、状态、元数据
- 使用 motor 异步驱动
- 同一集合内通过 _id 关联 Chroma 中的 chunks
"""

from datetime import datetime, timezone


class MongoDocumentStore:
    def __init__(self, uri: str = "mongodb://root:862697055@localhost:27017/?authSource=admin") -> None: ...
