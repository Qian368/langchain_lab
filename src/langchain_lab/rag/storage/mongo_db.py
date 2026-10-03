"""
MongoDB 存储层：管理原始文档全生命周期。
每个文档包含正文、文件名、索引状态、时间戳。
"""

from datetime import UTC, datetime

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_lab.config import MONGO_COLLECTION, MONGO_DB, USER_MONGO_URI


class MongoDocumentStore:
    """
    封装 MongoDB 的文档增删改查。
    数据模型：
    {
        "_id": ObjectId,
        "filename": str,
        "content": str,
        "content_hash": str,
        "tenant_id": "unnknown_id",
        "user_id": "unknown_id",
        "status": "pending" | "indexed" | "failed",
        "error_msg": str,
        "chunk_count": int,
        "created_at": datetime,
        "updated_at": datetime
    }
    """

    def __init__(self, uri: str = USER_MONGO_URI, db_name: str = MONGO_DB):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[MONGO_COLLECTION]

    def insert_document(
        self,
        filename: str,
        content: str,
        content_hash: str,
        tenant_id: str = "unknown_id",
        user_id: str = "unknown_id",
    ) -> str:
        """插入原始文档，返回 _id 字符串"""
        now = datetime.now(UTC)
        doc = {
            "filename": filename,
            "content": content,
            "content_hash": content_hash,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "status": "pending",
            "error_msg": "",
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def get_document(self, doc_id: str) -> dict | None:
        """按 _id 查询文档，返回完整文档（含 content）"""
        return self.collection.find_one({"_id": ObjectId(doc_id)})

    def list_documents(self, skip: int = 0, limit: int = 20, status: str | None = None) -> list[dict]:
        """
        文档列表，支持分页和状态过滤。

        返回的文档不含 content，避免列表接口传输全文。
        """
        query = {"status": status} if status else {}
        docs = []
        for doc in self.collection.find(query).sort("created_at", -1).skip(skip).limit(limit):
            doc = dict(doc)
            doc["_id"] = str(doc["_id"])
            doc.pop("content", None)
            docs.append(doc)
        return docs

    def update_document_status(self, doc_id: str, status: str, chunk_count: int = 0, error_msg: str = "") -> None:
        """更新文档索引状态"""
        self.collection.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "status": status,
                    "chunk_count": chunk_count,
                    "error_msg": error_msg,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

    def delete_document(self, doc_id: str) -> bool:
        """删除文档记录，返回是否删除成功"""
        result = self.collection.delete_one({"_id": ObjectId(doc_id)})
        return result.deleted_count > 0

    def delete_documents(self, doc_ids: list[str]) -> int:
        """
        批量删除多篇原始文档
        :param doc_ids: 文档_id字符串列表
        :return: 返回成功删除的文档数量
        """
        # 把字符串转ObjectId，过滤空值
        obj_ids = [ObjectId(did) for did in doc_ids if did]
        if not obj_ids:
            return 0
        res = self.collection.delete_many({"_id": {"$in": obj_ids}})
        return res.deleted_count
