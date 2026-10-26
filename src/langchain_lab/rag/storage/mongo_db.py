"""
MongoDB 存储层：管理原始文档全生命周期。
每个文档包含正文、文件名、索引状态、时间戳。
职责：仅做数据CRUD，不处理业务权限、角色逻辑。
"""

from datetime import UTC, datetime

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_lab.core.config.config_core import get_config


class MongoDocumentStore:
    """
    封装 MongoDB 的文档增删改查。
    数据模型：
    {
        "_id": ObjectId,
        "filename": str,
        "content": str,
        "content_hash": str,
        "tenant_id": str,
        "user_id": str,
        "status": "pending" | "indexed" | "failed",
        "error_msg": str,
        "chunk_count": int,
        "created_at": datetime,
        "updated_at": datetime
    }
    """

    def __init__(self, uri: str | None = None, db_name: str | None = None, collection: str | None = None):
        # 未显式传入则读统一配置 rag.mongo.*（原硬编码 USER_MONGO_URI/MONGO_DB/MONGO_COLLECTION 已收敛）
        mongo_cfg = get_config().rag.mongo
        self.client = MongoClient(uri or mongo_cfg.user_uri)
        self.db = self.client[db_name or mongo_cfg.db_name]
        self.collection: Collection = self.db[collection or mongo_cfg.collection]

    def insert_document(
        self,
        filename: str,
        content: str,
        content_hash: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        """插入原始文档，返回 _id 字符串。tenant_id/user_id 改为必传，去掉默认值"""
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

    def get_document(self, doc_id: str, tenant_id: str) -> dict | None:
        """按 _id + tenant_id 查询文档，返回完整文档（含 content），防止越权读取"""
        filter_spec = {"_id": ObjectId(doc_id), "tenant_id": tenant_id}
        return self.collection.find_one(filter_spec)

    def list_documents(self, filter: dict, skip: int = 0, limit: int = 20, projection: dict | None = None) -> list[dict]:
        """
        通用分页列表查询。
        filter: 由上层service组装好的查询条件（包含tenant_id基础隔离）
        projection: 字段投影，例如 {"content":0} 用来排除大文本字段
        返回：文档列表，_id转字符串
        """
        cursor = self.collection.find(filter, projection=projection).sort("created_at", -1).skip(skip).limit(limit)
        docs = []
        for doc in cursor:
            item = dict(doc)
            item["_id"] = str(item["_id"])
            docs.append(item)
        return docs

    def count_documents(self, filter: dict) -> int:
        """统计符合条件的文档总数，用于分页，filter由上层service组装"""
        return self.collection.count_documents(filter)

    def update_document_status(self, doc_id: str, tenant_id: str, status: str, chunk_count: int = 0, error_msg: str = "") -> None:
        """更新文档索引状态；必须校验tenant_id，防止跨租户篡改"""
        self.collection.update_one(
            {"_id": ObjectId(doc_id), "tenant_id": tenant_id},
            {
                "$set": {
                    "status": status,
                    "chunk_count": chunk_count,
                    "error_msg": error_msg,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

    def delete_document(self, doc_id: str, tenant_id: str) -> bool:
        """删除单篇文档，必须校验租户，返回是否删除成功"""
        result = self.collection.delete_one({"_id": ObjectId(doc_id), "tenant_id": tenant_id})
        return result.deleted_count > 0

    def delete_documents(self, doc_ids: list[str], tenant_id: str) -> int:
        """
        批量删除单个租户下多篇原始文档
        :param doc_ids: 文档_id字符串列表
        :param tenant_id: 租户ID，限定删除范围
        :return: 返回成功删除的文档数量
        """
        obj_ids = [ObjectId(did) for did in doc_ids if did]
        if not obj_ids:
            return 0
        res = self.collection.delete_many({"_id": {"$in": obj_ids}, "tenant_id": tenant_id})
        return res.deleted_count
