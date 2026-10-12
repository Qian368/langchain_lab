import json
import sqlite3
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from sqlite3 import OperationalError
from typing import Any

import chromadb
from bson import ObjectId, json_util
from colorama import init
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_lab.core.config._config import ADMIN_MONGO_URI, BACKUP_DIR, CHROMA_COLLECTION, CHROMA_PATH, META_DB_PATH, MONGO_COLLECTION, MONGO_DB
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.OPERATE)
init(autoreset=True)


class MongoOperation:
    def __init__(self, uri: str = ADMIN_MONGO_URI, db_name: str = MONGO_DB):
        """Mongo底层运维操作类，仅提供存储原语，不实例化业务Service"""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[MONGO_COLLECTION]

    def init_index(self):
        """
        安全初始化/修改索引
        初始化Mongo业务索引，仅首次部署/新增索引时手动执行；
        定义列表统一维护索引，后续新增/修改索引只改下方索引配置
        """
        coll = self.collection
        idx_info = coll.index_information()
        print("当前已有索引：", list(idx_info.keys()))
        # ========== 索引配置清单，只在这里维护 ==========
        index_defs: list[tuple[str, list[tuple[str, int]], bool, dict[str, Any]]] = [
            # (索引名称, 索引key（字段名，排序方向）列表, 是否唯一, 额外参数)
            ("idx_tenant_content_hash", [("tenant_id", 1), ("content_hash", 1)], True, {"background": True}),
            ("status_idx", [("status", 1)], False, {"background": True}),
            # 新增索引，直接在这里追加
            # ("xxx_idx", [(field,1)], False, {"background":True})
        ]
        for idx_name, keys, is_unique, opts in index_defs:
            if idx_name not in idx_info:
                coll.create_index(keys, unique=is_unique, name=idx_name, **opts)
                logger.info(f"✅ 创建索引 {idx_name}, keys={keys}, unique={is_unique}")
            else:
                logger.info(f"⏭️ 索引 {idx_name} 已存在，跳过")

    def show_and_get_indexes(self) -> list[str]:
        """打印并返回索引名列表"""
        print("=== 当前 MongoDB 索引列表 ===")
        index_names: list[str] = []
        for idx in self.collection.list_indexes():
            print(idx)
            print()
            index_names.append(idx["name"])
        return index_names

    def drop_index_by_name(self, index_names: list[str]):
        """批量按索引名称删除Mongo索引，传入索引名列表"""
        info = self.collection.index_information()
        for name in index_names:
            if name not in info:
                print(f"⏭️ 索引 {name} 不存在，跳过")
                continue
            self.collection.drop_index(name)
            print(f"🗑️ 索引 {name} 已删除")

    def find_duplicate_filename(self, tenant_id: str) -> list[dict[str, Any]]:
        """
        在**指定租户内**查找重复文件名
        :param tenant_id: 目标租户ID，强制租户隔离，禁止全库扫描
        """
        pipeline = [
            {"$match": {"tenant_id": tenant_id}},
            {"$group": {"_id": "$filename", "count": {"$sum": 1}, "ids": {"$push": "$_id"}}},
            {"$match": {"count": {"$gt": 1}}},
        ]
        return list(self.collection.aggregate(pipeline))

    def get_keep_and_remove_ids(self, dup_group: dict) -> tuple[ObjectId, list[ObjectId]]:
        """
        输入聚合返回的一组重复记录，选出保留id、待删除 id 列表
        业务逻辑：保留最新一条，删掉旧的重复条目。
        dup_group: {"_id": filename, "count":3, "ids":[ObjectId...]}
        return: keep_id, remove_ids
        """
        ids = dup_group["ids"]
        # ObjectID 直接比较大小，小=更早创建
        ids_sorted = sorted(ids, reverse=True)
        keep_id = ids_sorted[0]
        remove_ids = ids_sorted[1:]
        return keep_id, remove_ids


class MongoBackup:
    def __init__(
        self,
        uri: str = ADMIN_MONGO_URI,
        db_name: str = MONGO_DB,
        collection_name: str = MONGO_COLLECTION,
        backup_dir: str | Path = BACKUP_DIR,
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_collection(self, query: dict | None = None, label: str = ""):
        """
        备份集合数据 + 索引定义到 JSON 文件
        :param query: 查询条件，默认备份全部
        :param label: 备份标识，例如 to_delete，会作为文件名前缀
        :return: 备份的数据文件路径。索引文件路径 = 数据文件路径 + ".index.json"
        """
        query = query or {}
        docs = list(self.collection.find(query))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{label}_" if label else ""
        file_path = self.backup_dir / f"{prefix}{self.db_name}_{self.collection_name}_{timestamp}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, default=json_util.default, ensure_ascii=False, indent=2)
        # 备份索引
        index_file_path = Path(str(file_path) + ".index.json")
        indexes = self.collection.index_information()
        with open(index_file_path, "w", encoding="utf-8") as f:
            json.dump(indexes, f, default=json_util.default, ensure_ascii=False, indent=2)
        print(f"备份完成：{file_path}，共 {len(docs)} 条数据")
        print(f"索引定义已保存：{index_file_path}，共 {len(indexes) - 1} 个用户索引")
        return file_path

    def restore_collection(
        self,
        backup_file: str | Path,
        target_db_name: str | None = None,
        target_collection_name: str | None = None,
        overwrite: bool = False,
        restore_indexes: bool = True,
    ):
        """
        从备份 JSON 文件恢复数据到指定集合。
        :param backup_file: 数据备份文件路径（backup_collection 的返回值）
        :param target_db_name: 目标数据库名，默认使用初始化时的 db_name
        :param target_collection_name: 目标集合名，默认使用初始化时的 collection_name
        :param overwrite: True 则先清空目标集合中的现有数据再插入
        :param restore_indexes: 是否恢复索引（默认 True）
        """
        target_db = target_db_name or self.db_name
        target_col = target_collection_name or self.collection_name
        # 读取数据
        with open(backup_file, encoding="utf-8") as f:
            docs = json.load(f, object_hook=json_util.object_hook)
        db = self.client[target_db]
        collection: Collection = db[target_col]
        # 覆盖模式
        if overwrite:
            result = collection.delete_many({})
            print(f"已清空目标集合 {target_db}.{target_col}，删除 {result.deleted_count} 条现有数据")
        # 插入数据
        if docs:
            collection.insert_many(docs, ordered=False)
            print(f"数据恢复完成：已插入 {len(docs)} 条到 {target_db}.{target_col}")
        else:
            print("备份文件中没有数据，跳过插入")
        # 恢复索引
        if restore_indexes:
            index_file = Path(str(backup_file) + ".index.json")
            if index_file.exists():
                with open(index_file, encoding="utf-8") as f:
                    indexes = json.load(f, object_hook=json_util.object_hook)
                self._restore_indexes(collection, indexes)
            else:
                print("未找到索引备份文件，跳过索引恢复")

    def _restore_indexes(self, collection: Collection, indexes: dict):
        """根据索引备份信息重建索引"""
        created_count = 0
        existing_indexes = collection.index_information()
        for name, idx_info in indexes.items():
            if name == "_id_":
                continue
            # 如果同名索引已存在，跳过避免冲突
            if name in existing_indexes:
                print(f"索引 [{name}] 已存在，跳过")
                continue
            key = idx_info["key"]
            # create_index 不接受 ns、v 等内部字段，过滤掉
            options = {k: v for k, v in idx_info.items() if k not in ("key", "name", "ns", "v")}
            collection.create_index(key, name=name, **options)
            created_count += 1
        print(f"索引恢复完成，新建 {created_count} 个索引")


class SqliteOperation:
    def __init__(self, db_path: str | Path = META_DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ping(self) -> bool:
        """健康检查轻量探测，仅验证连通性，不做统计"""
        try:
            with self._get_conn() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def init_db(self):
        """建表 + 追加字段 + 创建索引（如果不存在）"""
        with self._get_conn() as conn:
            # 1. 初始化表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    section TEXT DEFAULT '',
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("chunks 表结构已校验（CREATE TABLE IF NOT EXISTS）")
            # 2. 变更表结构
            # 只在列不存在的时候添加
            with suppress(OperationalError):
                conn.execute("ALTER TABLE chunks ADD COLUMN tenant_id TEXT DEFAULT 'unknown_id'")
                logger.info("✅ 成功新增字段 tenant_id")
            with suppress(OperationalError):
                conn.execute("ALTER TABLE chunks ADD COLUMN user_id TEXT DEFAULT 'unknown_id'")
                logger.info("✅ 成功新增字段 user_id")
            # 3. 变更索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            logger.info("索引 idx_chunks_doc_id 校验完成")
            # ✅ 新增复合索引：优先tenant_id，租户隔离，再按doc_id查找分片
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tenant_doc ON chunks(tenant_id, doc_id)")
            logger.info("复合索引 idx_chunks_tenant_doc (tenant_id,doc_id) 校验完成")
        logger.info("✅ SQLite chunks 库初始化全部完成")

    def drop_indexes(self, index_list: list[str]):
        """批量删除chunks表相关索引"""
        with self._get_conn() as conn:
            for idx in index_list:
                conn.execute(f"DROP INDEX IF EXISTS {idx}")
                logger.info(f"✅ 索引 {idx} 删除完成")
        logger.info("✅ chunks表所有业务索引清理完毕")

    def count_all_chunks(self) -> int:
        """【仅本地调试，生产禁用！全表count，大数据量会卡顿】"""
        sql = "SELECT COUNT(*) AS cnt FROM chunks"
        with self._get_conn() as conn:
            row = conn.execute(sql).fetchone()
            return row["cnt"] if row else 0


class VectorOperation:
    def __init__(self, path: str | Path = CHROMA_PATH, collection_name: str = CHROMA_COLLECTION):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ping(self) -> bool:
        """健康检查轻量探测，peek少量样本，不做全量count"""
        try:
            self.collection.peek(limit=1)
            return True
        except Exception:
            return False
