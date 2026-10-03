"""运维脚本：发现并删除重复的文档并同步更新异构数据库"""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from bson import ObjectId
from colorama import Fore, init

from langchain_lab.rag.operation.base import MongoBackup, MongoOperation, SqliteOperation, VectorOperation
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.OPERATE)
init(autoreset=True)


# region MongoDB
def route_temp():
    mongo_db = MongoOperation()  # noqa: F841
    sql_db = SqliteOperation()  # noqa: F841

    sql_db.drop_indexes(["sqlite_autoindex_chunks_1"])
    sql_db.init_db()


def route_remove_duplication():
    """找到重复文件，同步删除向量数据库、关系型数据、文档型数据库的重复数据"""
    operation = MongoOperation()
    backup = MongoBackup()

    _ = operation.show_and_get_indexes()
    dup_list = operation.find_duplicate_filename()
    print(f"发现重复文件数量：{len(dup_list)}")
    for item in dup_list:
        print(item)

    all_remove_ids: list[ObjectId] = []
    for item in dup_list:
        filename = item["_id"]
        keep_id, remove_ids = operation.get_keep_and_remove_ids(item)
        print(f"\n==== 文件：{filename} ====")
        print(f"✅ 保留doc_id: {keep_id}")
        print(f"🗑️ 待删除doc_ids: {remove_ids}")
        all_remove_ids.extend(remove_ids)

    print(f"\n汇总所有待删除doc_id列表（共{len(all_remove_ids)}项）：{all_remove_ids}")
    print(Fore.GREEN, "==== 以上为 DRY RUN 模式，仅打印，不会删除任何数据 ====")

    # ========== 确认无误后，再打开下面这段执行 ==========
    run_operation = True
    if run_operation:  # noqa: SIM102
        # ===== 执行真正的数据库操作 =====
        if len(all_remove_ids) > 0:
            confirm = input(Fore.RED + f"确认删除 {len(all_remove_ids)} 条重复文档？输入 YES 继续：")
            if confirm.strip().lower() != "yes":
                print("操作终止")
                return

            # 备份 + 删除逻辑
            # 1. 先备份待删除的全部文档
            backup.backup_collection(query={"_id": {"$in": all_remove_ids}}, label="delete_dup")
            # 2. 批量同步删除
            del_cnt = operation.batch_delete_dup_docs(all_remove_ids)
            print(f"MongoDB删除完成，删除 {del_cnt} 条文档")


def route_modify_index():
    """删除旧业务索引，重新初始化自定义索引（_id系统索引保留）"""
    mongo_db = MongoOperation()
    # 打印当前索引
    mongo_db.show_and_get_indexes()
    # 删除旧自动生成的索引
    mongo_db.drop_index_by_name(["filename_unique"])
    # 创建新命名索引，按初始化标准更新索引，安全更新
    mongo_db.init_index()
    # 再打印一遍，确认结果
    mongo_db.show_and_get_indexes()


def route_remove_uuid_filename():
    """查找并批量删除filename是uuid4的旧文档"""
    mongo_db = MongoOperation()
    backup = MongoBackup()

    # uuid4正则
    pattern = r"^[0-9a-f]{32}\.md$"
    cursor = mongo_db.collection.find({"filename": {"$regex": pattern}})
    docs = list(cursor)
    print(f"找到 {len(docs)} 条filename是uuid4的文档")
    for d in docs:
        print(f"doc_id: {d['_id']}, filename: {d['filename']}")
    doc_ids = [d["_id"] for d in docs]
    if not doc_ids:
        print("没有找到uuid格式filename文档，结束")
        return

    print(f"\n汇总待删除doc_id，共 {len(doc_ids)} 条")
    print(Fore.GREEN, "==== DRY RUN，仅打印，未删除 ====")

    run_operation = True
    if run_operation:
        confirm = input(Fore.RED + f"确认删除 {len(doc_ids)} 条UUID文件名文档？输入 YES 继续：")
        if confirm.strip().lower() != "yes":
            print("操作终止")
            return
        # 备份
        backup.backup_collection(query={"_id": {"$in": doc_ids}}, label="delete_uuid_filename")
        # 批量删除Mongo + SQLite + Chroma
        del_cnt = mongo_db.batch_delete_dup_docs(doc_ids)
        print(f"MongoDB删除完成，删除 {del_cnt} 条文档")


def route_backfill_tenant_user_hash_from_content():
    """
    运维回填脚本：给存量文档补 tenant_id, user_id, content_hash
    直接读取文档内content字段计算哈希，不需要访问磁盘文件
    筛选：缺少tenant_id的旧文档
    """
    mongo_db = MongoOperation()
    coll = mongo_db.collection
    filter_query = {"tenant_id": {"$exists": False}}
    docs = list(coll.find(filter_query))
    logger.info(f"待回填文档总数：{len(docs)}")

    for doc in docs:
        doc_id = doc["_id"]
        filename = doc.get("filename", "")
        content = doc.get("content", "")

        try:
            content_hash = _calc_text_sha256(content)
        except Exception as e:
            logger.info(f"❌ 计算hash失败 doc={doc_id} filename={filename}: {e}")
            content_hash = "sha256:content_error"

        update_data = {"$set": {"tenant_id": "unknown_id", "user_id": "unknown_id", "content_hash": content_hash}}
        coll.update_one({"_id": doc_id}, update_data)
        logger.info(f"✅ 更新 doc {doc_id}, filename={filename}, hash={content_hash}")

    logger.info("✅ 回填任务全部结束")
    # 校验：统计还有多少文档没有tenant_id，必须=0
    remain = coll.count_documents({"tenant_id": {"$exists": False}})
    logger.info(f"剩余未回填文档数量: {remain}")


def route_backfill_chroma_metadata():
    _backfill_chroma_metadata()


def _backfill_chroma_metadata(dry_run: bool = True):
    """
    一次性运维脚本：Chroma存量向量回填 tenant_id / user_id
    直接原生SQL查询，不调用业务封装函数
    前提：
        1. Chroma的id就是chunk_id（你当前的存储方式）
        2. SQLite chunks表已回填 tenant_id / user_id
    """
    vector_db = VectorOperation()
    coll = vector_db.collection

    # 取出全部chunk_id（Chroma的ids就是chunk_id），不加载向量
    all_ids_res = coll.get(include=[])
    all_chunk_ids: list[str] = all_ids_res["ids"]
    total = len(all_chunk_ids)
    logger.info(f"Chroma 待处理向量总数：{total}")
    if total == 0:
        logger.info("Chroma无数据，直接退出回填")
        return

    batch_del_ids: list[str] = []
    new_ids: list[str] = []
    new_embeddings: list[Sequence[float]] = []
    new_metadatas: list[Mapping[str, Any]] = []

    batch_size = 50
    for start in range(0, total, batch_size):
        batch_chunk_ids = all_chunk_ids[start : start + batch_size]
        logger.info(f"处理批次 {start} ~ {start + batch_size}, 数量={len(batch_chunk_ids)}")

        # 读取这批chunk_id对应的向量和旧metadata
        batch_data = coll.get(ids=batch_chunk_ids, include=["embeddings", "metadatas"])
        ids_batch = batch_data["ids"]
        emb_batch = batch_data["embeddings"]

        if emb_batch is None:
            logger.info(f"跳过，批次 {start} ~ {start + batch_size}, 无向量字段，无法执行回填")
            logger.info(f"本批次向量id 或 chunk_id 包括：{ids_batch}")
        else:
            sqlite = SqliteOperation()
            with sqlite._get_conn() as conn:
                for idx, cid in enumerate(ids_batch):
                    emb = emb_batch[idx]
                    emb_list = list(emb)
                    row = conn.execute("""SELECT doc_id, tenant_id, user_id FROM chunks WHERE chunk_id = ?""", (cid,)).fetchone()
                    if not row:
                        logger.warning(f"跳过：chunk_id={cid} 在SQLite找不到记录")
                        continue

                    tenant_id = row["tenant_id"]
                    user_id = row["user_id"]
                    doc_id = row["doc_id"]

                    batch_del_ids.append(cid)
                    new_ids.append(cid)
                    new_embeddings.append(emb_list)
                    new_meta = {"doc_id": doc_id, "tenant_id": tenant_id, "user_id": user_id}
                    new_metadatas.append(new_meta)
                    logger.info(f"预览 chunk_id={cid}, doc_id={doc_id}, tenant={tenant_id}, user={user_id}")

        logger.info(f"汇总：待删除{len(batch_del_ids)}条，待重写{len(new_ids)}条")
    if dry_run:
        logger.info("=== DRY RUN模式，不执行删除和写入，结束 ===")
        return

    if batch_del_ids:
        coll.delete(ids=batch_del_ids)
        logger.info(f"✅ 删除旧向量 {len(batch_del_ids)}")
    if new_ids:
        coll.add(ids=new_ids, embeddings=new_embeddings, metadatas=new_metadatas, documents=[""] * len(new_ids))
        logger.info(f"✅ 重新插入带租户元数据向量 {len(new_ids)}")

    logger.info("✅ Chroma元数据回填任务完成")


def _calc_text_sha256(text: str) -> str:
    """对文本字符串计算sha256"""
    h = hashlib.sha256()
    # 字符串转utf8字节
    h.update(text.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


# endregion

# region Sqlite


# endregion


# ========== 脚本入口，路由执行 ==========
def temp():
    pass


# region
import inspect
import sys

import questionary


def main():
    # 动态获取路由函数，也可以传 prefix="" 获取所有公开函数
    routes = get_current_functions(prefix="route_")

    if not routes:
        print("未找到任何符合条件的函数")
        return

    # 使用 questionary 动态呈现函数名列表
    choice = questionary.select("请选择要执行的路由函数：", choices=list(routes.keys())).ask()

    if not choice:
        return

    # 确认执行
    if questionary.confirm(f"确认执行 {choice} 吗？").ask():
        routes[choice]()
    else:
        print("已取消")


def get_current_functions(prefix="route_"):
    """
    获取当前模块中定义且名称符合指定前缀的函数。
    prefix 为空字符串时，获取所有当前模块定义的非下划线开头函数。
    """
    current_module = sys.modules[__name__]
    funcs = {}

    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        # 只保留定义在当前模块内的函数，排除 import 进来的
        if func.__module__ == __name__ and ((prefix and name.startswith(prefix)) or (not prefix and not name.startswith("_"))):
            funcs[name] = func

    return funcs


if __name__ == "__main__":
    temp()
# endregion
