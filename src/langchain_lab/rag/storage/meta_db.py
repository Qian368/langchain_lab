"""
SQLite 存储层：管理 chunk 元数据与文本，以及文档与 chunk 的映射。
同时提供简易关键字检索（Bigram + LIKE），用于混合检索。
职责：仅做数据存取，内置底层租户隔离兜底约束；业务权限、角色逻辑上移至service。
"""

import sqlite3
from pathlib import Path
from typing import Any

from langchain_lab.core.config._config import META_DB_PATH


class MetaDB:
    """
    SQLite 封装。
    表结构：
    CREATE TABLE chunks (
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
    """

    def __init__(self, db_path: str | Path = META_DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ping(self) -> bool:
        """仅用于健康检查：验证SQLite连接正常"""
        try:
            with self._get_conn() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        批量插入 chunks。
        每个chunk必须携带 tenant_id, user_id。
        参数格式：
        [{
            "chunk_id": str,
            "doc_id": str,
            "tenant_id": str,
            "user_id": str,
            "content": str,
            "source": str,
            "section": str,
            "chunk_index": int
        }]
        """
        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, tenant_id, user_id, content, source, section, chunk_index)
                VALUES
                    (:chunk_id, :doc_id, :tenant_id, :user_id, :content, :source, :section, :chunk_index)
                """,
                chunks,
            )

    def get_chunks_by_ids(self, chunk_ids: list[str], tenant_id: str) -> list[dict]:
        """按 chunk_id 列表批量查询，增加租户兜底校验；结果保持传入顺序"""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        params = [*chunk_ids, tenant_id]
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM chunks
                WHERE chunk_id IN ({placeholders}) AND tenant_id = ?
                """,
                params,
            ).fetchall()
            row_map = {row["chunk_id"]: dict(row) for row in rows}
            return [row_map[cid] for cid in chunk_ids if cid in row_map]

    def get_chunk_ids_by_doc_id(self, doc_id: str, tenant_id: str) -> list[str]:
        """按 doc_id + tenant_id 查询该文档下全部 chunk_id（用于删除向量）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id FROM chunks
                WHERE doc_id = ? AND tenant_id = ?
                """,
                (doc_id, tenant_id),
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def delete_chunks_by_doc_id(self, doc_id: str, tenant_id: str) -> int:
        """按 doc_id + tenant_id 删除该文档所有chunk，返回删除行数；租户兜底防护"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                DELETE FROM chunks
                WHERE doc_id = ? AND tenant_id = ?
                """,
                (doc_id, tenant_id),
            )
            return cursor.rowcount

    def delete_chunks_by_doc_ids(self, doc_ids: list[str], tenant_id: str) -> int:
        """批量删除单个租户下多个doc_id对应的chunk，返回删除行数"""
        if not doc_ids:
            return 0
        placeholders = ", ".join(["?"] * len(doc_ids))
        sql = f"""
            DELETE FROM chunks
            WHERE doc_id IN ({placeholders}) AND tenant_id = ?
        """
        params = [*doc_ids, tenant_id]
        with self._get_conn() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def batch_delete_by_doc_ids(self, doc_ids: list[str], tenant_id: str, batch_size=900) -> int:
        """分批删除，规避IN参数过多；限定单个租户"""
        total = 0
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i : i + batch_size]
            total += self.delete_chunks_by_doc_ids(batch, tenant_id)
        return total

    def count_chunks(self, filter: dict[str, Any]) -> int:
        """
        通用统计，filter由上层service组装
        示例filter: {"tenant_id":"xxx", "doc_id":"yyy"}
        """
        if not filter:
            return 0
        keys = list(filter.keys())
        where_parts = [f"{k} = ?" for k in keys]
        values = [filter[k] for k in keys]
        sql = f"SELECT COUNT(*) AS cnt FROM chunks WHERE {' AND '.join(where_parts)}"
        with self._get_conn() as conn:
            row = conn.execute(sql, values).fetchone()
        return row["cnt"] if row else 0

    def keyword_search(self, tokens: list[str], tenant_id: str, limit: int = 10) -> list[dict]:
        """
        简易关键字检索：对每个 token 做 LIKE 匹配，按命中次数排序。
        强制带上tenant_id底层隔离，返回 [{"chunk_id", "doc_id", "source", "section", "content", "hits"}]
        """
        if not tokens:
            return []
        conditions = []
        params = {}
        hit_cases = []
        for i, token in enumerate(tokens):
            param = f"t{i}"
            conditions.append(f"content LIKE :{param}")
            hit_cases.append(f"(CASE WHEN content LIKE :{param} THEN 1 ELSE 0 END)")
            params[param] = f"%{token}%"
        # 追加租户过滤，底层硬隔离
        sql = f"""
            SELECT chunk_id, doc_id, source, section, content,
                   ({" + ".join(hit_cases)}) AS hits
            FROM chunks
            WHERE tenant_id = :tid AND ({" OR ".join(conditions)})
            ORDER BY hits DESC, length(content) ASC
            LIMIT :limit
        """
        params["tid"] = tenant_id
        params["limit"] = limit
        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            result = [dict(row) for row in rows]
            return result

    # 可选：通用列表查询方法（和mongo的list范式对齐，service组装filter）
    def list_chunks(
        self,
        filter: dict[str, Any],
        skip: int = 0,
        limit: int = 20,
    ) -> list[dict]:
        """
        通用chunk分页查询，filter由service组装。
        用于管理员/普通用户不同视图查询。
        """
        if not filter:
            return []
        keys = list(filter.keys())
        where_parts = [f"{k} = ?" for k in keys]
        values = [filter[k] for k in keys]
        sql = f"""
            SELECT * FROM chunks
            WHERE {" AND ".join(where_parts)}
            ORDER BY chunk_index ASC
            LIMIT ? OFFSET ?
        """
        values.extend([limit, skip])
        with self._get_conn() as conn:
            rows = conn.execute(sql, values).fetchall()
            return [dict(r) for r in rows]
