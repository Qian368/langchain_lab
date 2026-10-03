"""
SQLite 存储层：管理 chunk 元数据与文本，以及文档与 chunk 的映射。
同时提供简易关键字检索（Bigram + LIKE），用于混合检索。
"""

import sqlite3
from pathlib import Path

from langchain_lab.config import META_DB_PATH


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

    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        批量插入 chunks。
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

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """按 chunk_id 列表批量查询，结果保持传入顺序"""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        row_map = {row["chunk_id"]: dict(row) for row in rows}
        return [row_map[cid] for cid in chunk_ids if cid in row_map]

    def get_chunk_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """按 doc_id 查所有 chunk_id（用于删除向量）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有 chunk，返回删除条数"""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount

    def delete_chunks_by_doc_ids(self, doc_ids: list[str]) -> int:
        """批量删除多个 doc_ids 对应的 chunk, 返回删除行数"""
        if not doc_ids:
            return 0
        # 生成 ? 占位符，数量等于列表长度
        placeholders = ", ".join(["?"] * len(doc_ids))
        sql = f"DELETE FROM chunks WHERE doc_id IN ({placeholders})"
        with self._get_conn() as conn:
            cursor = conn.execute(sql, doc_ids)
            return cursor.rowcount

    def batch_delete_by_doc_ids(self, doc_ids: list[str], batch_size=900) -> int:
        """分批删除工具函数，避免IN参数过多报错"""
        total = 0
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i : i + batch_size]
            total += self.delete_chunks_by_doc_ids(batch)
        return total

    def count_chunks(self, doc_id: str | None = None) -> int:
        """统计 chunk 数量，可选按 doc_id 过滤"""
        if doc_id:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM chunks WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
        else:
            with self._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    def keyword_search(self, tokens: list[str], limit: int = 10) -> list[dict]:
        """
        简易关键字检索：对每个 token 做 LIKE 匹配，按命中次数排序。

        返回 [{"chunk_id", "doc_id", "source", "section", "content", "hits"}]
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

        # WHERE 先筛选， CASE 返回累加合计命中数，再排序
        sql = f"""
            SELECT chunk_id, doc_id, source, section, content,
                   ({" + ".join(hit_cases)}) AS hits
            FROM chunks
            WHERE {" OR ".join(conditions)}
            ORDER BY hits DESC, length(content) ASC
            LIMIT :limit
        """
        params["limit"] = limit

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            result = [dict(row) for row in rows]  # 在连接还活着的时候完成拷贝
        return result
