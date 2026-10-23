"""SessionStore：SQLite 持久化（schema 面向 SaaS 检索 / 压缩归档 / 会话分支 预留）

- 序列化用 LangChain messages_to_dict / messages_from_dict，tool_call_id 无损往返
- sqlite3 同步库，读写用 asyncio.to_thread 包装，不阻塞事件循环
- schema 预留：
  - sessions.summary（LLM 压缩摘要，供 LLM 视图置顶；各分支共享）
  - messages.turn_id（最近 N 轮界定）、messages.branch_id + seq（分支拐点）
  - sessions.user_id/tenant_id（SaaS 检索）
"""
import asyncio
import json
import sqlite3
import time
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class SessionStore:
    """SQLite 会话存储：save 整轮/整会话，load 按分支读回"""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT, tenant_id TEXT,
                    title TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    created_at REAL, updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    branch_id TEXT DEFAULT 'root',
                    seq INTEGER,
                    turn_id INTEGER DEFAULT 0,
                    role TEXT,
                    payload TEXT,
                    archived INTEGER DEFAULT 0,
                    UNIQUE(session_id, branch_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                """
            )

    # —— 保存 ——
    async def save(
        self,
        session_id: str,
        messages: list[BaseMessage],
        session_ctx: dict | None = None,
        title: str | None = None,
        branch_id: str = "root",
    ) -> None:
        """整轮/整会话落库（重写该分支消息）。async 包装，不阻塞事件循环"""
        await asyncio.to_thread(self._save_sync, session_id, messages, session_ctx, title, branch_id)

    def _save_sync(self, session_id, messages, session_ctx, title, branch_id) -> None:
        now = time.time()
        dicts = messages_to_dict(messages)
        user_id = (session_ctx or {}).get("user_id", "")
        tenant_id = (session_ctx or {}).get("tenant_id", "")
        title = title or ""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, user_id, tenant_id, title, summary, status, created_at, updated_at)
                   VALUES(?,?,?,?,'','active',?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     user_id=excluded.user_id,
                     tenant_id=excluded.tenant_id,
                     title=CASE WHEN excluded.title<>'' THEN excluded.title ELSE sessions.title END,
                     updated_at=excluded.updated_at""",
                (session_id, user_id, tenant_id, title, now, now),
            )
            # 整轮重写：清掉该分支旧消息，按序重插（seq 递增，turn_id 暂置 0）
            conn.execute("DELETE FROM messages WHERE session_id=? AND branch_id=?", (session_id, branch_id))
            conn.executemany(
                "INSERT INTO messages(session_id, branch_id, seq, turn_id, role, payload, archived) VALUES(?,?,?,?,?,?,0)",
                [
                    (session_id, branch_id, i, 0, str(m.type), json.dumps(d, ensure_ascii=False), 0)
                    for i, (m, d) in enumerate(zip(messages, dicts))
                ],
            )

    # —— 读取 ——
    async def load(self, session_id: str, branch_id: str = "root") -> list[BaseMessage] | None:
        """按分支读回会话消息；会话不存在返回 None，无消息返回 []"""
        return await asyncio.to_thread(self._load_sync, session_id, branch_id)

    def _load_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            exists = conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if exists is None:
                return None
            rows = conn.execute(
                "SELECT payload FROM messages WHERE session_id=? AND branch_id=? ORDER BY seq",
                (session_id, branch_id),
            ).fetchall()
        if not rows:
            return []
        return messages_from_dict([json.loads(r[0]) for r in rows])
