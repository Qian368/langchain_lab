"""SessionStore：SQLite 持久化（schema 面向 SaaS 检索 / 压缩归档 / 会话分支 预留）

- 序列化用 LangChain messages_to_dict / messages_from_dict，tool_call_id 无损往返
- sqlite3 同步库，读写用 asyncio.to_thread 包装，不阻塞事件循环
- schema：
  - sessions.summary（LLM 压缩摘要，各分支共享）/ user_id / tenant_id（SaaS 检索）
  - messages.branch_id + seq（分支内独立编号）
  - branches（branch_id / parent_id / branch_point_seq）：分支树，拐点定位
"""
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class SessionStore:
    """SQLite 会话存储：save 整轮/整会话，分支创建/加载，list 按租户列出"""

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
                CREATE TABLE IF NOT EXISTS branches(
                    branch_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    parent_id TEXT,           -- 根为 NULL
                    branch_point_seq INTEGER, -- 拐点在父分支的 seq（根为 NULL）
                    created_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_branches_session ON branches(session_id);
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
        summary: str | None = None,
    ) -> None:
        """整轮/整会话落库（重写该分支消息）。async 包装，不阻塞事件循环"""
        await asyncio.to_thread(
            self._save_sync, session_id, messages, session_ctx, title, branch_id, summary
        )

    def _save_sync(self, session_id, messages, session_ctx, title, branch_id, summary) -> None:
        now = time.time()
        dicts = messages_to_dict(messages)
        user_id = (session_ctx or {}).get("user_id", "")
        tenant_id = (session_ctx or {}).get("tenant_id", "")
        title = title or ""
        summary = summary or ""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, user_id, tenant_id, title, summary, status, created_at, updated_at)
                   VALUES(?,?,?,?,?,'active',?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     user_id=excluded.user_id,
                     tenant_id=excluded.tenant_id,
                     title=CASE WHEN excluded.title<>'' THEN excluded.title ELSE sessions.title END,
                     summary=CASE WHEN excluded.summary<>'' THEN excluded.summary ELSE sessions.summary END,
                     updated_at=excluded.updated_at""",
                (session_id, user_id, tenant_id, title, summary, now, now),
            )
            # 确保 root 分支行存在（子分支由 create_branch 创建）
            if branch_id == "root":
                conn.execute(
                    """INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, created_at)
                       VALUES(?,?,NULL,NULL,?) ON CONFLICT(branch_id) DO NOTHING""",
                    (branch_id, session_id, now),
                )
            # 整轮重写：清掉该分支旧消息，按序重插（seq 递增，turn_id 暂置 0）
            conn.execute(
                "DELETE FROM messages WHERE session_id=? AND branch_id=?", (session_id, branch_id)
            )
            conn.executemany(
                "INSERT INTO messages(session_id, branch_id, seq, turn_id, role, payload, archived) VALUES(?,?,?,?,?,?,0)",
                [
                    (session_id, branch_id, i, 0, str(m.type), json.dumps(d, ensure_ascii=False), 0)
                    for i, (m, d) in enumerate(zip(messages, dicts))
                ],
            )

    # —— 读取 ——
    async def load(self, session_id: str, branch_id: str = "root") -> list[BaseMessage] | None:
        """别名：加载指定分支历史（向后兼容）；会话不存在返回 None"""
        return await self.load_branch(session_id, branch_id)

    async def load_branch(self, session_id: str, branch_id: str = "root") -> list[BaseMessage] | None:
        """按分支取完整历史：子分支 = 父分支拐点前(seq<=point) + 本分支消息；root = 全量"""
        return await asyncio.to_thread(self._load_branch_sync, session_id, branch_id)

    def _load_branch_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if exists is None:
                return None
            if branch_id != "root":
                branch = conn.execute(
                    "SELECT parent_id, branch_point_seq FROM branches WHERE branch_id=? AND session_id=?",
                    (branch_id, session_id),
                ).fetchone()
                if branch is None:
                    raise ValueError(f"分支 '{branch_id}' 不存在或不属于会话 '{session_id}'")
                parent_id, point = branch
                rows = []
                if parent_id:
                    if point is not None:
                        rows += conn.execute(
                            "SELECT payload FROM messages WHERE session_id=? AND branch_id=? AND seq<=? ORDER BY seq",
                            (session_id, parent_id, point),
                        ).fetchall()
                    else:
                        rows += conn.execute(
                            "SELECT payload FROM messages WHERE session_id=? AND branch_id=? ORDER BY seq",
                            (session_id, parent_id),
                        ).fetchall()
                rows += conn.execute(
                    "SELECT payload FROM messages WHERE session_id=? AND branch_id=? ORDER BY seq",
                    (session_id, branch_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM messages WHERE session_id=? AND branch_id='root' ORDER BY seq",
                    (session_id,),
                ).fetchall()
        if not rows:
            return []
        return messages_from_dict([json.loads(r[0]) for r in rows])

    # —— 分支 ——
    async def create_branch(
        self, session_id: str, parent_id: str = "root", branch_point_seq: int | None = None
    ) -> str:
        """在父分支拐点处创建子分支，返回新 branch_id"""
        return await asyncio.to_thread(self._create_branch_sync, session_id, parent_id, branch_point_seq)

    def _create_branch_sync(self, session_id, parent_id, branch_point_seq):
        branch_id = str(uuid.uuid4())
        with sqlite3.connect(self._db_path) as conn:
            parent = conn.execute(
                "SELECT 1 FROM branches WHERE branch_id=? AND session_id=?", (parent_id, session_id)
            ).fetchone()
            if parent is None:
                raise ValueError(f"父分支 '{parent_id}' 不存在或不属于会话 '{session_id}'")
            conn.execute(
                "INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, created_at) VALUES(?,?,?,?,?)",
                (branch_id, session_id, parent_id, branch_point_seq, time.time()),
            )
        return branch_id

    async def list_branches(self, session_id: str) -> list[dict]:
        """列出会话全部分支（id/parent/拐点）"""
        return await asyncio.to_thread(self._list_branches_sync, session_id)

    def _list_branches_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT branch_id, parent_id, branch_point_seq, created_at FROM branches WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [
            {"branch_id": r[0], "parent_id": r[1], "branch_point_seq": r[2], "created_at": r[3]}
            for r in rows
        ]

    # —— 压缩摘要 ——
    async def load_summary(self, session_id: str) -> str | None:
        """读会话级压缩摘要；无会话或摘要为空返回 None"""
        return await asyncio.to_thread(self._load_summary_sync, session_id)

    def _load_summary_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None or not row[0]:
            return None
        return row[0]

    # —— 列出（SaaS 检索预留）——
    async def list_sessions(self, user_id: str, tenant_id: str) -> list[dict]:
        """按 user/tenant 列出会话元数据（id/title/status/时间戳），按最近更新倒序"""
        return await asyncio.to_thread(self._list_sessions_sync, user_id, tenant_id)

    def _list_sessions_sync(self, user_id, tenant_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, title, status, created_at, updated_at
                   FROM sessions WHERE user_id=? AND tenant_id=?
                   ORDER BY updated_at DESC""",
                (user_id, tenant_id),
            ).fetchall()
        return [
            {"session_id": r[0], "title": r[1], "status": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]
