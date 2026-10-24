"""ContextStore：会话上下文持久化（订阅者模式，事件溯源）

- 会话不引用本 store：订阅 SESSION_STATE_UPDATED（当前视图快照）落库；SESSION_CLOSED 标记关闭；
  CONTEXT_COMPRESSED 立即落库摘要；MESSAGES_ARCHIVED 增量归档每轮新增真实消息（append-only）
- 双轨持久化：
  * message_archive（append-only）= 全量历史，每轮增量追加，压缩改写视图不影响，可 load_full_history 重建
  * messages 表（覆盖）= 当前视图（压缩后 = System+Human摘要+最近keep轮），load_branch 恢复时注入会话
- 入口用它加载视图历史（load_branch）注入会话，并实现分支操作回调；load_full_history 供审计/SaaS 检索
- 公开 save/load 仅供入口写库与测试；运行时持久化一律走订阅者（事件即真相）
- 分支存储采用【快照式】：子分支自包含完整视图历史；load_branch 统一读本分支当前视图
"""
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType


class ContextStore:
    """SQLite 会话存储：订阅状态快照/归档落库；load/分支/summary/list 供入口恢复与 SaaS 检索"""

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
                    role TEXT,
                    payload TEXT,
                    UNIQUE(session_id, branch_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, branch_id);
                CREATE TABLE IF NOT EXISTS message_archive(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    branch_id TEXT DEFAULT 'root',
                    seq INTEGER,          -- 本轮内局部序号（全量顺序按自增 id）
                    role TEXT,
                    payload TEXT,
                    created_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_archive_session ON message_archive(session_id, branch_id);
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

    # ===================== 订阅者 =====================
    def subscribe_to_bus(self, bus: EventBus) -> None:
        """注册为持久化订阅者：视图快照 + 会话关闭 + 压缩摘要 + 每轮增量归档"""
        bus.subscribe(EventType.SESSION_STATE_UPDATED, self._on_state_updated)
        bus.subscribe(EventType.SESSION_CLOSED, self._on_session_closed)
        bus.subscribe(EventType.CONTEXT_COMPRESSED, self._on_compressed)
        bus.subscribe(EventType.MESSAGES_ARCHIVED, self._on_archived)

    async def _on_state_updated(self, event: Event) -> None:
        """async handler：当前视图快照落库（不阻塞事件循环）"""
        payload = event.payload or {}
        dicts = payload.get("messages") or []
        if not dicts:
            return
        await asyncio.to_thread(
            self._save_snapshot_sync,
            event.session_id,
            dicts,
            payload.get("session_ctx"),
            payload.get("branch_id") or "root",
            payload.get("summary"),
        )

    def _on_session_closed(self, event: Event) -> None:
        """标记会话关闭（轻量，同步即可）"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("UPDATE sessions SET status='closed' WHERE session_id=?", (event.session_id,))

    def _on_compressed(self, event: Event) -> None:
        """压缩完成立即落库新摘要（不等下轮状态快照，手动提前压缩也能即时持久化）"""
        summary = (event.payload or {}).get("summary")
        if not summary:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",
                (summary, time.time(), event.session_id),
            )

    async def _on_archived(self, event: Event) -> None:
        """每轮增量归档（append-only）：本轮新增真实消息落 message_archive，全量历史可重建"""
        payload = event.payload or {}
        dicts = payload.get("messages") or []
        if not dicts:
            return
        await asyncio.to_thread(
            self._append_archive_sync,
            event.session_id,
            payload.get("branch_id") or "root",
            dicts,
        )

    # ===================== 保存/归档（公开，供入口写库与测试；运行时持久化走订阅者） =====================
    async def save(
        self,
        session_id: str,
        messages: list[BaseMessage],
        session_ctx: dict | None = None,
        branch_id: str = "root",
        summary: str | None = None,
    ) -> None:
        """兼容入口/测试：BaseMessage 列表写当前视图快照"""
        dicts = messages_to_dict(messages)
        await asyncio.to_thread(
            self._save_snapshot_sync, session_id, dicts, session_ctx, branch_id, summary
        )

    async def load(self, session_id: str, branch_id: str = "root") -> list[BaseMessage] | None:
        """别名：加载指定分支当前视图历史；会话不存在返回 None"""
        return await self.load_branch(session_id, branch_id)

    def _save_snapshot_sync(
        self,
        session_id: str,
        dicts: list[dict],
        session_ctx: dict | None,
        branch_id: str,
        summary: str | None,
    ) -> None:
        """视图快照落库：覆盖 messages 表（当前视图），不碰 archive（增量归档由 MESSAGES_ARCHIVED 负责）"""
        now = time.time()
        user_id = (session_ctx or {}).get("user_id", "")
        tenant_id = (session_ctx or {}).get("tenant_id", "")
        summary = summary or ""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, user_id, tenant_id, title, summary, status, created_at, updated_at)
                   VALUES(?,?,?,?,'active',?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     user_id=excluded.user_id, tenant_id=excluded.tenant_id,
                     summary=CASE WHEN excluded.summary<>'' THEN excluded.summary ELSE sessions.summary END,
                     updated_at=excluded.updated_at""",
                (session_id, user_id, tenant_id, summary, now, now),
            )
            if branch_id == "root":
                conn.execute(
                    """INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, created_at)
                       VALUES(?,?,NULL,NULL,?) ON CONFLICT(branch_id) DO NOTHING""",
                    (branch_id, session_id, now),
                )
            conn.execute(
                "DELETE FROM messages WHERE session_id=? AND branch_id=?", (session_id, branch_id)
            )
            conn.executemany(
                "INSERT INTO messages(session_id, branch_id, seq, role, payload) VALUES(?,?,?,?,?)",
                [
                    (session_id, branch_id, i, d.get("type", ""), json.dumps(d, ensure_ascii=False))
                    for i, d in enumerate(dicts)
                ],
            )

    def _append_archive_sync(self, session_id: str, branch_id: str, dicts: list[dict]) -> None:
        """追加归档：append-only，仅写不删；压缩改写视图不影响全量历史"""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT INTO message_archive(session_id, branch_id, seq, role, payload, created_at) VALUES(?,?,?,?,?,?)",
                [
                    (session_id, branch_id, i, d.get("type", ""), json.dumps(d, ensure_ascii=False), now)
                    for i, d in enumerate(dicts)
                ],
            )

    # ===================== 读取（入口恢复用） =====================
    async def load_branch(self, session_id: str, branch_id: str = "root") -> list[BaseMessage] | None:
        """读当前视图（压缩后=System+Human摘要+最近keep轮），注入会话；会话不存在返回 None"""
        return await asyncio.to_thread(self._load_branch_sync, session_id, branch_id)

    def _load_branch_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            exists = conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if exists is None:
                return None
            if branch_id != "root":
                branch = conn.execute(
                    "SELECT 1 FROM branches WHERE branch_id=? AND session_id=?", (branch_id, session_id)
                ).fetchone()
                if branch is None:
                    raise ValueError(f"分支 '{branch_id}' 不存在或不属于会话 '{session_id}'")
            rows = conn.execute(
                "SELECT payload FROM messages WHERE session_id=? AND branch_id=? ORDER BY seq",
                (session_id, branch_id),
            ).fetchall()
        if not rows:
            return []
        return messages_from_dict([json.loads(r[0]) for r in rows])

    async def load_full_history(self, session_id: str, branch_id: str = "root") -> list[BaseMessage]:
        """从归档表重建全量历史（append-only，含被压缩移出的早期轮）；无则返回空列表"""
        return await asyncio.to_thread(self._load_full_history_sync, session_id, branch_id)

    def _load_full_history_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT payload FROM message_archive WHERE session_id=? AND branch_id=? ORDER BY id",
                (session_id, branch_id),
            ).fetchall()
        if not rows:
            return []
        return messages_from_dict([json.loads(r[0]) for r in rows])

    # ===================== 分支（入口实现会话分支回调） =====================
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

    # ===================== 摘要 =====================
    async def load_summary(self, session_id: str) -> str | None:
        """读会话级压缩摘要；无会话或摘要为空返回 None"""
        return await asyncio.to_thread(self._load_summary_sync, session_id)

    def _load_summary_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT summary FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None or not row[0]:
            return None
        return row[0]

    # ===================== 列出（SaaS 检索预留） =====================
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
