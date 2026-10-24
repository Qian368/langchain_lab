"""ContextStore：会话上下文持久化（订阅者模式，单一真相源 = message_archive）

设计（取消覆盖式快照，archive 为唯一真相源）：
- 【分支场景驱动】覆盖式快照只留最新视图，无法从历史轮次分叉（如会话 20 轮、从第 10 轮分支）。
  改为 archive 存全量无损真实消息，恢复/分支都从它构建 llm 视图（build_llm_context），支持任意拐点。
- 【KV 缓存一致】archive 用 messages_to_dict 无损序列化（含 tool_calls 等，不简化）；恢复构建的视图
  与压缩后喂给 LLM 的 _messages 逐 token 一致（同 system+摘要+最近 keep 轮真实消息，纯函数 build_llm_context）。
- 订阅：MESSAGES_ARCHIVED（增量归档，带 event_id 幂等去重）/ SESSION_CLOSED（标关闭）/ CONTEXT_COMPRESSED（存摘要）。
- 不再订阅 SESSION_STATE_UPDATED（无覆盖快照落库；该事件仍发布，供前端/日志订阅）。
- 恢复：入口调 ensure_session（存 system_prompt）+ load_view（从 archive+summary+system 构建视图）。
- 分支：create_branch 返回 (new_id, 拐点视图)；拐点视图 = System + 前 point 轮真实消息（不用当前摘要）。
- 读取：load_view（llm 视图恢复）/ load_full_history（全量真实消息，审计/前端早期展开）/ load_summary / list_sessions。
"""
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage, messages_from_dict

from langchain_lab.core.context.views import build_llm_context
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType

_DEFAULT_SYSTEM_PROMPT = "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务"


class ContextStore:
    """SQLite 会话存储（单一真相源）：archive 增量归档 + system_prompt/summary 元数据；恢复/分支从 archive 构建视图"""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """开发期 schema 演进：旧库需重建（删 messages 覆盖快照表）。"""
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT, tenant_id TEXT,
                    title TEXT DEFAULT '',
                    system_prompt TEXT DEFAULT '',  -- 原始 system prompt（恢复构建视图用，保证 KV 前缀一致）
                    summary TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    created_at REAL, updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS message_archive(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    branch_id TEXT DEFAULT 'root',
                    event_id TEXT DEFAULT '',   -- 源头事件幂等键（UUID）
                    seq INTEGER,                -- 事件内局部序号（全量顺序按自增 id）
                    role TEXT,
                    payload TEXT,
                    created_at REAL,
                    UNIQUE(session_id, branch_id, event_id, seq)  -- 幂等：重放/重投递时 INSERT OR IGNORE 天然去重
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
        """注册为持久化订阅者：增量归档（幂等）+ 会话关闭 + 压缩摘要；不订阅视图快照（无覆盖存储）"""
        bus.subscribe(EventType.MESSAGES_ARCHIVED, self._on_archived)
        bus.subscribe(EventType.SESSION_CLOSED, self._on_session_closed)
        bus.subscribe(EventType.CONTEXT_COMPRESSED, self._on_compressed)

    async def _on_archived(self, event: Event) -> None:
        """每轮增量归档（append-only）：本轮新增真实消息落 archive；event_id 幂等去重"""
        payload = event.payload or {}
        dicts = payload.get("messages") or []
        if not dicts:
            return
        await asyncio.to_thread(
            self._append_archive_sync,
            event.session_id,
            payload.get("branch_id") or "root",
            event.event_id,
            dicts,
        )

    def _on_session_closed(self, event: Event) -> None:
        """标记会话关闭（轻量，同步即可）"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("UPDATE sessions SET status='closed' WHERE session_id=?", (event.session_id,))

    def _on_compressed(self, event: Event) -> None:
        """压缩完成立即落库新摘要（不等状态事件，手动提前压缩也能即时持久化）"""
        summary = (event.payload or {}).get("summary")
        if not summary:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",
                (summary, time.time(), event.session_id),
            )

    # ===================== 注册 / 归档（公开，供入口与测试） =====================
    async def ensure_session(
        self, session_id: str, session_ctx: dict | None = None, system_prompt: str | None = None
    ) -> None:
        """注册/更新会话行：存 user/tenant + 原始 system_prompt（恢复构建视图用，KV 前缀一致）。入口创建/恢复时调用"""
        await asyncio.to_thread(self._ensure_session_sync, session_id, session_ctx, system_prompt)

    def _ensure_session_sync(self, session_id, session_ctx, system_prompt):
        now = time.time()
        user_id = (session_ctx or {}).get("user_id", "")
        tenant_id = (session_ctx or {}).get("tenant_id", "")
        system_prompt = system_prompt or ""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, user_id, tenant_id, title, system_prompt, summary, status, created_at, updated_at)
                   VALUES(?,?,?,?,'',?,'active',?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     user_id=excluded.user_id, tenant_id=excluded.tenant_id,
                     system_prompt=CASE WHEN excluded.system_prompt<>'' THEN excluded.system_prompt ELSE sessions.system_prompt END,
                     updated_at=excluded.updated_at""",
                (session_id, user_id, tenant_id, system_prompt, now, now),
            )
            conn.execute(
                """INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, created_at)
                   VALUES(?,?,NULL,NULL,?) ON CONFLICT(branch_id) DO NOTHING""",
                ("root", session_id, now),
            )

    def _append_archive_sync(self, session_id: str, branch_id: str, event_id: str, dicts: list[dict]) -> None:
        """追加归档：append-only，仅写不删；INSERT OR IGNORE 保证事件幂等（同 event_id 重放跳过）"""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO message_archive(session_id, branch_id, event_id, seq, role, payload, created_at) VALUES(?,?,?,?,?,?,?)",
                [
                    (session_id, branch_id, event_id, i, d.get("type", ""), json.dumps(d, ensure_ascii=False), now)
                    for i, d in enumerate(dicts)
                ],
            )

    # ===================== 读取（恢复 / 审计） =====================
    async def load_view(
        self, session_id: str, branch_id: str = "root", keep_recent_turns: int = 6
    ) -> list[BaseMessage] | None:
        """恢复视图：从 archive(全量真实消息)+summary+system_prompt 构建 llm 视图（复用 build_llm_context，KV 逐 token 一致）

        有摘要 → System + Human(摘要) + 最近 keep 轮；无摘要 → System + 全量真实消息。会话不存在返回 None。
        """
        return await asyncio.to_thread(self._load_view_sync, session_id, branch_id, keep_recent_turns)

    def _load_view_sync(self, session_id, branch_id, keep):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT system_prompt, summary FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            system_prompt, summary = row
            if branch_id != "root":
                branch = conn.execute(
                    "SELECT 1 FROM branches WHERE branch_id=? AND session_id=?", (branch_id, session_id)
                ).fetchone()
                if branch is None:
                    raise ValueError(f"分支 '{branch_id}' 不存在或不属于会话 '{session_id}'")
            rows = conn.execute(
                "SELECT payload FROM message_archive WHERE session_id=? AND branch_id=? ORDER BY id",
                (session_id, branch_id),
            ).fetchall()
        real = messages_from_dict([json.loads(r[0]) for r in rows])
        system = SystemMessage(content=system_prompt or _DEFAULT_SYSTEM_PROMPT)
        base = [system] + real
        if not summary:
            return base  # 未压缩：全量真实消息 + system，即原来喂给 LLM 的视图
        return build_llm_context(base, summary, keep)  # 已压缩：System + Human摘要 + 最近 keep 轮

    async def load_full_history(self, session_id: str, branch_id: str = "root") -> list[BaseMessage]:
        """从 archive 重建全量真实消息（append-only，含被压缩移出的早期轮）；无则返回空列表"""
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

    # ===================== 分支 =====================
    async def create_branch(
        self, session_id: str, parent_id: str = "root", branch_point_seq: int | None = None,
        keep_recent_turns: int = 6,
    ) -> tuple[str, list[BaseMessage]]:
        """在父分支拐点处创建子分支，返回 (new_branch_id, 拐点视图)。

        拐点视图 = System + 父分支 archive 前 point 条真实消息（不用当前摘要，因为拐点可能早于/覆盖摘要，
        应是拐点处干净历史切片）；支持压缩后从历史轮次分叉（archive 已无损保留早期轮）。
        """
        return await asyncio.to_thread(
            self._create_branch_sync, session_id, parent_id, branch_point_seq, keep_recent_turns
        )

    def _create_branch_sync(self, session_id, parent_id, point, keep):
        if not isinstance(point, int) or point < 1:
            raise ValueError("分支拐点 point 需为正整数（对话消息序号）")
        new_branch_id = str(uuid.uuid4())
        with sqlite3.connect(self._db_path) as conn:
            parent = conn.execute(
                "SELECT 1 FROM branches WHERE branch_id=? AND session_id=?", (parent_id, session_id)
            ).fetchone()
            if parent is None:
                raise ValueError(f"父分支 '{parent_id}' 不存在或不属于会话 '{session_id}'")
            sp = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            rows = conn.execute(
                "SELECT payload FROM message_archive WHERE session_id=? AND branch_id=? ORDER BY id LIMIT ?",
                (session_id, parent_id, point),
            ).fetchall()
            if len(rows) < point:
                raise ValueError(f"拐点 {point} 超出当前分支已归档消息数（{len(rows)}）")
            conn.execute(
                "INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, created_at) VALUES(?,?,?,?,?)",
                (new_branch_id, session_id, parent_id, point, time.time()),
            )
        system_prompt = sp[0] if sp else ""
        system = SystemMessage(content=system_prompt or _DEFAULT_SYSTEM_PROMPT)
        point_view = [system] + messages_from_dict([json.loads(r[0]) for r in rows])
        return new_branch_id, point_view

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

    # ===================== 摘要 / system =====================
    async def load_summary(self, session_id: str) -> str | None:
        """读会话级压缩摘要；无会话或摘要为空返回 None"""
        return await asyncio.to_thread(self._load_summary_sync, session_id)

    def _load_summary_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT summary FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None or not row[0]:
            return None
        return row[0]

    async def load_system_prompt(self, session_id: str) -> str | None:
        """读会话原始 system prompt（恢复构建视图用）"""
        return await asyncio.to_thread(self._load_system_prompt_sync, session_id)

    def _load_system_prompt_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT system_prompt FROM sessions WHERE session_id=?", (session_id,)).fetchone()
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
