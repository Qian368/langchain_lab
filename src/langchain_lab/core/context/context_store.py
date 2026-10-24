"""ContextStore：会话上下文持久化（订阅者模式，单一真相源 = message_archive + 分支树）

设计（树形会话，最终形态）：
- 【消息唯一】message_archive 一行一条真实消息（append-only），子分支【不复制】继承消息；
  压缩摘要作为 kind='summary' 消息也进归档（append，历史摘要保留），恢复取最近一条。
- 【分支树】branches 表 parent_id 链 = 会话树（root→b1→b1.1…），branch_point_seq 记录在父分支哪条消息分叉。
  任意位置分叉/恢复 = 沿链回溯 branch_history（根全量 + 各级增量，按拐点截断），支持继承历史分叉。
- 【KV 一致】archive 用 messages_to_dict 无损序列化；恢复视图 = 回溯完整消息 + 最近摘要 → build_llm_context
  （同压缩时函数与输入，逐 token 一致，KV 前缀命中）。
- 【两级删除】软删（级联子树 + 标记 archive，不物理删、不冲突）+ gc_purge（物理删不可达，彻底释放）。
- 【幂等】事件带 event_id，archive UNIQUE(session_id,branch_id,event_id,seq) + INSERT OR IGNORE。
- 订阅：MESSAGES_ARCHIVED（增量归档）/ SESSION_CLOSED（标关闭）/ CONTEXT_COMPRESSED（更新会话级 summary 文本）/
  SUMMARY_ARCHIVED（摘要消息按分支落 archive，由 ChatSession 携带当前 branch_id 发出）。
- 命名：sessions.name / branches.name（展示用，默认值、非唯一）。
"""
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict

from langchain_lab.core.context.views import build_llm_context
from langchain_lab.core.event_bus import EventBus
from langchain_lab.core.events import Event, EventType

_DEFAULT_SYSTEM_PROMPT = "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务"
_DEFAULT_SESSION_NAME = "新会话"


class ContextStore:
    """SQLite 会话存储：archive 消息唯一 + 分支树；回溯读取；两级删除；幂等去重；命名"""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """开发期 schema 演进：旧库需重建。"""
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT, tenant_id TEXT,
                    title TEXT DEFAULT '',
                    name TEXT DEFAULT '',          -- 会话名（展示，默认值非唯一）
                    system_prompt TEXT DEFAULT '', -- 原始 system（恢复构建视图，KV 一致）
                    summary TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',  -- active / deleted（软删）
                    created_at REAL, updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS message_archive(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    branch_id TEXT DEFAULT 'root',
                    event_id TEXT DEFAULT '',   -- 源头事件幂等键（UUID）
                    seq INTEGER,                -- 事件内局部序号
                    role TEXT,
                    kind TEXT DEFAULT 'message', -- message / summary（摘要也进归档）
                    payload TEXT,
                    deleted INTEGER DEFAULT 0,   -- 软删标记（1=已删除，gc 物理清理）
                    created_at REAL,
                    UNIQUE(session_id, branch_id, event_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_archive_session ON message_archive(session_id, branch_id);
                CREATE TABLE IF NOT EXISTS branches(
                    branch_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    parent_id TEXT,           -- 根为 NULL
                    branch_point_seq INTEGER, -- 拐点（在父分支哪条消息分叉；根为 NULL）
                    name TEXT DEFAULT '',     -- 分支名（展示，默认值非唯一）
                    status TEXT DEFAULT 'active', -- active / deleted（软删）
                    created_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_branches_session ON branches(session_id);
                """
            )

    # ===================== 订阅者 =====================
    def subscribe_to_bus(self, bus: EventBus) -> None:
        """注册持久化订阅者：增量归档 + 会话关闭 + 会话级 summary 更新 + 摘要消息按分支归档"""
        bus.subscribe(EventType.MESSAGES_ARCHIVED, self._on_archived)
        bus.subscribe(EventType.SESSION_CLOSED, self._on_session_closed)
        bus.subscribe(EventType.CONTEXT_COMPRESSED, self._on_compressed)
        bus.subscribe(EventType.SUMMARY_ARCHIVED, self._on_summary_archived)

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

    async def _on_compressed(self, event: Event) -> None:
        """压缩完成：更新会话级 summary 文本（各分支共享；摘要消息归档另走 SUMMARY_ARCHIVED）"""
        summary = (event.payload or {}).get("summary")
        if not summary:
            return
        await asyncio.to_thread(self._update_summary_sync, event.session_id, summary)

    async def _on_summary_archived(self, event: Event) -> None:
        """摘要消息按当前分支落 archive（kind=summary，历史摘要保留；event_id 幂等）。

        由 ChatSession 压缩后携带当前 branch_id 发出，确保非 root 分支压缩摘要归属正确。
        """
        summary = (event.payload or {}).get("summary")
        if not summary:
            return
        await asyncio.to_thread(
            self._append_summary_sync,
            event.session_id,
            (event.payload or {}).get("branch_id") or "root",
            event.event_id,
            summary,
        )

    # ===================== 注册 / 归档 =====================
    async def ensure_session(
        self, session_id: str, session_ctx: dict | None = None,
        system_prompt: str | None = None, name: str | None = None,
    ) -> None:
        """注册/更新会话行：存 user/tenant + system_prompt + name（恢复/新建用）。恢复传 system_prompt=None 保留已有"""
        await asyncio.to_thread(self._ensure_session_sync, session_id, session_ctx, system_prompt, name)

    def _ensure_session_sync(self, session_id, session_ctx, system_prompt, name):
        now = time.time()
        user_id = (session_ctx or {}).get("user_id", "")
        tenant_id = (session_ctx or {}).get("tenant_id", "")
        system_prompt = system_prompt or ""
        name = name or _DEFAULT_SESSION_NAME
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, user_id, tenant_id, title, name, system_prompt, summary, status, created_at, updated_at)
                   VALUES(?,?,?,?,'','','active',?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     user_id=excluded.user_id, tenant_id=excluded.tenant_id,
                     name=CASE WHEN excluded.name<>'' AND excluded.name<>'%s' THEN excluded.name ELSE sessions.name END,
                     system_prompt=CASE WHEN excluded.system_prompt<>'' THEN excluded.system_prompt ELSE sessions.system_prompt END,
                     status=CASE WHEN sessions.status='deleted' THEN 'active' ELSE sessions.status END,
                     updated_at=excluded.updated_at"""
                % _DEFAULT_SESSION_NAME,
                (session_id, user_id, tenant_id, name, system_prompt, now, now),
            )
            conn.execute(
                """INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, name, status, created_at)
                   VALUES(?,?,NULL,NULL,?,'active',?) ON CONFLICT(branch_id) DO NOTHING""",
                ("root", session_id, _DEFAULT_SESSION_NAME, now),
            )

    def _append_archive_sync(self, session_id, branch_id, event_id, dicts) -> None:
        """追加归档真实消息：append-only，INSERT OR IGNORE 幂等"""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO message_archive(session_id, branch_id, event_id, seq, role, kind, payload, deleted, created_at) VALUES(?,?,?,?,?,'message',?,0,?)",
                [
                    (session_id, branch_id, event_id, i, d.get("type", ""), json.dumps(d, ensure_ascii=False), now)
                    for i, d in enumerate(dicts)
                ],
            )

    def _append_summary_sync(self, session_id, branch_id, event_id, summary) -> None:
        """摘要消息（kind=summary）append 进指定分支 archive；同一 event_id 幂等"""
        msg_dict = messages_to_dict(
            HumanMessage(content=f"[对话历史摘要]\n{summary}", additional_kwargs={"meta": "summary"})
        )
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO message_archive(session_id, branch_id, event_id, seq, role, kind, payload, deleted, created_at) VALUES(?,?,'summary',0,'human','summary',?,0,?)",
                (session_id, branch_id, event_id, json.dumps(msg_dict, ensure_ascii=False), time.time()),
            )

    def _update_summary_sync(self, session_id, summary) -> None:
        """更新会话级 summary 文本（各分支共享）"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",
                (summary, time.time(), session_id),
            )

    # ===================== 分支树读取（回溯） =====================
    def _collect_branch_rows(self, conn, session_id, branch_id) -> list[tuple[str, str]]:
        """沿 parent 链回溯，返回该分支完整历史的 (kind, payload_json) 序列（含继承，不含 system）"""
        chain: list[tuple[str, int | None]] = []
        cur = branch_id
        while True:
            row = conn.execute(
                "SELECT parent_id, branch_point_seq FROM branches WHERE branch_id=? AND session_id=? AND status='active'",
                (cur, session_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"分支 '{cur}' 不存在/已删除/不属于会话 '{session_id}'")
            parent_id, point = row
            if parent_id is None:
                chain.append((cur, None))
                break
            chain.append((cur, point))
            cur = parent_id
        chain.reverse()  # 根→该分支
        rows: list[tuple[str, str]] = []
        for i, (bid, _point) in enumerate(chain):
            is_last = i == len(chain) - 1
            if is_last:
                part = conn.execute(
                    "SELECT kind, payload FROM message_archive WHERE session_id=? AND branch_id=? AND deleted=0 ORDER BY id",
                    (session_id, bid),
                ).fetchall()
            else:
                next_point = chain[i + 1][1]  # 该分支被下一级分叉的拐点
                part = conn.execute(
                    "SELECT kind, payload FROM message_archive WHERE session_id=? AND branch_id=? AND deleted=0 ORDER BY id LIMIT ?",
                    (session_id, bid, next_point),
                ).fetchall()
            rows += part
        return rows

    async def branch_history(self, session_id: str, branch_id: str = "root") -> list[BaseMessage]:
        """返回分支完整历史消息（含继承 + 摘要消息，不含 system）"""
        return await asyncio.to_thread(self._branch_history_sync, session_id, branch_id)

    def _branch_history_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = self._collect_branch_rows(conn, session_id, branch_id)
        return messages_from_dict([json.loads(p) for (_k, p) in rows])

    # ===================== 读取（恢复视图 / 全量） =====================
    async def load_view(
        self, session_id: str, branch_id: str = "root", keep_recent_turns: int = 6
    ) -> list[BaseMessage] | None:
        """恢复视图：回溯完整历史 + 最近 kind=summary → build_llm_context（System + 摘要 + 最近 keep 轮）"""
        return await asyncio.to_thread(self._load_view_sync, session_id, branch_id, keep_recent_turns)

    def _load_view_sync(self, session_id, branch_id, keep):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id=? AND status!='deleted'", (session_id,)
            ).fetchone()
            if row is None:
                return None
            system_prompt = row[0]
            rows = self._collect_branch_rows(conn, session_id, branch_id)
        summary = None
        real_dicts: list[dict] = []
        for kind, payload in rows:
            d = json.loads(payload)
            if kind == "summary":
                summary = d.get("data", {}).get("content") or d.get("content")  # 最近一条覆盖
            else:
                real_dicts.append(d)
        system = SystemMessage(content=system_prompt or _DEFAULT_SYSTEM_PROMPT)
        base = [system] + messages_from_dict(real_dicts)
        if not summary:
            return base
        return build_llm_context(base, summary, keep)

    async def load_full_history(self, session_id: str, branch_id: str = "root") -> list[BaseMessage]:
        """从归档表重建某分支全量真实消息（含继承/摘要，不含 system）；无则空列表"""
        return await asyncio.to_thread(self._load_full_history_sync, session_id, branch_id)

    def _load_full_history_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = self._collect_branch_rows(conn, session_id, branch_id)
        return messages_from_dict([json.loads(p) for (_k, p) in rows])

    # ===================== 分支（任意位置分叉，回溯拐点视图） =====================
    async def create_branch(
        self, session_id: str, parent_id: str = "root", branch_point_seq: int | None = None,
        keep_recent_turns: int = 6, name: str | None = None,
    ) -> tuple[str, list[BaseMessage]]:
        """在父分支任意位置分叉（含继承历史），返回 (new_id, 拐点视图)。拐点视图 = System + 父分支完整历史前 point 条"""
        return await asyncio.to_thread(
            self._create_branch_sync, session_id, parent_id, branch_point_seq, keep_recent_turns, name
        )

    def _create_branch_sync(self, session_id, parent_id, point, keep, name):
        if not isinstance(point, int) or point < 1:
            raise ValueError("分支拐点 point 需为正整数（父分支历史消息序号）")
        new_branch_id = str(uuid.uuid4())
        with sqlite3.connect(self._db_path) as conn:
            parent = conn.execute(
                "SELECT 1 FROM branches WHERE branch_id=? AND session_id=? AND status='active'",
                (parent_id, session_id),
            ).fetchone()
            if parent is None:
                raise ValueError(f"父分支 '{parent_id}' 不存在/已删除/不属于会话 '{session_id}'")
            rows = self._collect_branch_rows(conn, session_id, parent_id)
            if len(rows) < point:
                raise ValueError(f"拐点 {point} 超出父分支历史消息数（{len(rows)}）")
            sp = conn.execute("SELECT system_prompt FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            now = time.time()
            n = conn.execute(
                "SELECT COUNT(*) FROM branches WHERE session_id=? AND parent_id=? AND status='active'",
                (session_id, parent_id),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO branches(branch_id, session_id, parent_id, branch_point_seq, name, status, created_at) VALUES(?,?,?,?,?,'active',?)",
                (new_branch_id, session_id, parent_id, point, name or f"分支{n + 1}", now),
            )
        system_prompt = sp[0] if sp else ""
        system = SystemMessage(content=system_prompt or _DEFAULT_SYSTEM_PROMPT)
        point_view = [system] + messages_from_dict([json.loads(p) for (_k, p) in rows[:point]])
        return new_branch_id, point_view

    async def list_branches(self, session_id: str) -> list[dict]:
        """列出会话 active 分支（id/parent/拐点/name）"""
        return await asyncio.to_thread(self._list_branches_sync, session_id)

    def _list_branches_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT branch_id, parent_id, branch_point_seq, name, created_at FROM branches WHERE session_id=? AND status='active' ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [
            {"branch_id": r[0], "parent_id": r[1], "branch_point_seq": r[2], "name": r[3], "created_at": r[4]}
            for r in rows
        ]

    # ===================== 两级删除 =====================
    def _collect_descendants(self, conn, session_id, branch_id) -> list[str]:
        """收集分支及其所有 active 后代（含自身）"""
        result = []
        stack = [branch_id]
        while stack:
            bid = stack.pop()
            result.append(bid)
            for (child,) in conn.execute(
                "SELECT branch_id FROM branches WHERE session_id=? AND parent_id=? AND status='active'",
                (session_id, bid),
            ).fetchall():
                stack.append(child)
        return result

    async def delete_branch(self, session_id: str, branch_id: str) -> list[str]:
        """软删分支子树（级联后代 + 标记 archive）。返回被删的分支 id 列表"""
        return await asyncio.to_thread(self._delete_branch_sync, session_id, branch_id)

    def _delete_branch_sync(self, session_id, branch_id):
        with sqlite3.connect(self._db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM branches WHERE branch_id=? AND session_id=? AND status='active'",
                (branch_id, session_id),
            ).fetchone()
            if exists is None:
                raise ValueError(f"分支 '{branch_id}' 不存在/已删除/不属于会话 '{session_id}'")
            ids = self._collect_descendants(conn, session_id, branch_id)
            ph = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE branches SET status='deleted' WHERE session_id=? AND branch_id IN ({ph})",
                [session_id, *ids],
            )
            conn.execute(
                f"UPDATE message_archive SET deleted=1 WHERE session_id=? AND branch_id IN ({ph})",
                [session_id, *ids],
            )
        return ids

    async def delete_session(self, session_id: str) -> None:
        """软删整个会话（session + 全部分支 + 标记 archive）"""
        await asyncio.to_thread(self._delete_session_sync, session_id)

    def _delete_session_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("UPDATE sessions SET status='deleted' WHERE session_id=?", (session_id,))
            conn.execute("UPDATE branches SET status='deleted' WHERE session_id=?", (session_id,))
            conn.execute("UPDATE message_archive SET deleted=1 WHERE session_id=?", (session_id,))

    async def gc_purge(self) -> dict:
        """物理清理：删软删标记的 archive/branches/sessions，真正释放空间。返回清理计数"""
        return await asyncio.to_thread(self._gc_purge_sync)

    def _gc_purge_sync(self):
        with sqlite3.connect(self._db_path) as conn:
            a = conn.execute("DELETE FROM message_archive WHERE deleted=1").rowcount
            b = conn.execute("DELETE FROM branches WHERE status='deleted'").rowcount
            s = conn.execute("DELETE FROM sessions WHERE status='deleted'").rowcount
        return {"archive": a, "branches": b, "sessions": s}

    # ===================== 命名 =====================
    async def rename_session(self, session_id: str, name: str) -> None:
        """重命名会话（需非空）"""
        await asyncio.to_thread(self._rename_session_sync, session_id, name)

    def _rename_session_sync(self, session_id, name):
        if not name or not name.strip():
            raise ValueError("会话名不能为空")
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("UPDATE sessions SET name=?, updated_at=? WHERE session_id=?", (name.strip(), time.time(), session_id))

    async def rename_branch(self, session_id: str, branch_id: str, name: str) -> None:
        """重命名分支（需非空）"""
        await asyncio.to_thread(self._rename_branch_sync, session_id, branch_id, name)

    def _rename_branch_sync(self, session_id, branch_id, name):
        if not name or not name.strip():
            raise ValueError("分支名不能为空")
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE branches SET name=? WHERE branch_id=? AND session_id=? AND status='active'",
                (name.strip(), branch_id, session_id),
            )

    # ===================== 摘要 / system / 列出 =====================
    async def load_summary(self, session_id: str) -> str | None:
        """读会话级最近压缩摘要；无则 None"""
        return await asyncio.to_thread(self._load_summary_sync, session_id)

    def _load_summary_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id=? AND status!='deleted'", (session_id,)
            ).fetchone()
        if row is None or not row[0]:
            return None
        return row[0]

    async def load_system_prompt(self, session_id: str) -> str | None:
        """读会话原始 system prompt（恢复构建视图用）"""
        return await asyncio.to_thread(self._load_system_prompt_sync, session_id)

    def _load_system_prompt_sync(self, session_id):
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id=? AND status!='deleted'", (session_id,)
            ).fetchone()
        if row is None or not row[0]:
            return None
        return row[0]

    async def list_sessions(self, user_id: str, tenant_id: str) -> list[dict]:
        """按 user/tenant 列出非删除会话（id/name/status/时间戳），最近更新倒序"""
        return await asyncio.to_thread(self._list_sessions_sync, user_id, tenant_id)

    def _list_sessions_sync(self, user_id, tenant_id):
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, name, status, created_at, updated_at
                   FROM sessions WHERE user_id=? AND tenant_id=? AND status!='deleted'
                   ORDER BY updated_at DESC""",
                (user_id, tenant_id),
            ).fetchall()
        return [
            {"session_id": r[0], "name": r[1], "status": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]
