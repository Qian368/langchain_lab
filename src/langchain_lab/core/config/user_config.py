"""配置面：用户/租户配置存储（SQLite 单表，按身份维度持久化用户设置覆盖）

- 配置面 = 用户/租户维度的设置（模型选择、上下文策略、提示词等），与全局 yaml 分离：
  全局是部署级默认（config/*.yaml），此处是运行级覆盖（用户/租户可改）。
- ConfigResolver 把 全局 + 覆盖 合并出 EffectiveConfig 注入会话；会话/接入层不读全局。
- 连接策略与 context_store 一致：每次独立连接 + 线程锁串行（避免跨协程/线程共享连接）。
"""
from __future__ import annotations

import json
import sqlite3
import threading


class UserConfigStore:
    """用户/租户配置持久化（按 tenant_id + user_id 维度）"""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        conn = self._connect()
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS user_config (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (tenant_id, user_id)
                )"""
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, tenant_id: str, user_id: str) -> dict:
        """读取该用户/租户的配置覆盖（无则空 dict）"""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT config_json FROM user_config WHERE tenant_id=? AND user_id=?",
                    (tenant_id, user_id),
                ).fetchone()
                if row is None:
                    return {}
                return json.loads(row["config_json"])
            finally:
                conn.close()

    def set(self, tenant_id: str, user_id: str, config: dict) -> None:
        """写入该用户/租户的配置覆盖（整量替换，upsert）"""
        payload = json.dumps(config, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO user_config (tenant_id,user_id,config_json,updated_at) VALUES (?,?,?,datetime('now')) "
                    "ON CONFLICT(tenant_id,user_id) DO UPDATE SET config_json=excluded.config_json, updated_at=datetime('now')",
                    (tenant_id, user_id, payload),
                )
                conn.commit()
            finally:
                conn.close()
