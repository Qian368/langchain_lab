from contextvars import ContextVar
from threading import Lock
from typing import Any

from langchain_lab.core.config.app_schema.mcp_models import SessionCtxTemplate

_session_ctx: ContextVar[SessionCtxTemplate] = ContextVar("exec_context")
_session_ctx_template: SessionCtxTemplate
_template_lock = Lock()

# 动态计算顶层核心字段集合：所有模型字段 - extra 扩展容器
_TOP_FIELDS = set(SessionCtxTemplate.model_fields.keys()) - {"extra"}


class _ResetToken:
    def __init__(self, cv: ContextVar, token):
        self._cv = cv
        self._token = token

    def reset(self) -> None:
        self._cv.reset(self._token)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()


# ===== 对外暴露API =====
def setup_exec_context() -> None:
    """服务启动：初始化全局模板，订阅配置变更回调"""
    global _session_ctx_template
    from langchain_lab.core.config.config_core import get_config

    config = get_config()

    with _template_lock:
        # ✅ 改用浅拷贝，避开锁对象深拷贝异常
        base_template = config.mcp.ctx_template.model_copy(deep=False)
        # ✅ 手动拷贝 extra 字典，切断与配置源的引用共享
        base_template.extra = base_template.extra.copy()
        _session_ctx_template = base_template

    # 订阅配置模板内部所有字段变更，同步更新全局模板
    def on_session_ctx_change(sender, field_name, old_val, new_val):
        global _session_ctx_template
        with _template_lock:
            setattr(_session_ctx_template, field_name, new_val)

    config.mcp.ctx_template.subscribe_all(on_session_ctx_change)


def bind_exec_context(
    override_session_ctx: dict[str, Any] | None = None,
    extra_session_ctx: dict[str, Any] | None = None,
) -> _ResetToken:
    """
    会话入口：从全局模板拷贝快照，绑定到当前协程
    :param override_session_ctx: 顶层字段覆盖，作用于所有模型核心字段（HTTP Header身份用）
    :param extra_session_ctx: 扩展字段增补，仅写入 .extra 字典，只新增不覆盖已有key
    """
    with _template_lock:
        base_template = _session_ctx_template.model_copy(deep=False)
        # 切断 extra 字典引用，避免全局模板与会话快照共享可变对象
        snapshot_extra = base_template.extra.copy()

    # ✅ 动态生成顶层字段快照，不再硬编码字段名
    snapshot_top = {k: getattr(base_template, k) for k in _TOP_FIELDS}

    # 1. 顶层覆盖：只允许修改核心字段，自动过滤非法key
    if override_session_ctx is not None:
        for k, v in override_session_ctx.items():
            if k in _TOP_FIELDS:
                snapshot_top[k] = v

    # 2. 扩展增补：只新增不存在的key，不修改已有扩展值
    if extra_session_ctx is not None:
        for k, v in extra_session_ctx.items():
            if k not in snapshot_extra:
                snapshot_extra[k] = v

    # 组装回强类型模型
    snapshot = SessionCtxTemplate(
        **snapshot_top,
        extra=snapshot_extra,
    )

    token = _session_ctx.set(snapshot)
    return _ResetToken(_session_ctx, token)


def get_exec_context() -> SessionCtxTemplate:
    """下游工具直接调用，拿到当前协程绑定的身份快照"""
    return _session_ctx.get()
