from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any, ClassVar

from blinker import NamedSignal, signal
from pydantic import BaseModel, ConfigDict


# ============ 基类：字段即属性，赋值自动发射事件 ============
class EventedModel(BaseModel):
    # 显式使用 ConfigDict，确保 validate_assignment 配置生效
    model_config = ConfigDict(validate_assignment=True)

    # 类级信号字典：每个子类独立拥有，惰性填充
    _signals: ClassVar[dict[str, NamedSignal]]

    # 内部虚拟字段：实例整体变更事件标识
    _INSTANCE_EVENT = "__instance__"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 每个子类先创建独立空字典，信号惰性生成
        cls._signals = {}

    def __init__(self, **data):
        super().__init__(**data)
        # 实例级线程锁：可重入锁，支持批量/原子更新嵌套加锁
        self._lock = threading.RLock()
        # 初始化完成标记：只有初始化完成后的修改才触发事件
        self._initialized = True

    def __setattr__(self, name: str, value: Any):
        # 非模型字段直接走父类
        if name not in type(self).model_fields:
            super().__setattr__(name, value)
            return

        # 初始化阶段：只赋值，不发射事件
        if not getattr(self, "_initialized", False):
            super().__setattr__(name, value)
            return

        # 线程安全：加锁保护读取-比较-赋值-发射全过程
        with self._lock:
            old = getattr(self, name, None)
            # 新旧值相等，直接跳过赋值和信号触发
            if old == value:
                return
            super().__setattr__(name, value)  # Pydantic 校验并赋值
            self._publish_sync(name, old, value)

    # ---------- 内部工具：惰性获取/创建字段信号 ----------
    def _get_signal(self, field: str) -> NamedSignal:
        """惰性获取字段信号，不存在则创建；字段非法抛 KeyError"""
        cls = type(self)
        # 内部虚拟字段（实例级事件）跳过模型字段校验
        if field != self._INSTANCE_EVENT and field not in cls.model_fields:
            raise KeyError(f"未知配置项: {field}")
        if field not in cls._signals:
            cls._signals[field] = signal(f"config:{cls.__name__}:{field}")
        return cls._signals[field]

    # ---------- 内部私有发射原语 ----------
    def _publish_sync(self, field: str, old: Any, new: Any) -> None:
        """内部同步发射字段变更事件"""
        sig = self._get_signal(field)
        sig.send(self, old=old, new=new)

    async def _publish_async(self, field: str, old: Any, new: Any) -> None:
        """内部异步发射字段变更事件（会 await 所有协程回调）"""
        sig = self._get_signal(field)
        await sig.send_async(self, old=old, new=new)

    # ---------- 公开兜底：仅发射事件，不修改数据 ----------
    def emit_event(self, field: str, old: Any, new: Any) -> None:
        """
        手动同步发射字段变更事件。
        ⚠️ 只广播事件！不会修改模型字段，不执行 Pydantic 校验。
        绝大多数业务场景不要使用，优先使用 update_and_emit()。
        """
        self._publish_sync(field, old, new)

    async def emit_event_async(self, field: str, old: Any, new: Any) -> None:
        """
        手动异步发射字段变更事件（会 await 所有协程回调）。
        ⚠️ 只广播事件！不会修改模型字段，不执行 Pydantic 校验。
        绝大多数业务场景不要使用，优先使用 update_and_emit()。
        """
        await self._publish_async(field, old, new)

    # ---------- 主业务API：更新并发射事件 ----------
    def update_and_emit(self, field: str, changes: Any = None, **kw_changes):
        """更新配置，修改完成后自动发射变更事件。
        用法：
            config.update_and_emit("enable_tools", False)          # 普通字段：位置传新值
            config.update_and_emit("llm", {"temperature": 0.2})    # 子模型：位置传更新字典
            config.update_and_emit("llm", temperature=0.2)         # 子模型：关键字传子字段
            config.update_and_emit("system_prompt", "新提示词")     # 普通字段：位置传新值
        注意：普通字段仅支持位置参数传新值，关键字参数仅对子模型生效。
        """
        # 字段合法性校验
        self._get_signal(field)
        current = getattr(self, field)

        # 子模型字段：支持字典合并更新，或直接替换整个模型
        if isinstance(current, BaseModel):
            if isinstance(changes, dict):
                changes = {**changes, **kw_changes}
            elif changes is None:
                changes = kw_changes
            # 如果 changes 本身就是模型实例，直接使用；否则构造新实例（强制完整校验）
            if isinstance(changes, BaseModel):
                new_value = changes
            else:
                # 先合并字典再解包，避免重复关键字；强制走初始化完整校验
                merged_data = {**current.model_dump(), **changes}
                new_value = type(current)(**merged_data)
        else:
            # 普通字段：直接使用 changes 作为新值，忽略 kw_changes
            new_value = changes

        # 触发 __setattr__，自动发射事件（内部自带 old==new 判断和锁）
        setattr(self, field, new_value)
        return new_value

    def atomic_update_and_emit(self, field: str, changes: Any = None, **kw_changes):
        """
        原子更新单个字段（线程安全），内部自动加锁并发射事件。
        多线程场景下单字段更新推荐用这个；多字段批量更新请使用 batch_update_and_emit。
        """
        # 锁已下沉到 __setattr__，这里保留接口兼容，逻辑等价
        return self.update_and_emit(field, changes, **kw_changes)

    def batch_update_and_emit(self, updates: dict[str, Any]) -> None:
        """
        批量原子更新多个字段（线程安全），全部字段在同一锁内执行，依次发射字段事件，
        全部更新完成后额外发射一次【实例整体变更事件】。
        :param updates: 字段名->更新值的字典，子模型字段支持传嵌套字典。
        示例：
            config.batch_update_and_emit({
                "enable_tools": False,
                "llm": {"temperature": 0.2},
                "system_prompt": "新提示词"
            })
        """
        with self._lock:
            # 批量更新前完整快照
            old_snapshot = self.model_dump()
            for field, value in updates.items():
                if isinstance(value, dict):
                    self.update_and_emit(field, value)
                else:
                    self.update_and_emit(field, value)
            # 批量更新完成，发射实例整体变更事件
            new_snapshot = self.model_dump()
            self._publish_sync(self._INSTANCE_EVENT, old_snapshot, new_snapshot)

    # ================== 发布订阅封装 ==================
    def subscribe(self, field: str, callback: Callable[[Any, Any, Any], None]) -> None:
        """
        订阅单个字段变更事件。
        :param field: 字段名
        :param callback: 回调函数 callback(sender, old, new)
        """
        sig = self._get_signal(field)
        sig.connect(callback, sender=self)

    def unsubscribe(self, field: str, callback: Callable[[Any, Any, Any], None]) -> None:
        """取消单个字段的订阅"""
        sig = self._get_signal(field)
        sig.disconnect(callback, sender=self)

    def subscribe_once(self, field: str, callback: Callable[[Any, Any, Any], None]) -> None:
        """
        一次性订阅单个字段变更，触发一次后自动解绑。
        :param field: 字段名
        :param callback: 回调函数 callback(sender, old, new)
        """

        def wrapper(sender, old, new):
            self.unsubscribe(field, wrapper)
            callback(sender, old, new)

        # 保存 wrapper 引用，防止提前回收
        object.__setattr__(self, f"_once_wrap_{id(callback)}", wrapper)
        self.subscribe(field, wrapper)

    def subscribe_all(self, callback: Callable[[Any, str, Any, Any], None]) -> None:
        """
        订阅当前实例【所有字段】变更。
        回调签名：callback(sender, field_name, old, new)
        """
        wrappers: dict[str, Callable] = {}
        # 以模型定义的字段为准
        for field_name in type(self).model_fields:

            def wrapper(sender, old, new, fname=field_name):
                callback(sender, fname, old, new)

            wrappers[field_name] = wrapper
            self.subscribe(field_name, wrapper)
        # 保存所有 wrapper 引用用于解绑
        object.__setattr__(self, f"_wrap_all_{id(callback)}", wrappers)

    def unsubscribe_all(self, callback: Callable[[Any, str, Any, Any], None]) -> None:
        """取消全部字段订阅，和 subscribe_all 配对"""
        wrappers = getattr(self, f"_wrap_all_{id(callback)}", None)
        if wrappers is None:
            return
        for field_name, wrapper in wrappers.items():
            self.unsubscribe(field_name, wrapper)
        with contextlib.suppress(AttributeError):
            delattr(self, f"_wrap_all_{id(callback)}")

    # ========== 新增：实例级整体变更订阅 ==========
    def subscribe_instance(self, callback: Callable[[Any, Any, Any], None]) -> None:
        """
        订阅【实例整体变更事件】，仅在 batch_update_and_emit 批量更新完成后触发一次。
        回调签名：callback(sender, old_snapshot, new_snapshot)
        - old_snapshot / new_snapshot 为批量更新前后的完整模型字典快照
        ⚠️ 单个字段直接赋值 / update_and_emit 不会触发此事件，仅批量更新触发。
        适用场景：配置整体变更后统一刷新缓存、重建上下文、重载资源等。
        """
        sig = self._get_signal(self._INSTANCE_EVENT)
        sig.connect(callback, sender=self)

    def unsubscribe_instance(self, callback: Callable[[Any, Any, Any], None]) -> None:
        """取消实例整体变更事件订阅"""
        sig = self._get_signal(self._INSTANCE_EVENT)
        sig.disconnect(callback, sender=self)

    def subscribe_async(self, field: str, callback: Callable[[Any, Any, Any], Any]) -> None:
        """
        订阅单个字段变更（异步回调版本）。
        ⚠️ 注意：异步回调仅在调用 emit_event_async() 时才会被 await 执行；
        普通同步更新（update_and_emit / 直接赋值）不会执行异步回调。
        :param field: 字段名
        :param callback: 异步回调函数 async callback(sender, old, new)
        """
        self.subscribe(field, callback)

    def unsubscribe_async(self, field: str, callback: Callable[[Any, Any, Any], Any]) -> None:
        """取消异步订阅回调"""
        self.unsubscribe(field, callback)
