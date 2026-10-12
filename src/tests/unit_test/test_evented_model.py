from __future__ import annotations

import threading

import pytest
from pydantic import BaseModel, Field, ValidationError

from langchain_lab.core.config.evented_model import EventedModel


# ===================== 测试用嵌套子模型 =====================
class SubConfig(BaseModel):
    """嵌套子模型，用于测试嵌套字段更新"""

    value: int = Field(default=0, ge=0, le=100)


# ===================== 测试用主模型（改名，避开Test前缀） =====================
class SampleEventedModel(EventedModel):
    """EventedModel 的最小测试子类，覆盖普通字段与嵌套字段"""

    name: str = "default"
    count: int = Field(default=10, ge=1, le=500)  # 上限放宽到500，满足并发测试
    enabled: bool = True
    sub: SubConfig = Field(default_factory=SubConfig)


# ===================== 第二个测试子类，验证信号独立 =====================
class AnotherEventedModel(EventedModel):
    """另一个子类，用于验证各子类信号字典相互独立"""

    title: str = "init_title"
    level: int = 1


# ===================== Fixture =====================
@pytest.fixture
def model() -> SampleEventedModel:
    return SampleEventedModel()


# ===================== 基础功能测试 =====================
def test_update_and_emit_triggers_event(model: SampleEventedModel):
    """测试 update_and_emit 修改字段后正确触发事件，old/new 值正确"""
    received_old = None
    received_new = None

    def callback(sender, old, new):
        nonlocal received_old, received_new
        received_old = old
        received_new = new

    model.subscribe("name", callback)
    model.update_and_emit("name", "new_name")

    assert received_old == "default"
    assert received_new == "new_name"
    assert model.name == "new_name"


def test_same_value_skip_event(model: SampleEventedModel):
    """测试新旧值相等时，不触发事件也不执行赋值"""
    hit_count = 0

    def callback(sender, old, new):
        nonlocal hit_count
        hit_count += 1

    model.subscribe("enabled", callback)
    # 赋值与原值完全相同
    model.update_and_emit("enabled", True)

    assert hit_count == 0


def test_direct_assignment_triggers_event(model: SampleEventedModel):
    """测试直接通过属性赋值也能触发事件（__setattr__ 逻辑生效）"""
    hit = False

    def callback(sender, old, new):
        nonlocal hit
        hit = True

    model.subscribe("name", callback)
    model.name = "direct_assign"

    assert hit is True
    assert model.name == "direct_assign"


def test_initialization_no_event():
    """测试实例初始化阶段赋值不触发事件，仅后续修改触发"""
    events = []

    def callback(sender, old, new):
        events.append((old, new))

    instance = SampleEventedModel(name="init_name")
    instance.subscribe("name", callback)

    # 初始化完成前的赋值不应产生事件
    assert len(events) == 0

    # 后续修改才触发事件
    instance.name = "modified"
    assert len(events) == 1
    assert events[0][0] == "init_name"
    assert events[0][1] == "modified"


# ===================== 订阅机制测试 =====================
def test_subscribe_unsubscribe(model: SampleEventedModel):
    """测试订阅接收事件，取消订阅后不再接收"""
    hit_count = 0

    def callback(sender, old, new):
        nonlocal hit_count
        hit_count += 1

    model.subscribe("count", callback)
    model.update_and_emit("count", 20)
    assert hit_count == 1

    model.unsubscribe("count", callback)
    model.update_and_emit("count", 30)
    assert hit_count == 1


def test_subscribe_once_only_fires_once(model: SampleEventedModel):
    """测试一次性订阅：触发一次后自动解绑"""
    hit_count = 0

    def callback(sender, old, new):
        nonlocal hit_count
        hit_count += 1

    model.subscribe_once("name", callback)
    model.update_and_emit("name", "first")
    model.update_and_emit("name", "second")

    assert hit_count == 1


def test_subscribe_all_receives_field_name(model: SampleEventedModel):
    """测试 subscribe_all 接收所有字段变更，回调正确携带 field_name"""
    changed_fields = set()

    def callback(sender, field_name, old, new):
        changed_fields.add(field_name)

    model.subscribe_all(callback)
    model.update_and_emit("name", "test")
    model.update_and_emit("enabled", False)
    model.update_and_emit("count", 50)

    assert changed_fields == {"name", "enabled", "count"}


def test_unsubscribe_all_stops_all(model: SampleEventedModel):
    """测试 unsubscribe_all 后不再接收任何字段事件"""
    hit_count = 0

    def callback(sender, field_name, old, new):
        nonlocal hit_count
        hit_count += 1

    model.subscribe_all(callback)
    model.update_and_emit("name", "a")
    assert hit_count == 1

    model.unsubscribe_all(callback)
    model.update_and_emit("name", "b")
    model.update_and_emit("enabled", False)
    assert hit_count == 1


# ===================== 子模型更新测试 =====================
def test_nested_submodel_update_by_dict(model: SampleEventedModel):
    """测试嵌套子模型通过字典更新，触发父级字段事件"""
    received_new = None

    def callback(sender, old, new):
        nonlocal received_new
        received_new = new

    model.subscribe("sub", callback)
    model.update_and_emit("sub", {"value": 50})

    assert isinstance(received_new, SubConfig)
    assert received_new.value == 50
    assert model.sub.value == 50


def test_nested_submodel_replace_instance(model: SampleEventedModel):
    """测试嵌套子模型直接替换整个实例，也能正常触发事件"""
    received_new = None

    def callback(sender, old, new):
        nonlocal received_new
        received_new = new

    new_sub = SubConfig(value=88)
    model.subscribe("sub", callback)
    model.update_and_emit("sub", new_sub)

    assert received_new is new_sub
    assert model.sub.value == 88


# ===================== 批量/原子更新测试 =====================
def test_batch_update_and_emit(model: SampleEventedModel):
    """测试批量更新：多个字段依次触发事件，值全部更新"""
    events = []

    def callback(sender, field_name, old, new):
        events.append(field_name)

    model.subscribe_all(callback)
    model.batch_update_and_emit({"name": "batch", "enabled": False, "sub": {"value": 30}})

    assert "name" in events
    assert "enabled" in events
    assert "sub" in events
    assert model.name == "batch"
    assert model.enabled is False
    assert model.sub.value == 30


def test_atomic_update_and_emit_basic(model: SampleEventedModel):
    """测试原子更新基础功能：值正确变更并触发事件"""
    hit = False

    def callback(sender, old, new):
        nonlocal hit
        hit = True

    model.subscribe("count", callback)
    model.atomic_update_and_emit("count", 99)

    assert hit is True
    assert model.count == 99


# ===================== 纯事件发射测试 =====================
def test_emit_event_only_broadcast_no_modify(model: SampleEventedModel):
    """测试 emit_event 只发射事件，不修改模型字段值"""
    hit = False
    original = model.name

    def callback(sender, old, new):
        nonlocal hit
        hit = True

    model.subscribe("name", callback)
    model.emit_event("name", original, "fake_value")

    assert hit is True
    assert model.name == original  # 模型值保持不变


# ===================== 异常与校验测试 =====================
def test_unknown_field_raises_keyerror(model: SampleEventedModel):
    """测试访问不存在的字段抛出 KeyError"""
    with pytest.raises(KeyError):
        model.update_and_emit("not_exist_field", 123)
    with pytest.raises(KeyError):
        model.subscribe("not_exist_field", lambda s, o, n: None)


def test_pydantic_validation_works(model: SampleEventedModel):
    """测试 Pydantic 字段边界校验仍然生效（直接赋值 + update_and_emit 双路径）"""
    # 1. 直接属性赋值越界，抛出 ValidationError
    with pytest.raises(ValidationError, match="less_than_equal"):
        model.count = 600

    # 2. update_and_emit 普通字段越界，抛出 ValidationError
    with pytest.raises(ValidationError, match="less_than_equal"):
        model.update_and_emit("count", 600)

    # 3. 子模型嵌套字段越界，抛出 ValidationError
    with pytest.raises(ValidationError, match="greater_than_equal"):
        model.update_and_emit("sub", {"value": -1})


# ===================== 架构隔离测试 =====================
def test_subclass_signals_independent():
    """测试不同子类的信号字典相互独立，不共享污染"""
    # 触发一次订阅，确保信号惰性创建
    SampleEventedModel()._get_signal("name")
    AnotherEventedModel()._get_signal("title")

    # 每个子类只包含自身定义的字段信号
    assert set(SampleEventedModel._signals.keys()).issubset({"name", "count", "enabled", "sub"})
    assert set(AnotherEventedModel._signals.keys()).issubset({"title", "level"})
    # 信号字典不是同一个对象
    assert SampleEventedModel._signals is not AnotherEventedModel._signals


# ===================== 线程安全测试 =====================
def test_multi_thread_atomic_update_safe():
    """测试多线程并发下 atomic_update_and_emit 的线程安全性"""
    model = SampleEventedModel()
    initial_count = model.count
    iterations = 100
    thread_count = 2

    def worker():
        for _ in range(iterations):
            current = model.count
            model.atomic_update_and_emit("count", current + 1)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 初始值 + 线程数 × 循环次数
    assert model.count == initial_count + thread_count * iterations


# ===================== 异步事件测试 =====================
@pytest.mark.asyncio
async def test_emit_event_async(model: SampleEventedModel):
    """测试异步发射事件正常执行异步回调"""
    hit = False

    async def async_callback(sender, old, new):
        nonlocal hit
        hit = True

    model.subscribe_async("name", async_callback)
    await model.emit_event_async("name", "old_val", "async_new")

    assert hit is True
