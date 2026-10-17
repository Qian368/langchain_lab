"""
全局配置单例
全进程唯一配置入口，统一管理初始化与重载
"""

import threading

from langchain_lab.core.config.app_config import AppConfig
from langchain_lab.core.config.evented_model import EventedModel
from langchain_lab.core.config.ident_config import IdentConfig
from langchain_lab.core.config.loader import load_config

# 全局单例
_CONFIG: AppConfig | None = None
_CFG_LOCK = threading.Lock()


# region ===== 全局配置 =====
def get_config() -> AppConfig:
    """获取全局配置单例（线程安全）"""
    global _CONFIG
    if _CONFIG is None:
        with _CFG_LOCK:
            if _CONFIG is None:
                _CONFIG = load_config()
    return _CONFIG


def reload_config() -> AppConfig:
    """
    全局平台配置热重载
    加载新yaml/env配置，递归同步到原有全局实例，订阅全部保留
    """
    global _CONFIG
    with _CFG_LOCK:
        new_cfg = load_config()
        if _CONFIG is None:
            return new_cfg
        sync_model_fields(_CONFIG, new_cfg)
    return _CONFIG


def sync_model_fields(dst: EventedModel, src: EventedModel):
    """
    【全局配置专用】递归把src的值同步到dst
    - 不替换任何子模型对象引用，只修改叶子字段
    - 只有值不一样时才赋值，触发EventedModel的变更事件
    - dst：原有全局实例（保留所有已注册订阅）
    - src：从yaml/env新加载出来的临时实例
    """
    for field_name in type(src).model_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)

        # 如果两边都是EventedModel子模型，递归深入，不替换对象
        if isinstance(src_val, EventedModel) and isinstance(dst_val, EventedModel):
            sync_model_fields(dst_val, src_val)
        else:
            # 叶子字段：值不同才赋值，触发事件
            if dst_val != src_val:
                setattr(dst, field_name, src_val)


# endregion

# region ===== 客户端配置 =====
# key: (tenant_id, user_id), value: 用户独立EventedModel配置实例
_ident_cfg_cache: dict[tuple[str, str], IdentConfig] = {}
_cache_cfg_lock = threading.RLock()


def reload_tenant_config(tenant_id: str, user_id: str) -> IdentConfig:
    """
    重载用户配置：新建EventedModel实例，注册订阅，替换缓存
    旧实例无引用后GC回收，旧订阅自动失效
    """
    with _cache_cfg_lock:
        # 1. DB读取原始配置
        raw_dict = load_tenant_config_from_db(tenant_id, user_id)
        # 2. 构造全新独立EventedModel实例
        new_user_cfg = IdentConfig(**raw_dict)
        # 3. 给新实例挂载订阅回调
        init_tenant_cache_mirror(new_user_cfg)
        # 4. 元组作为key，替换缓存里旧实例
        cache_key = (tenant_id, user_id)
        _ident_cfg_cache[cache_key] = new_user_cfg
        return new_user_cfg


def get_tenant_config(tenant_id: str, user_id: str) -> IdentConfig:
    """
    获取用户配置，缓存不存在时自动从DB加载并放入缓存
    """
    cache_key = (tenant_id, user_id)
    with _cache_cfg_lock:
        if cache_key in _ident_cfg_cache:
            return _ident_cfg_cache[cache_key]
    # 不在缓存，触发加载（锁内只做判断，DB/实例创建放锁外，减少锁持有时间）
    return reload_tenant_config(tenant_id, user_id)


def load_tenant_config_from_db(tenant_id: str, user_id: str) -> dict:
    """从数据库读取原始配置字典，业务自行实现查询逻辑"""
    # TODO: select config_json from user_config where tenant_id=? and user_id=?
    return {}


def init_tenant_cache_mirror(cfg: IdentConfig) -> None:
    """
    为新创建的用户配置实例注册订阅回调
    用户配置：实例独立，不做递归字段同步；实例销毁订阅自动解绑
    """
    # TODO: 在这里订阅 cfg 的字段变更事件，挂载业务回调
    # cfg.subscribe("some_field", callback_func)
    return None


# endregion
