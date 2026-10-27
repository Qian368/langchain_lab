"""配置面：ConfigResolver 覆盖链（全局 → 租户 → 用户 → EffectiveConfig）

- 全局 yaml（config/*.yaml）是部署级默认；租户/用户配置（UserConfigStore）是运行级覆盖。
- resolve 把 overrides 按域路径深合并到全局模型，产出 EffectiveConfig 注入会话。
- 覆盖示例：{"context": {"keep_recent_turns": 8}}、{"llm": {"default_instance": "zhipuai"}}。
- 会话/接入层不读全局，只从注入的 EffectiveConfig 取用；overrides 为空时直接返回全局（零拷贝）。
"""
from __future__ import annotations

from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归深合并：dict 键递归，非 dict 直接覆盖（保证嵌套域可部分覆盖）"""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_config(global_cfg: Any, overrides: dict[str, Any] | None = None) -> Any:
    """把覆盖合并到全局配置，产出 EffectiveConfig。

    overrides 为空 → 直接返回全局（不重建，兼容 mock/零覆盖场景）。
    非空 → 深合并后重建配置对象（触发 schema 校验，无效覆盖在此抛错）。
    """
    if not overrides:
        return global_cfg
    merged = deep_merge(global_cfg.model_dump(), dict(overrides))
    return global_cfg.__class__(**merged)
