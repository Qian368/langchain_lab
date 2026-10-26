"""环境变量占位展开工具：${VAR} / %VAR% → 真实值

供配置模型在加载期调用，避免把密钥/连接串明文写进 yaml：
- yaml 里写 ${SOME_VAR} 占位
- 真实值放 .env（gitignore，不入库）
- 加载后由本工具替换为环境变量实际值

展开支持任意嵌套：str / dict（含嵌套 dict）/ list（含嵌套 list）统一递归处理。
"""
from __future__ import annotations

import os
import re
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|%([^%]+)%")


def expand_vars(value: str) -> str:
    """展开单个字符串中的 ${VAR}/%VAR%：存在则替换，缺失则替换为空串"""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, "")

    return _ENV_PATTERN.sub(_replace, value)


def _expand_value(value: Any) -> Any:
    """递归展开一个值中的占位符：dict/list 逐项递归，str 替换，其它原样返回"""
    if isinstance(value, dict):
        return {k: _expand_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_value(v) for v in value]
    if isinstance(value, str):
        return expand_vars(value)
    return value


def expand_model_fields(obj: Any, fields: tuple[str, ...]) -> None:
    """对 EventedModel 的若干字段做递归占位展开（dict/list 任意嵌套，对象就地修改）。

    ⚠️ 用 object.__setattr__ 绕过 EventedModel 的赋值事件发射：
    加载期展开，不发事件（_initialized 之前本身也不发事件，这里显式保证一致性）。
    """
    for field in fields:
        value = getattr(obj, field, None)
        expanded = _expand_value(value)
        if expanded is not value:
            object.__setattr__(obj, field, expanded)


def expand_model_env(obj: Any) -> None:
    """展开对象【所有字段】中的 ${VAR}/%VAR% 占位（dict/list 任意嵌套递归）。

    共享加载期入口：统一"任意字符串字段都支持占位符"的语义，
    避免各配置模型各自维护字段白名单造成认知不一致。
    - 用 type(obj).model_fields（类访问），避免 Pydantic V3 弃用实例 .model_fields
    - object.__setattr__ 绕过 EventedModel 赋值事件发射（加载期不发事件）
    - 非 str/dict/list 字段（int/float/Path/bool/嵌套 EventedModel）原样返回：
      嵌套 EventedModel 实例内部字段由该子模型自身的 expand_env validator 负责递归
    """
    for field in type(obj).model_fields:
        value = getattr(obj, field)
        expanded = _expand_value(value)
        if expanded is not value:
            object.__setattr__(obj, field, expanded)
