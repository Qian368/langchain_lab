"""环境变量占位展开工具：${VAR} / %VAR% → 真实值

供配置模型在加载期调用，避免把密钥/连接串明文写进 yaml：
- yaml 里写 ${SOME_VAR} 占位
- 真实值放 .env（gitignore，不入库）
- 加载后由本工具替换为环境变量实际值
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


def expand_model_fields(obj: Any, fields: tuple[str, ...]) -> None:
    """对 EventedModel 的若干 str 字段做占位展开（对象就地修改）。

    ⚠️ 用 object.__setattr__ 绕过 EventedModel 的赋值事件发射：
    加载期展开，不发事件（_initialized 之前本身也不发事件，这里显式保证一致性）。
    """
    for field in fields:
        value = getattr(obj, field, None)
        if isinstance(value, str):
            object.__setattr__(obj, field, expand_vars(value))
