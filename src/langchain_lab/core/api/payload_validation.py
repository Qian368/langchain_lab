"""从指令注册表生成 payload 的 Pydantic 校验模型

注册表（commands/instruction_registry.py）的每条指令声明 payload 字段描述
（type / required / minimum / minLength / description），这里据此用 pydantic.create_model
动态生成每个指令的校验模型，避免"每个指令手写一个 Pydantic 类 + 注册表各写一份字段"。
校验是 API 层的事（FastAPI/Pydantic），注册表只提供 schema，二者不重复实现。
"""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import BaseModel, Field, create_model

from langchain_lab.core.commands import get_spec

_TYPE_MAP = {"integer": int, "string": str}


@cache
def _model_for(action: str) -> type[BaseModel]:
    """按注册表 payload schema 动态生成该指令的校验模型（缓存）"""
    spec = get_spec(action)
    payload = (spec or {}).get("payload") or {}
    fields: dict[str, Any] = {}
    for fname, meta in payload.items():
        typ = _TYPE_MAP.get(meta.get("type"), str)
        kwargs: dict = {"description": meta.get("description")}
        if meta.get("minimum") is not None:
            kwargs["ge"] = meta["minimum"]
        if meta.get("minLength") is not None:
            kwargs["min_length"] = meta["minLength"]
        if meta.get("required"):
            fields[fname] = (typ, Field(..., **kwargs))
        else:
            fields[fname] = (typ | None, Field(None, **kwargs))
    name = "".join(p.title() for p in action.split("_")) + "Payload"
    return create_model(name, **fields)


def validate_payload(action: str, payload: dict) -> dict:
    """按注册表 schema 校验 payload，返回干净 dict（多余字段被忽略）。

    无载荷指令（payload schema 为空）返回空 dict。
    """
    model = _model_for(action)
    if not model.model_fields:
        return {}
    return model.model_validate(payload).model_dump(exclude_unset=True)
