"""控制面指令注册表：通信格式约定的统一编写处（全局单例字典）

- 指令集是"通信格式约定"，不是运行时可配置项 → 用 Python 常量（模块级单例），
  不走 yaml/jsonc 配置中心，也不需要加载器；直接 import，IDE 补全/类型友好
- 所有指令的定义（name / aliases / description / payload 字段 / handler）集中一处，
  避免各处写重复字面量（写错难查）；新增指令只在这里加一条
- 前端帮助请求：build_public_specs() 直接构造控制面 API 列表（不含内部 handler）
- 运行时路由：_handle_command_event 按 name 查 handler_for → getattr(self, ...) 通用分发，
  不散落 if/elif 与指令名字符串
"""

from __future__ import annotations

from typing import Any

# 全局单例：指令契约（通信格式约定，随代码版本演进）
# 每条：aliases=终端别名；description=帮助文案；scope=作用域；
#       payload=载荷字段说明（JSON 可序列化，供前端渲染表单 / 将来 API 层生成校验）；
#       handler=内部分派目标方法名；standalone=独立路径（自带 turn 边界，不走通用分发）
INSTRUCTION_SPECS: dict[str, dict[str, Any]] = {
    "branch_list": {
        "aliases": [":branches", "/branches", ":bl"],
        "description": "列出当前会话的全部分支（含拐点序号与当前分支标记）",
        "scope": "session",
        "payload": {},  # 无载荷
        "handler": "_do_branch_list",
    },
    "branch_create": {
        "aliases": [":branch", "/branch"],
        "description": "在指定对话消息序号处创建子分支，后续输入进入该分支",
        "scope": "session",
        "payload": {
            "point": {
                "type": "integer",
                "minimum": 1,
                "required": True,
                "description": "对话消息序号（>=1，相对真实消息）",
            },
        },
        "handler": "_do_branch_create",
    },
    "branch_switch": {
        "aliases": [":switch", "/switch"],
        "description": "切换到指定分支（restore 该分支视图）",
        "scope": "session",
        "payload": {
            "branch_id": {
                "type": "string",
                "required": True,
                "description": "目标分支 id",
            },
        },
        "handler": "_do_branch_switch",
    },
    "branch_delete": {
        "aliases": [":delete", "/delete"],
        "description": "删除指定分支（级联软删子树；branch_id=root 删除整个会话）",
        "scope": "session",
        "payload": {
            "branch_id": {
                "type": "string",
                "required": True,
                "description": "待删分支 id；root=删除整个会话",
            },
        },
        "handler": "_do_branch_delete",
    },
    "rename": {
        "aliases": [":rename", "/rename"],
        "description": "重命名当前分支（root 分支则重命名会话）",
        "scope": "session",
        "payload": {
            "name": {
                "type": "string",
                "minLength": 1,
                "required": True,
                "description": "新名字（非空）",
            },
        },
        "handler": "_do_rename",
    },
    "update-compressor": {
        "aliases": [":update-compressor", "/update-compressor"],
        "description": "提供上下文压缩器配置使用",
        "scope": "session",
        "payload": {
            "retention_budget": {
                "type": "integer",
                "minimum": 1,
                "required": True,
                "description": "压缩后保留预算（token）；= 模型窗口 × (1 - headroom)，可手动覆盖",
            },
            "keep_recent_turns": {
                "type": "integer",
                "minimum": 1,
                "required": True,
                "description": "压缩后保留的最近轮数（下限）",
            },
        },
        "handler": "_do_update_compressor",
    },
    # ===== 独立路径指令：自带 turn 边界/归档/作废，不走通用指令分发 =====
    "regenerate": {
        "aliases": [":rerun", "/rerun", ":regenerate"],
        "description": "重跑当前最后一条人类消息对应轮（复用工具结果，作废旧回答）",
        "scope": "session",
        "payload": {},
        "handler": "_do_regenerate",
        "standalone": True,  # 独立路径：自带 turn 边界/归档/作废，不走通用分发
    },
}


def get_spec(name: str | None) -> dict[str, Any] | None:
    """按 name 取指令契约；未知指令（含 None）返回 None"""
    if name is None:
        return None
    return INSTRUCTION_SPECS.get(name)


def is_known(name: str) -> bool:
    """指令名是否在注册表中"""
    return name in INSTRUCTION_SPECS


def handler_for(name: str | None) -> str:
    """返回该指令的分派方法名（运行时 getattr(self, name) 调用）；None/未注册指令抛 KeyError"""
    if name is None:
        raise KeyError(name)
    return INSTRUCTION_SPECS[name]["handler"]


def build_public_specs() -> list[dict[str, Any]]:
    """构造面向前端/API 的控制面指令列表（去掉内部 handler 字段），帮助请求直接返回。

    每条包含 name（前端据此发 POST /commands 的 action）与 aliases/description/scope/payload。
    """
    return [{"name": name, **{k: v for k, v in spec.items() if k != "handler"}} for name, spec in INSTRUCTION_SPECS.items()]
