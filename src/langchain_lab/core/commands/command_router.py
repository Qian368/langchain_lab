"""CommandRouter：控制面指令分发与实现（从 ChatSession 抽出的协作者）

- 职责：按指令注册表（instruction_registry）分发并实现管理类指令
  （branch_list/create/switch/delete、rename、update-compressor）。
- ChatSession 收敛为聚合根（状态机+事件源+一轮编排），指令处理委托本类；
  通过 SessionCommandTarget 协议操作会话（读/写视图、分支、压缩器、发事件），解耦不反向依赖具体类。
- regenerate 是独立路径（强耦合对话循环/归档），保留在 ChatSession。
- 新增指令只需在注册表加一条 + 本类加同名 handler，会话与接入层自动多一个可用动作。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from langchain_core.messages import BaseMessage

from langchain_lab.core.bus.events import Event, EventType
from langchain_lab.core.commands.instruction_registry import get_spec, handler_for
from langchain_lab.core.context.compressor import SessionCompressor


class SessionCommandTarget(Protocol):
    """会话命令操作接口：CommandRouter 通过它操作会话，不依赖 ChatSession 具体实现"""

    messages: list[BaseMessage]
    branch_id: str
    compressor: SessionCompressor | None

    branch_lister: Callable[[], Awaitable[list[dict]]] | None
    branch_creator: Callable[[str, int], Awaitable[tuple[str, list[BaseMessage]]]] | None
    branch_loader: Callable[[str], Awaitable[list[BaseMessage] | None]] | None
    branch_deleter: Callable[[str], Awaitable[list[str]]] | None
    session_renamer: Callable[[str], Awaitable[None]] | None
    branch_renamer: Callable[[str, str], Awaitable[None]] | None

    def notice(self, text: str) -> None: ...
    def emit(self, event_type: EventType, payload: dict | None = None) -> None: ...
    async def flush(self) -> None: ...
    def emit_session_state(self) -> None: ...


class CommandRouter:
    """指令分发器：按注册表 name 通用分发；各管理类指令 handler 实现于此"""

    async def handle(self, target: SessionCommandTarget, event: Event) -> None:
        """分发控制指令（适配器已解析+校验 action/payload）；不解析原始文本"""
        target.emit(EventType.AGENT_TURN_START)
        payload = event.payload or {}
        action = payload.get("action")
        if get_spec(action) is None:
            target.notice(f"未知指令 action={action}（可用指令见 /help）")
            target.emit_session_state()
            target.emit(EventType.AGENT_TURN_DONE)
            await target.flush()
            return
        try:
            handler = getattr(self, handler_for(action))
            await handler(target, payload)
            target.emit_session_state()
        except Exception as exc:
            target.emit(EventType.ERROR_OCCURRED, payload={"error": repr(exc), "phase": "command"})
        finally:
            target.emit(EventType.AGENT_TURN_DONE)
            await target.flush()

    # —— 分支操作（走注入回调，运行时不碰持久化实现）——
    async def _do_branch_list(self, target: SessionCommandTarget, payload: dict) -> None:
        if target.branch_lister is None:
            target.notice("无上下文存储，无法列出分支")
            return
        branches = await target.branch_lister()
        if not branches:
            target.notice("暂无分支")
            return
        lines = [
            f"  {b['branch_id']}  parent={b['parent_id'] or '-'}  point={b['branch_point_seq']}  {b.get('name', '')}"
            + (" *" if b["branch_id"] == target.branch_id else "")
            for b in branches
        ]
        target.notice("分支列表：\n" + "\n".join(lines))

    async def _do_branch_create(self, target: SessionCommandTarget, payload: dict) -> None:
        """创建子分支：store 从 archive 构建拐点视图（System + 前 point 轮真实消息）"""
        point = payload.get("point")
        if not isinstance(point, int) or point < 1:
            target.notice("用法：branch_create 需 point>=1（对话消息序号）")
            return
        if target.branch_creator is None:
            target.notice("无上下文存储，无法创建分支")
            return
        try:
            new_branch, point_view = await target.branch_creator(target.branch_id, point)
        except ValueError as exc:
            target.notice(str(exc))
            return
        target.messages = point_view
        target.branch_id = new_branch
        target.notice(f"已创建分支 {new_branch}（拐点 seq={point}），后续输入进入该分支")

    async def _do_branch_switch(self, target: SessionCommandTarget, payload: dict) -> None:
        branch_id = payload.get("branch_id")
        if not branch_id:
            target.notice("用法：branch_switch 需 branch_id")
            return
        if target.branch_loader is None:
            target.notice("无上下文存储，无法切换分支")
            return
        try:
            restored = await target.branch_loader(branch_id)
        except ValueError as exc:
            target.notice(str(exc))
            return
        if restored is None:
            target.notice(f"分支 {branch_id} 不存在或不属于本会话")
            return
        target.branch_id = branch_id
        target.messages = restored
        target.notice(f"已切换到分支 {branch_id}（{len(restored) - 1} 条对话）")

    async def _do_branch_delete(self, target: SessionCommandTarget, payload: dict) -> None:
        """删除分支（级联软删子树）。删 root=删整个会话；删当前分支则切回根分支"""
        branch_id = payload.get("branch_id")
        if not branch_id:
            target.notice("用法：branch_delete 需 branch_id（root=删除整个会话）")
            return
        if target.branch_deleter is None:
            target.notice("无上下文存储，无法删除分支")
            return
        try:
            deleted = await target.branch_deleter(branch_id)
        except ValueError as exc:
            target.notice(str(exc))
            return
        target.notice(f"已删除分支 {branch_id}（含 {len(deleted) - 1} 个后代分支）")
        if target.branch_id not in deleted:
            return
        if branch_id == "root" or "root" in deleted:
            target.notice("根分支（整个会话）已删除，本次会话结束")
            target.emit(EventType.SESSION_DELETED, payload={"branch_id": branch_id})
        else:
            target.branch_id = "root"
            if target.branch_loader is None:
                target.messages = [target.messages[0]] if target.messages else []
                target.notice("当前分支已删除，已切回根分支（无上下文存储，重置为新会话）")
                return
            try:
                restored = await target.branch_loader("root")
                target.messages = restored if restored else ([target.messages[0]] if target.messages else [])
                target.notice("当前分支已删除，已切回根分支")
            except Exception:
                target.messages = [target.messages[0]] if target.messages else []
                target.notice("当前分支已删除，已切回根分支（视图重建失败，重置为新会话）")

    async def _do_rename(self, target: SessionCommandTarget, payload: dict) -> None:
        """重命名：当前分支为 root → 会话名；否则分支名"""
        name = payload.get("name")
        if not name or not name.strip():
            target.notice("用法：rename 需非空名字")
            return
        name = name.strip()
        if target.branch_id == "root":
            if target.session_renamer is None:
                target.notice("无上下文存储，无法重命名会话")
                return
            await target.session_renamer(name)
            target.notice(f"会话已重命名为「{name}」")
        else:
            if target.branch_renamer is None:
                target.notice("无上下文存储，无法重命名分支")
                return
            await target.branch_renamer(target.branch_id, name)
            target.notice(f"分支 {target.branch_id} 已重命名为「{name}」")

    async def _do_update_compressor(self, target: SessionCommandTarget, payload: dict) -> None:
        """更新上下文压缩器配置：保留轮数下限 + 保留预算（token）"""
        keep_recent_turns = payload.get("keep_recent_turns")
        retention_budget = payload.get("retention_budget")

        if not isinstance(keep_recent_turns, int) or keep_recent_turns < 1:
            target.notice("keep_recent_turns 必须是 ≥1 的整数")
            return
        if not isinstance(retention_budget, int) or retention_budget < 1:
            target.notice("retention_budget 必须是 ≥1 的整数（token）")
            return
        if keep_recent_turns * 64 > retention_budget:
            target.notice("保留轮数对应的 token 估算不能超过 retention_budget")

        if target.compressor is None:
            target.notice("压缩器实例未初始化，无法更新配置")
            return

        target.compressor.update_config(keep_recent_turns=keep_recent_turns, retention_budget=retention_budget)
        target.notice(f"压缩器配置已更新：keep_recent_turns={keep_recent_turns}, retention_budget={retention_budget}")
