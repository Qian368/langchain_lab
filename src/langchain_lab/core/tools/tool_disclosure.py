"""
工具渐进式披露策略层
- 职责：根据上下文决定当前轮次给 LLM 暴露哪些工具
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from langchain_lab.core.tools.base_tools import get_base_tools
from langchain_lab.mcp.client.mcp_tool_registry import _mcp_tool_registry, discover_mcp_tools


async def get_init_tools(session_ctx: dict) -> list[BaseTool]:
    """获取当前身份的完整工具集：基础工具 + 有权限的 MCP 工具"""
    base_tools = get_base_tools()
    mcp_tools = await _mcp_tool_registry.get_tools_for_identity(session_ctx)
    return base_tools + mcp_tools


async def combination_mcp_tools(mcp_id: str) -> list[BaseTool]:
    """跳过鉴权：预加载指定 MCP 工具到缓存，服务启动时可调用一次"""
    base_tools = get_base_tools()
    _mcp_tools_cache = await discover_mcp_tools(mcp_id)
    return base_tools + _mcp_tools_cache


# ========== 标签自动推断（方案一：按来源分组，零侵入） ==========
def _infer_tool_tag(tool: BaseTool) -> str:
    """
    根据工具名自动推断标签：
    - MCP 工具（命名格式 mcp_id__tool_name）→ 标签 = mcp_id
    - 其他（基础工具）→ 标签 = "base"
    """
    name = tool.name
    if "__" in name:
        # rag__upload_file → tag = "rag"
        return name.split("__", 1)[0]
    return "base"


# ========== 抽象基类 ==========
class BaseToolDisclosurePolicy(ABC):
    """
    工具披露策略抽象基类
    - 输入：全量工具列表 + 对话上下文
    - 输出：当前轮次可见的工具子集
    """

    @abstractmethod
    async def get_visible_tools(
        self,
        all_tools: list[BaseTool],
        context: dict[str, Any] | None = None,
    ) -> list[BaseTool]:
        """返回当前轮次可见的工具子集"""
        ...


# ========== 全量放行策略（默认，和现在行为完全一致） ==========
class FullDisclosurePolicy(BaseToolDisclosurePolicy):
    """不做任何过滤，全部暴露。作为默认策略，零破坏性。"""

    async def get_visible_tools(
        self,
        all_tools: list[BaseTool],
        context: dict[str, Any] | None = None,
    ) -> list[BaseTool]:
        return all_tools


# ========== 静态标签策略（按标签白名单过滤） ==========
class StaticTagPolicy(BaseToolDisclosurePolicy):
    """
    静态标签策略：只暴露指定标签集合内的工具。
    标签通过 infer_tool_tag() 自动推断，不需要工具定义时手动声明。

    示例：
        policy = StaticTagPolicy(enabled_tags={"rag", "base"})
        # 只暴露 rag MCP 的工具 + 基础工具，其他 MCP 工具全部隐藏
    """

    def __init__(self, enabled_tags: set[str]):
        self.enabled_tags = enabled_tags

    async def get_visible_tools(
        self,
        all_tools: list[BaseTool],
        context: dict[str, Any] | None = None,
    ) -> list[BaseTool]:
        return [tool for tool in all_tools if _infer_tool_tag(tool) in self.enabled_tags]


# ========== 反向黑名单策略（可选，直接排除指定标签） ==========
class ExcludeTagPolicy(BaseToolDisclosurePolicy):
    """
    排除指定标签的工具，其余全部暴露。
    适合"大部分都给，只藏掉几个敏感组"的场景。
    """

    def __init__(self, excluded_tags: set[str]):
        self.excluded_tags = excluded_tags

    async def get_visible_tools(
        self,
        all_tools: list[BaseTool],
        context: dict[str, Any] | None = None,
    ) -> list[BaseTool]:
        return [tool for tool in all_tools if _infer_tool_tag(tool) not in self.excluded_tags]


# 初始化策略（默认全量放行，以后换 StaticTagPolicy 就行）
disclosure_policy = FullDisclosurePolicy()


async def get_disclosure_tools(session_ctx: dict, messages: list[BaseMessage]) -> list[BaseTool]:
    """动态获取披露工具，适合 Agent 运行时动态更新"""
    all_tools = await get_init_tools(session_ctx)
    visible_tools = await disclosure_policy.get_visible_tools(
        all_tools,
        context={"session_ctx": session_ctx, "messages": messages},  # 上下文先留着，后面动态策略用得上
    )

    return visible_tools
