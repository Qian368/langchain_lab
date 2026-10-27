"""上下文管理配置 Schema（产品策略 + 压缩执行，独立一域）

设计：与模型窗口/运行时解耦
- 模型窗口 context_window 在 config/llm.yaml（模型能力，跟模型走，换模型只改它）
- 上下文策略（保留轮数下限 / 预留比例）与压缩执行（开关 / 摘要模型 / 摘要指令）在此独立演进，
  调上下文管理不动运行时/模型声明
- retention_budget = 模型窗口 × (1 - headroom_ratio)：压缩后保留最近轮的 token 预算，
  也是触发自动压缩的阈值（实际 token 超此即压）
"""
from pydantic import Field

from langchain_lab.core.config.evented_model import EventedModel


class CompressionSettings(EventedModel):
    """上下文压缩【执行】配置（压缩作为上下文管理能力；保留策略在 ContextSettings）"""

    enabled: bool = Field(default=True, description="是否启用上下文压缩")
    summarize_model: str | None = Field(default="default", description="压缩摘要用的 LLM 实例名")
    summarize_prompt: str | None = Field(
        default=None,
        description="压缩指令文本（HumanMessage 追加在压缩请求末尾；None 用内置默认）",
    )


class ContextSettings(EventedModel):
    """上下文管理配置（产品策略 + 压缩执行）"""

    keep_recent_turns: int = Field(
        default=6, ge=1, description="压缩后保留的最近轮数（下限）"
    )
    headroom_ratio: float = Field(
        default=0.3,
        gt=0,
        lt=1,
        description="预留比例（生成余量+system+摘要占模型窗口）；保留预算 = 窗口 × (1 - 比例)",
    )
    compression: CompressionSettings = Field(default_factory=CompressionSettings)

    def retention_budget(self, context_window: int) -> int:
        """保留预算 = 模型窗口 × (1 - headroom_ratio)。

        压缩后保留最近轮 ≤ 此值；实际 token 超此触发自动压缩（LLM 视图只保留最近 keep 轮 + 摘要置顶）。
        """
        return round(context_window * (1 - self.headroom_ratio))
