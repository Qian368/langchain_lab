from __future__ import annotations

from pydantic import BaseModel, Field


# ============ 子配置模型 ============
class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1, le=65536)


class MCPServerConfig(BaseModel):
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    transport: str = "stdio"
