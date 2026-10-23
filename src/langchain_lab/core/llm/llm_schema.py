"""LLM 多实例配置 Schema（并入统一全局配置中心）

- LLMConfig / LLMInstanceConfig 继承 EventedModel：字段变更发事件，支持热重载订阅
- ${VAR} 环境变量占位在 model_validator 展开（api_key 不落明文，.env + 占位）
- 配置加载统一走 core/config/loader.py 的 ConfigLoader，get_config().llm 全局导入
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from pydantic import Field, model_validator

from langchain_lab.core.config.evented_model import EventedModel

load_dotenv()

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|%([^%]+)%")


def _expand_vars(value: str) -> str:
    """展开 ${VAR} / %VAR% 环境变量：存在则替换，否则留空"""

    def _replace(match: re.Match):
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, "")

    return _ENV_PATTERN.sub(_replace, value)


class LLMInstanceConfig(EventedModel):
    """单个 LLM 实例配置"""

    provider: str | None = Field(default=None, min_length=1, description="模型提供商，如 deepseek / zhipuai / openai / ollama")
    model: str | None = Field(default=None, min_length=1, description="模型名称")
    api_key: str | None = Field(default=None, description="API 密钥（yaml 用 ${VAR} 占位，.env 注入）")
    base_url: str | None = Field(default=None, description="API 基础地址，不填用 SDK 默认")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=2048, gt=0, description="最大输出 token 数")
    timeout: float | None = Field(default=None, gt=0, description="请求超时（秒）")
    max_retries: int | None = Field(default=None, ge=0, description="最大重试次数")
    extra_headers: dict[str, str] | None = Field(default=None, description="额外请求头")


class LLMConfig(EventedModel):
    """LLM 多实例配置根对象"""

    default_instance: str | None = Field(default=None, description="默认模型实例")
    instances: dict[str, LLMInstanceConfig] = Field(
        default_factory=lambda: {"default": LLMInstanceConfig()}, description="LLM 实例字典"
    )

    @model_validator(mode="after")
    def expand_env(self):
        """统一加载后展开 ${VAR}：yaml 占位 → .env 实际值（api_key 等不落明文）"""
        for inst in self.instances.values():
            for field in ("provider", "model", "api_key", "base_url"):
                value = getattr(inst, field)
                if isinstance(value, str):
                    # 加载期赋值，绕过事件发射（_initialized 阶段不发事件）
                    object.__setattr__(inst, field, _expand_vars(value))
        return self

    def available_instances(self) -> dict[str, LLMInstanceConfig]:
        """返回当前环境变量下真正可用的实例（有 key 或本地模型）"""
        return {
            name: cfg
            for name, cfg in self.instances.items()
            if (cfg.provider and cfg.model and cfg.base_url)
            and (cfg.api_key or cfg.provider in ("ollama", "llamacpp"))
        }

    @model_validator(mode="after")
    def validate_available(self):
        if not self.instances:
            raise ValueError("配置中未配置任何模型实例")
        available = self.available_instances()
        if not available:
            raise ValueError("没有任何模型可用：请至少为某个实例提供 api_key（检查 .env 环境变量是否配置）")
        if self.default_instance:
            if self.default_instance not in self.instances:
                raise ValueError(f"默认实例 '{self.default_instance}' 不在配置中")
            if self.default_instance not in available:
                print("默认实例情况如下：\n")
                default = self.instances.get(self.default_instance)
                if default is not None:
                    data = default.model_dump().copy()
                    data["api_key"] = "***" if data.get("api_key") else "EMPTY"
                    print(f"{json.dumps(data, ensure_ascii=False, indent=2)}")
                raise ValueError(
                    f"默认实例 '{self.default_instance}' 不可用，请检查 provider, model, base_url, api_key 是否正确配置"
                )
        return self
