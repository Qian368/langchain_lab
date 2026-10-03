from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class LLMInstanceConfig(BaseModel):
    """单个 LLM 实例配置"""

    provider: str | None = Field(default=None, min_length=1, description="模型提供商，如 deepseek / zhipuai / openai / ollama")
    model: str | None = Field(default=None, min_length=1, description="模型名称")
    api_key: str | None = Field(default=None, description="API 密钥")
    base_url: str | None = Field(default=None, description="API 基础地址，不填用 SDK 默认")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=2048, gt=0, description="最大输出 token 数")
    timeout: float | None = Field(default=None, gt=0, description="请求超时（秒）")
    max_retries: int | None = Field(default=None, ge=0, description="最大重试次数")
    extra_headers: dict[str, str] | None = Field(default=None, description="额外请求头")


class LLMConfig(BaseModel):
    """LLM 多实例配置跟对象"""

    default_instance: str | None = Field(default=None, description="默认模型实例")
    instances: dict[str, LLMInstanceConfig] = Field(default_factory=lambda: {"default": LLMInstanceConfig()}, description="LLM 实例字典")

    def available_instances(self) -> dict[str, LLMInstanceConfig]:
        """
        返回当前环境变量下真正可用的实例
            - 有填 key 或者本地模型
        """
        return {
            name: cfg
            for name, cfg in self.instances.items()
            if (cfg.provider and cfg.model and cfg.base_url) and (cfg.api_key or cfg.provider in ("ollama", "llamacpp"))  # 本地模型不需要 key
        }

    @model_validator(mode="after")
    def validate_available(self):
        if not self.instances:
            raise ValueError("YAML 中未配置任何模型实例")

        available = self.available_instances()
        if not available:
            raise ValueError("没有任何模型可用：请至少为某个实例提供 api_key（检查 .env 环境变量是否配置）")

        # 如果显式指定 default_instance，必须是可用的
        if self.default_instance:
            if self.default_instance not in self.instances:
                raise ValueError(f"默认实例 '{self.default_instance}' 不在 YAML 文件配置中")
            if self.default_instance not in available:
                print("默认实例情况如下：\n")
                default = self.instances.get(self.default_instance)
                if default is not None:
                    data = default.model_dump().copy()
                    data["api_key"] = "***" if data.get("api_key") else "EMPTY"
                    print(f"{json.dumps(data, ensure_ascii=False, indent=2)}")
                raise ValueError(f"默认实例 '{self.default_instance}' 不可用，请检查 proveder, model, base_url, api_key, 是否正确配置")
        return self


def load_llm_config(path: str | Path = "config/llm.yaml") -> LLMConfig:
    """读取 YAML, 展开环境变量，并用 Pydantic 校验"""
    import re

    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # 读成 Python 数据对象

    _PATTERN = re.compile(r"\$\{([^}]+)\}|%([^%]+)%")

    def expand_vars(value: Any) -> Any:
        """递归展开 ${VAR} 环境变量"""
        if isinstance(value, dict):
            return {k: expand_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand_vars(v) for v in value]
        if isinstance(value, str):

            def _replace(match: re.Match):  # 用环境变量替换 ${VAR} 或 %VAR% 的字符串：有则替换，无则返回原字符串
                var_name = match.group(1) or match.group(2)  # 从value中捕获到的 匹配内容：${XXX} 或 %XXX%，里面的XXX
                return os.getenv(var_name, "")  # 变量不存在 → 返回空字符串；存在就返回变量值

            return _PATTERN.sub(_replace, value)  # 匹配器匹配替换：_replace 返回结果替换捕获的 匹配内容
        return value

    data = expand_vars(raw)
    config = LLMConfig.model_validate(data)

    return config
