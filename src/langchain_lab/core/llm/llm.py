"""模型组件：初始化 LLM 实例"""

import warnings
from functools import lru_cache

import questionary
from colorama import Fore, Style, init
from jwt.warnings import InsecureKeyLengthWarning
from langchain.chat_models import init_chat_model

from langchain_lab import __project_name__
from langchain_lab.core.llm.llm_schema import LLMConfig, LLMInstanceConfig, load_llm_config

warnings.filterwarnings("ignore", category=InsecureKeyLengthWarning)
init(autoreset=True)

from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.CORE)


class ModelNotFoundError(ValueError):
    """模型不存在，继承ValueError，语义上属于值错误"""


class ModelConfigError(RuntimeError):
    """模型配置缺失或不完整"""


@lru_cache
def _get_config() -> LLMConfig:
    """缓存配置，避免每次请求都读 YAML"""
    return load_llm_config()


def build_llm(cfg: LLMInstanceConfig):

    # 通用参数直接传给 init_chat_model
    kwargs = {
        "model": cfg.model,
        "model_provider": cfg.provider,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "api_key": cfg.api_key or None,
        "base_url": cfg.base_url or None,
    }

    # 这些参数有的 provider 支持有的不支持，先只在显式配置时尝试
    # 如果不需要超时/重试/额外头，可以不传，保持最简
    if cfg.timeout is not None:
        kwargs["timeout"] = cfg.timeout
    if cfg.max_retries is not None:
        kwargs["max_retries"] = cfg.max_retries
    if cfg.extra_headers:
        kwargs["extra_headers"] = cfg.extra_headers
    try:
        return init_chat_model(**kwargs)
    except (ImportError, ModuleNotFoundError) as e:
        raise ModelConfigError(
            f"使用 {cfg.provider} 需要安装依赖：uv add {__project_name__}[{cfg.provider}] 或 pip install {__project_name__}[{cfg.provider}]"
        ) from e
    except ValueError as e:
        raise ModelConfigError(f"不支持的模型提供商：{cfg.provider}，请检查 YAML 配置，详情：\n{e}") from e


def get_llm(name: str | None = None):
    """获取 LLM 实例。

    - name 显式指定：直接用，但该实例必须可用，否则报错
    - name 为空：依次走 default_instance → 自动选择 → questionary 选择
    """
    config = _get_config()
    available = config.available_instances()

    print(f"{Fore.BLUE}LLM 信息（不测试连通性，配置满足要求即可用）👇：")
    print(f"当前可用实例👉：{Fore.GREEN}{list(available)}{Style.RESET_ALL} 详情👇：")
    # 打印某个实例的 api_key 是否为空
    for llm, cfg in available.items():
        print(f"{Fore.GREEN}{llm}:  model={cfg.model}  api_key={'***' if cfg.api_key else 'EMPTY'},  base_url={cfg.base_url}")

    # 1. 显示指定
    if name:
        if name not in available:
            raise ValueError(f"实例 '{name}' 不存在，可选实例：{list(available)}")
        return build_llm(available[name])

    # 2. 默认实例
    if config.default_instance:
        return build_llm(available[config.default_instance])

    # 3. 只有一个可用实例，自动使用
    if len(available) == 1:
        name = next(iter(available))
        logger.info("使用唯一可用模型：%s", name)
        return build_llm(available[name])

    # 多个可用且没有默认 → questionary 选择
    choices = [questionary.Choice(title=f"{name} ({cfg.provider}：{cfg.model})", value=name) for name, cfg in available.items()]
    selected = questionary.select("检测到多个可用的模型，请选择：", choices=choices).ask()
    return build_llm(available[selected])


if __name__ == "__main__":
    get_llm()

    # import inspect
    # print(inspect.signature(init_chat_model))

    # import os
    # from dotenv import load_dotenv
    # load_dotenv()

    # from langchain.chat_models import init_chat_model

    # # 根据你实际配置填写
    # llm = init_chat_model(
    #     model="glm-5.2",                    # 换成你 YAML 里的 model
    #     model_provider="openai",            # 如果你用的是 OpenAI 兼容
    #     api_key=os.getenv("ZHIPUAI_API_KEY"),
    #     base_url="https://open.bigmodel.cn/api/paas/v4/",
    #     temperature=0.7,
    #     max_tokens=2048,
    #     # 不传 timeout 等高级参数
    # )

    # try:
    #     resp = llm.invoke("你好，请回复OK")
    #     print(resp.content)
    # except Exception as e:
    #     print(f"调用失败: {type(e).__name__}: {e}")
