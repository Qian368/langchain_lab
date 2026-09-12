"""模型组件：初始化 LLM 实例"""

import logging
import os
import warnings

from dotenv import load_dotenv
from jwt.warnings import InsecureKeyLengthWarning
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()
warnings.filterwarnings("ignore", category=InsecureKeyLengthWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelNotFoundError(ValueError):
    """模型不存在，继承ValueError，语义上属于值错误"""

    pass


def get_llm(name: str = "ChatZhipuAI"):
    """返回配置好的模型实例"""
    if name == "ChatZhipuAI":
        return ChatZhipuAI(
            model="glm-5.2",
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            temperature=0.7,
            max_tokens=3000,
        )

    llm_list: list[str] = ["ChatZhipuAI"]

    raise ModelNotFoundError(f"模型不存在，现有可用模型有：{llm_list}")
