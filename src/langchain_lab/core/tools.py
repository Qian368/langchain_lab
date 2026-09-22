"""工具组件：定义可复用工具"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


def get_tools():
    """返回工具列表"""
    return [add, multiply, local_retriever]


@tool
def add(a: int, b: int):
    """计算两个数的和"""
    return a + b


@tool
def multiply(a: int, b: int):
    """计算两个数的积"""
    return a * b


@tool
def local_retriever(query: str) -> str:
    """
    查询本地知识库，返回 json 格式文本
    """
    print("加载本地知识库中...")
    from langchain_lab.rag.retriever import Retriever

    kb_folder = os.getenv("VECTORSTORE_PATH")
    if kb_folder is not None:
        retriever = Retriever.load(kb_folder)
    else:
        print("未指定向量数据库目录，无法加载本地向量数据库，请在环境变量 .env 文件中添加向量数据库根地址")
        return "本地知识库出现异常，暂时无法使用"
    return retriever.build_json_content(query)
