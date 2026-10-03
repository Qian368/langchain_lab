"""工具组件：定义可复用工具"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg, tool

# from langchain_core.runnables import RunnableConfig

load_dotenv()


def get_tools():
    """返回工具列表"""
    return [
        add,
        multiply,
        local_retriever,
        test1_query_order,
    ]


@tool
def add(a: int, b: int):
    """计算两个数的和"""
    return a + b


@tool
def multiply(a: int, b: int):
    """计算两个数的积"""
    return a * b


# 带状态示例工具：业务参数 + 隐式注入参数
# 推荐：拆分注入，内部直接用
@tool(parse_docstring=True)
def test1_query_order(
    order_no: str,
    user_id: Annotated[str, InjectedToolArg],
    tenant_id: Annotated[str, InjectedToolArg],
) -> str:
    """
    查询当前用户的订单详情

    Args:
        order_no: 订单编号，格式为ORD_开头加数字
    """
    # 业务逻辑直接用 user_id / tenant_id，框架自动注入
    return f"这只是个示例订单查询工具。\n订单{order_no}查询完成，用户：{user_id}"


# # 合并注入，内部解析
# @tool
# def test2_query_order(order_no: str, config: RunnableConfig) -> str:
#     """
#     查询当前用户的订单详情
#     Args:
#         order_no: 订单编号，格式为ORD_开头加数字
#     """
#     ctx = config.get("configurable", {})
#     user_id = ctx["user_id"]
#     tenant_id = ctx["tenant_id"]
#     return f"这只是个示例订单查询工具。\n订单{order_no}查询完成，用户：{user_id}"


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
