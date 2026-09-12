"""工具组件：定义可复用工具"""

from langchain_core.tools import tool


@tool
def add(a: int, b: int):
    """计算两个数的和"""
    return a + b


@tool
def multiply(a: int, b: int):
    """计算两个数的积"""
    return a * b


def get_tools():
    """返回工具列表"""
    return [add, multiply]
