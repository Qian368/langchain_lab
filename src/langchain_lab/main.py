"""LangChain 实验台入口"""

import asyncio
import importlib
import inspect

from prompt_toolkit import prompt


def show_menu():
    print("\n" + "=" * 50)
    print("LangChain 实验台")
    print("=" * 50)
    print("实验1： LLM 基础调用（invoke / stream / batch）")
    print("实验2： 单轮对话（single——chat）：工具调用 + 流式输出 + 最小循环")
    print("实验3： 多轮对话（single——chat）：工具调用 + 流式输出 + 最小循环 + 美化终端交互")
    print("实验4： 提示词模板")
    print("实验5： LCEL 管道")
    print("实验6： 构建本地文本向量仓库，完事就可以进入实验3 进行多轮对话啦")
    print("0. 退出")
    print("=" * 50)


def run_experiment(module_name: str):
    """动态导入并运行实验模块，自动兼容同步/异步main"""
    try:
        module_path = f"langchain_lab.experiments.{module_name}"
        module = importlib.import_module(module_path)

        module = importlib.reload(module)  # 强制重新加载模块，重读磁盘源码
        main_func = module.main  # 不能再 reload 前取出main，否则， reload 后不会刷新 main

        # 判断是否为协程函数
        if inspect.iscoroutinefunction(main_func):
            asyncio.run(main_func())  # 异步main，用asyncio.run启动
        else:
            main_func()  # 同步main，直接调用
    except Exception as e:
        print(f"实验运行失败：{e}")


def main():
    experiments = {
        "1": "exp1_llm_basic",
        "2": "exp2_single_chat",
        "3": "exp3_multi_chat",
        "4": "exp4_prompt_template",
        "5": "exp5_lcel_pipeline",
        "6": "exp6_base_retrieve",
    }

    while True:
        show_menu()

        choice = prompt("请选择实验编号：").strip()

        if choice == "0":
            print("退出实验台")
            break

        if choice in experiments:
            run_experiment(module_name=experiments[choice])
        else:
            print("无效编号，请重新输入")


if __name__ == "__main__":
    main()
