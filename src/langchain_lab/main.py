"""LangChain 实验台入口"""

import importlib

from prompt_toolkit import prompt


def show_menu():
    print("\n" + "=" * 50)
    print("LangChain 实验台")
    print("=" * 50)
    print("1. LLM 基础调用（invoke / stream / batch）")
    print("2. 工具调用 + 流式输出 + 最小循环")
    print("3. 提示词模板")
    print("4. LCEL 管道")
    print("0. 退出")
    print("=" * 50)


def run_experiment(module_name: str):
    """动态导入并运行实验模块"""
    try:
        module = importlib.import_module(f"langchain_lab.experiments.{module_name}")
        module.main()
    except Exception as e:
        print(f"实验运行失败：{e}")


def main():
    experiments = {
        "1": "exp1_llm_basic",
        "2": "exp2_llm_tools",
        "3": "exp3_prompt_template",
        "4": "exp4_lcel_pipeline",
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
