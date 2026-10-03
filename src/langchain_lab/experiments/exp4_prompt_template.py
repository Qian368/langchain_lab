"""实验3：提示词模版"""

"""提示词模版也是可以 invoke 的类型，模版列表留空，传入字典填空"""

from langchain_lab.core.llm.llm import get_llm
from langchain_lab.core.messages import get_basic_prompt, get_multi_turn_prompt
from langchain_lab.util.prompt_style import Printer

printer = Printer()


def main():
    print("\n---实验3：提示词模版---")

    llm = get_llm()

    # 1. 基础模版
    print("\n [1] 基础模版")
    prompt = get_basic_prompt()
    messages = prompt.invoke(
        {
            "role": "运维助手",
            "style": "简洁",
            "question": "CPU 100% 怎么办？",
        }
    )
    printer.green(f"生成的消息列表：\n{messages.model_dump_json(ensure_ascii=False, indent=2)}")
    response = llm.invoke(messages)
    printer.magenta(f"模型回复{response.content}")

    # 2.多伦模版
    print("\n[2] 多轮模版")
    prompt = get_multi_turn_prompt()
    messages = prompt.invoke(
        {
            "role": "助手",
            "question": "什么是Langchain",
        }
    )
    response = llm.invoke(messages)
    printer.magenta(f"模型回复{response.content}")
