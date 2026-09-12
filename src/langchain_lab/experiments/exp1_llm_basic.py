"""实验1：模型的三种调用方式"""

from langchain.messages import HumanMessage
from langchain_core.language_models.base import LanguageModelInput
from prompt_toolkit import prompt

from langchain_lab.core.llm import get_llm


def main():
    print("\n---实验1：LLM 基础调用")

    llm = get_llm()

    # 1. invoke：同步调用
    print("\n[1] invoke 同步调用")
    human_prompt = prompt("请输入你的问题：").strip() or "用一句话解释什么是RAG"
    response = llm.invoke(f"{human_prompt}")
    print(f"类型：{type(response).__name__}")
    print(f"内容：{response.content}")

    # 2. stream：流式调用
    print("\n[2] stream 流式调用")
    human_prompt = prompt("请输入你的问题：").strip() or "写一首关于秋天的诗"
    for chunk in llm.stream(f"{human_prompt}"):
        print(chunk.content, end="", flush=True)
    print()

    # 3. batch：批量调用
    print("\n[3] batch 批量调用")
    q1 = prompt("请输入你的问题1：").strip() or "1+1等于几"
    q2 = prompt("请输入你的问题2：").strip() or "2+2等于几"
    q3 = prompt("请输入你的问题3：").strip() or "3+3等于几"

    # 简单正常推荐这种方式，运行时自动转为 HumanMessage，简单靠谱
    # result = llm.batch([q1, q2, q3])

    # 需要自定义 消息列表时采用一下方案，每个 batch 元素包成消息列表 [HumanMessage()]
    # 可以自由混入 SystemMessage 做多轮对话
    batch_input: list[LanguageModelInput] = [
        [HumanMessage(content=q1)],
        [HumanMessage(content=q2)],
        [HumanMessage(content=q3)],
    ]

    result = llm.batch(batch_input)
    for i, r in enumerate(result, 1):
        print(f"问题{i}：{r.content}")
