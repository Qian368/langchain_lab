"""实验4：LCEL 管道"""

"""
如果链是 prompt | chat_model，其中 chat_model 是类似 ChatOpenAI 的聊天模型，那么 stream() 返回的是 AIMessageChunk，你需要用 chunk.content 取文本。
如果链是 prompt | chat_model | StrOutputParser()，那么 StrOutputParser 会将 AIMessageChunk 中的 content 提取出来直接输出，所以外面拿到的就是 str。
"""

from langchain_lab.core.pipeline import get_basic_chain
from langchain_lab.util.prompt_style import Printer

printer = Printer()

# region 不推荐：硬编码方案
# class ChainInput(TypedDict):
#     role: str
#     style: str
#     question: str
# endregion


def build_chain_input(role: str, style: str, question: str) -> dict:
    return {
        "role": role,
        "style": style,
        "question": question,
    }


def main():

    print("\n --- 实验4：LCEL 管道---")
    chain = get_basic_chain()

    # 1. 同步调用
    printer.magenta("\n[1] 同步调用")
    # region 不推荐：硬编码方案
    # chain_input: ChainInput = {
    #     "role": "运维助手",
    #     "style": "简洁",
    #     "question": "CPU 100%怎么办？",
    # }
    # result = chain.invoke(cast(dict, chain_input))
    # endregion
    question = printer.prompt("请输入你的问题> ", printer.style_name.BLUE) or "写一首简短的关于秋天的诗"
    chain_input = build_chain_input(
        role="诗人",
        style="优美",
        question=question,
    )

    result = chain.invoke(chain_input)
    printer.magenta(f" llm 回复如下：👇👇👇 \n{result}", bright=True)

    # 2. 流式调用
    question = printer.prompt("请输入你的问题> ", printer.style_name.BLUE) or "CPU 100%怎么办？"
    chain_input = build_chain_input(
        role="运维助手",
        style="简洁",
        question=question,
    )
    with printer.stream_style(printer.style_name.MAGENTA):
        for chunk in chain.stream(chain_input):
            # print(chunk, end="", flush=True) 都OK
            printer.stream_chunk(chunk)
