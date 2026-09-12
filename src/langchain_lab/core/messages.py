"""消息组件：定义提示词模版和消息列表构造"""

from langchain_core.prompts import ChatPromptTemplate


def get_basic_prompt():
    """基础模版：system + human"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}，回答要{style}"),
            ("human", "{question}"),
        ]
    )


def get_multi_turn_prompt():
    """多轮对话模版"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}"),
            ("human", "我的问题是{question}"),
            ("ai", "思考中..."),
            ("human", "请直接回答"),
        ]
    )


def custom_print():
    """小结：不要相信 __repr__ 的输出去猜对象结构，__repr__ 只是人工拼出来的好看文本，完全不代表语法可解析"""
    from prompt_toolkit import prompt

    role = prompt("请输入提示词模版，role：").strip()
    style = prompt("请输入提示词模版，style：").strip()
    question = prompt("请输入提示词模板，question：").strip()

    Template = get_basic_prompt()
    prompt_value = Template.invoke(
        {
            "role": role,
            "style": style,
            "question": question,
        }
    )
    # print(type(prompt_value))  # 1. 看真实类型
    # print(dir(prompt_value))  # 2. 看这个对象有哪些可用成员（属性、方法）

    messages = prompt_value.to_messages()
    print("生成的消息列表：")
    for msg in messages:
        print(f"{msg.pretty_repr}")  # 采用官方美化方法打印

    # IDE 并未提示 prompt_value 有 messages 属性，因为类型没有标注暴露给 Pylance，但可以这么取属性。
    # messages = prompt_value.messages
    # for msg in messages:
    #     print(f"{msg}")


if __name__ == "__main__":
    custom_print()
