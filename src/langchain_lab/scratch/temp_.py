import json

from langchain_core.messages import SystemMessage


def temp():
    # print_attr()
    print_test()


def print_attr():
    attr_name = dir(SystemMessage)
    # filtered = [n for n in attr_name if not (n.startswith("__") and n.endswith("__"))]
    print(
        json.dumps(
            attr_name,
            indent=2,
            ensure_ascii=False,
        )
    )


def print_test():
    print("\n\n" + "=" * 38 + '\033[1;34m 开始聊天 or 输入"0" 退出 \033[0m' + "=" * 40)
