def temp():
    # print_attr()
    # print_test1()
    # print_test_2()
    # print_test_3()
    # dowmload_model()
    pass

if __name__ == "__main__":
    temp()


def print_attr():
    import json

    from langchain_core.messages import SystemMessage

    attr_name = dir(SystemMessage)
    # filtered = [n for n in attr_name if not (n.startswith("__") and n.endswith("__"))]
    print(
        json.dumps(
            attr_name,
            indent=2,
            ensure_ascii=False,
        )
    )


def print_test_1():
    print("\n\n" + "=" * 38 + '\033[1;34m 开始聊天 or 输入"0" 退出 \033[0m' + "=" * 40)


def print_test_2():
    import sys

    import colorama

    colorama.init(autoreset=True)
    print(type(sys.stdout))  # 看看实际类型
    print(hasattr(sys.stdout, "autoreset"))  # 看看有没有这个属性
    # print(sys.stdout.autoreset)  # 打印值


def print_test_3():
    from langchain_lab.util.prompt_style import Printer

    printer = Printer()
    printer.yellow(
        "代发是你们dfsae打发哦您呢斯蒂芬妮那倒是fdsafsafsadf地方萨芬是的\n是的防守打法vv代发突然好个人v成本结合处理了前景色。背景色如果要亮色（100–107），需单独扩展映射，目前你的标签区背景色暂不\n支处理了前景色。背景色如果要亮色（100–107），需单独扩展映射，目前你的标签区背景色暂不支"
    )
    printer.yellow(
        "代发是你们dfsae打发哦您呢斯蒂芬妮那倒是合同我让他和我通\n融通融好突然我忽然问哈让他处理了前景色。背景色如果要亮色（100–107），需单独扩展映射，目前你的标签区背景色暂不支处理了\n前景色。背景色如果要亮色（100–107），需单独扩展映射，目前你的标签区背景色暂不支",
        bright=True,
    )


def dowmload_model():
    import os

    from huggingface_hub import snapshot_download

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内加速，可留可删

    snapshot_download(
        repo_id="BAAI/bge-small-zh-v1.5",
        local_dir="../models/bge-small-zh-v1.5",
    )

    print("done")

