def temp():

    # print_dict()
    # print_attr()
    # print_test1()
    # print_test_2()
    # asyncio.run(test_printer())
    # dowmload_model()
    test_config()

    pass


if __name__ == "__main__":
    temp()


def print_dict():
    a = {}
    print(bool(a))


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


async def test_printer():
    from langchain_lab.util.prompt_style import Printer

    printer = Printer()
    printer.begin("代发是你们dfsae打发哦您呢斯蒂芬妮那倒是fdsafs")
    printer.yellow(
        "代发是你们dfsae打发哦您呢斯蒂芬妮那倒是合同我让他和我暂不支",
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


def test_config():
    from langchain_lab.core.config.config_core import get_config

    # 两次获取是同一个实例
    c1 = get_config()
    c2 = get_config()
    print(f"{c1 == c2}")
