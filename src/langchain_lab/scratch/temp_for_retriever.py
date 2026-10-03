import json


def temp():
    # test_simple_retriever()
    # test_retriever()
    # test_save_retriever()
    # test_add_retriever()
    test_final_retriever()
    pass


def test_simple_retriever():
    from langchain_lab.rag.retriever import SimpleRetriever

    # 模拟知识库（实际中这里应该是从文档加载并切分好的）
    documents = [
        "养老金申领需要达到法定退休年龄",
        "申领时需要连续缴费满15年",
        "社保转移可以通过当地人社局线上办理",
        "医保报销需要保留原始发票和处方单",
        "生育津贴需要在产后90天内申请",
    ]

    # 初始化检索器（这一步会向量化所有文档）
    retriever = SimpleRetriever(documents, top_k=3)

    # 测试检索
    query = "我缴费多少年才能领养老金？"
    print("===检索结果===")
    result = retriever.retrieve(query)
    if not result:
        print("抱歉，知识库中暂无相关信息")
    else:
        for score, doc in result:
            print(f"相似度分数：{score:.4f}；内容：{doc}")

    # 测试拼装 Prompt
    print("\n=== 生成的 Prompt ===")
    print(retriever.build_prompt(query))


def test_retriever():
    from langchain_lab.rag.retriever import Document, Retriever

    # 模拟知识库（实际中这里应该是从文档加载并切分好的）
    documents = [
        Document(
            id="doc_001",
            text="养老金申领需要达到法定退休年龄",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="申领时需要连续缴费满15年",  # 不传 id，自动生成
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="社保转移可以通过当地人社局线上办理",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="养老保险缴费年限越长，退休后领取的养老金越高",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="医保报销需要保留原始发票和处方单",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="生育津贴需要在产后90天内申请",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
    ]

    # 初始化检索器（这一步会向量化所有文档）
    retriever = Retriever(documents, top_k=3)

    # 测试检索
    query = "我缴费多少年才能领养老金？"
    print("===检索结果===")
    result = retriever.retrieve(query)
    if not result:
        print("抱歉，知识库中暂无相关信息")
    else:
        for score, doc in result:
            print(f"相似度分数：{score:.4f}；内容：{doc.text}")

    # 测试拼装 Prompt
    print("\n=== 生成的 Prompt ===")
    print(retriever.build_content(query))


def test_save_retriever():
    from langchain_lab.rag.retriever import Document, Retriever, serialize_ret_result

    documents = [
        Document(
            id="doc_001",
            text="养老金申领需要达到法定退休年龄",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="申领时需要连续缴费满15年",  # 不传 id，自动生成
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="社保转移可以通过当地人社局线上办理",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="养老保险缴费年限越长，退休后领取的养老金越高",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="医保报销需要保留原始发票和处方单",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="生育津贴需要在产后90天内申请",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
    ]

    # 第一次：构建并保存
    retriever = Retriever(documents)
    retriever.save("kb_index")

    # 第二次：加载
    retriever2 = Retriever.load("kb_index")
    res = retriever2.retrieve("我缴费多少年才能领养老金？")
    print(json.dumps(serialize_ret_result(res), ensure_ascii=False, indent=2))


def test_add_retriever():
    from langchain_lab.rag.retriever import Document, Retriever

    # 构建知识库
    documents = [
        Document(
            id="doc_001",
            text="养老金申领需要达到法定退休年龄",
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
        Document(
            text="申领时需要连续缴费满15年",  # 不传 id，自动生成
            metadata={"source": "社保政策.md", "topic": "养老"},
        ),
    ]

    retriever = Retriever(documents)
    retriever.save("kb_index")

    # 加载
    retriever = Retriever.load("kb_index")

    # 增量添加
    retriever.add_documents(
        [
            Document(
                id="doc_003",
                text="社保转移可以通过当地人社局线上办理",
                metadata={"source": "办事指南.md"},
            ),
        ]
    )
    retriever.save("kb_index")

    # 检索
    results = retriever.retrieve("我缴费多少年才能领养老金？")
    for score, doc in results:
        print(f"{score:.4f}  {doc.id}  {doc.metadata}  {doc.text}")


def test_final_retriever():
    from langchain_lab.rag.loader import FileLoader
    from langchain_lab.rag.retriever import Retriever
    from langchain_lab.rag.splitter import TextSplitter
    from langchain_lab.util.get_path import get_folder

    # 1. 加载文件
    print("======选中要检索的文件夹======")
    note_folder = get_folder()
    kb_folder = "kb_index"

    loader = FileLoader(note_folder)  # 笔记根目录
    raw_docs = loader.load_directory(note_folder)  # 批量加载所有 md/txt

    # 2. 切分
    splitter = TextSplitter()
    all_chunks = []
    for raw_doc in raw_docs:
        chunks = splitter.split_raw_document(raw_doc)
        all_chunks.extend(chunks)

    print(f"共生成 {len(all_chunks)} 个 chunk")
    for chunk in all_chunks[:3]:
        print(chunk.metadata, chunk.text[:200])

    # 3. 构建索引
    retriever = Retriever(all_chunks)
    retriever.save(kb_folder)

    # ========== 前：构建入库；后：载入检索 ===========
    kb_folder = "kb_index"
    retriever = Retriever.load(kb_folder)

    # 4. 检索
    results = retriever.retrieve("Python的模块的导入与加载具有什么机制？")

    input_value = ""
    while True:
        for score, doc in results:
            print("-" * 100 + "\n" + f"得分：{score:.4f}  文件来源：[{doc.metadata.get('source')}] \n文件内容：{doc.text[:500]}")
        input_value = input("\n请输入下个问题（输入'quit'退出）> ")
        if input_value == "quit":
            break
        results = retriever.retrieve(input_value)

    print("=" * 10 + "用户选择退出" + "=" * 10)


# ========== 脚本入口，路由执行 ==========
# region
import inspect
import sys

import questionary


def main():
    # 动态获取路由函数，也可以传 prefix="" 获取所有公开函数
    routes = get_current_functions(prefix="test_")

    if not routes:
        print("未找到任何符合条件的函数")
        return

    # 使用 questionary 动态呈现函数名列表
    choice = questionary.select("请选择要执行的路由函数：", choices=list(routes.keys())).ask()

    if not choice:
        return

    # 确认执行
    if questionary.confirm(f"确认执行 {choice} 吗？").ask():
        routes[choice]()
    else:
        print("已取消")


def get_current_functions(prefix="route_"):
    """
    获取当前模块中定义且名称符合指定前缀的函数。
    prefix 为空字符串时，获取所有当前模块定义的非下划线开头函数。
    """
    current_module = sys.modules[__name__]
    funcs = {}

    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        # 只保留定义在当前模块内的函数，排除 import 进来的
        if func.__module__ == __name__ and ((prefix and name.startswith(prefix)) or (not prefix and not name.startswith("_"))):
            funcs[name] = func

    return funcs


if __name__ == "__main__":
    main()
# endregion
