"""实验6： 构建本地文本向量仓库，完事就可以进入实验3 进行多轮对话啦"""

import os

from dotenv import load_dotenv

from langchain_lab.rag.loader import FileLoader
from langchain_lab.rag.retriever import Retriever
from langchain_lab.rag.splitter import TextSplitter
from langchain_lab.util.get_path import get_folder

load_dotenv()


def main():
    construct_retriever()
    # base_rag()


def construct_retriever():
    """启动程序，构建本地知识库"""
    # 1. 加载文件
    print("======选中要检索的文件夹======")
    note_folder = get_folder()
    kb_folder = os.getenv("VECTORSTORE_PATH")

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
    if kb_folder is not None:
        retriever.save(kb_folder)
    else:
        print("无法保存到向量数据库，请在环境变量 .env 文件中添加向量数据库根地址")
    print("========== 已完成构建 ========\n\n")


# def base_rag():
#     """
#     - 已经将工具添加至 langchain_lab.core.tools
#     - 直接调用 实验3 的带有知识库查询的 agent 多轮对话
#     """
#     import asyncio
#     from langchain_lab.experiments.exp3_multi_chat import main
#     asyncio.run(main())
