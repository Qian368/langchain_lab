def test_retriever():
    from langchain_lab.rag.retriever import SimpleRetriever

    # 模拟知识库（实际中这里应该是从文档加载并切分好的）
    top_k: int = 3
    documents = [
        "养老金申领需要达到法定退休年龄",
        "申领时需要连续缴费满15年",
        "社保转移可以通过当地人社局线上办理",
        "养老保险缴费年限越长，退休后领取的养老金越高",
        "医保报销需要保留原始发票和处方单",
        "生育津贴需要在产后90天内申请",
    ]

    # 初始化检索器（这一步会向量化所有文档）
    retriever = SimpleRetriever(documents, top_k)

    query = "我缴费多少年才能领养老金？"
    print("===测试检索结果===")
    result = retriever.retrieve(query)
    for score, doc in result:
        assert score > 0.7
        assert doc in documents
    assert len(result) <= top_k

    print("\n=== 测试拼装 Prompt ===")
    assert "养老金" in retriever.build_prompt(query)
