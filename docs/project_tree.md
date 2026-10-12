# 项目目录结构

- 生成时间：`2026-10-12 17:56:25`
- 项目根目录：`E:\project-dev\ai-agent\practice\langchain_lab`

```text
langchain_lab/
├── .vscode/
│   └── settings.json
├── config/
│   ├── llm.yaml
│   ├── mcp.yaml
│   ├── project.yaml
│   └── rag.yaml
├── data/
├── docs/
│   ├── pint流式响应单chunk.tool_call_chunks示例 .jsonc
│   ├── pint流式响应单chunk示例.jsonc
│   ├── pint流式响应最后6个chunk示例(观察变化).jsonc
│   ├── 向量化存储及返回索引流程.md
│   └── 文档加载+Chunk切割流程.md
├── logs/
├── src/
│   ├── langchain_lab/
│   │   ├── core/
│   │   │   ├── config/
│   │   │   │   ├── app_scheme/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── llm_models.py
│   │   │   │   │   ├── mcp_models.py
│   │   │   │   │   ├── project_models.py
│   │   │   │   │   └── rag_models.py
│   │   │   │   ├── ident_scheme/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── _config.py
│   │   │   │   ├── _scheme.py
│   │   │   │   ├── app_config.py
│   │   │   │   ├── config_core.py
│   │   │   │   ├── evented_model.py
│   │   │   │   ├── ident_config.py
│   │   │   │   └── loader.py
│   │   │   ├── llm/
│   │   │   │   ├── llm.py
│   │   │   │   └── llm_schema.py
│   │   │   ├── tools/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── tool_middleware.py
│   │   │   │   └── tools.py
│   │   │   ├── __init__.py
│   │   │   ├── messages.py
│   │   │   └── pipeline.py
│   │   ├── experiments/
│   │   │   ├── __init__.py
│   │   │   ├── exp1_llm_basic.py
│   │   │   ├── exp2_single_chat.py
│   │   │   ├── exp3_multi_chat.py
│   │   │   ├── exp4_prompt_template.py
│   │   │   ├── exp5_lcel_pipeline.py
│   │   │   ├── exp6_base_retrieve.py
│   │   │   ├── exp7_rag_indexing.py
│   │   │   └── exp8_rag_querying.py
│   │   ├── rag/
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── main_rag.py
│   │   │   │   └── schemas.py
│   │   │   ├── operation/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── clean_older_file.py
│   │   │   │   ├── playground.mongodb.js
│   │   │   │   └── script_1.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── document_service.py
│   │   │   │   └── query_service.py
│   │   │   ├── storage/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── meta_db.py
│   │   │   │   ├── mongo_db.py
│   │   │   │   └── vector_store.py
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   ├── loader.py
│   │   │   ├── retriever.py
│   │   │   └── splitter.py
│   │   ├── scratch/
│   │   │   ├── temp.py
│   │   │   └── temp_for_retriever.py
│   │   ├── util/
│   │   │   ├── __init__.py
│   │   │   ├── get_path.py
│   │   │   ├── logger.py
│   │   │   ├── prompt_style.py
│   │   │   └── test_printer.py
│   │   ├── __init__.py
│   │   └── main.py
│   └── tests/
│       ├── unit_test/
│       │   └── test_evented_model.py
│       └── test_retriever.py
├── .gitignore
├── .python-version
├── compose.yml
├── gen_tree.py
├── pyproject.toml
├── README.md
└── uv.lock
```
