# 项目目录结构

- 生成时间：`2026-10-27 02:26:35`
- 项目根目录：`E:\project-dev\ai-agent\practice\langchain_lab`

```text
langchain_lab/
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── config/
│   ├── llm.yaml
│   ├── mcp.yaml
│   ├── project.yaml
│   ├── rag.yaml
│   └── runtime.yaml
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
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── control_api.py
│   │   │   │   └── payload_validation.py
│   │   │   ├── commands/
│   │   │   │   ├── __init__.py
│   │   │   │   └── instruction_registry.py
│   │   │   ├── config/
│   │   │   │   ├── app_schema/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── llm_models.py
│   │   │   │   │   ├── mcp_models.py
│   │   │   │   │   ├── project_models.py
│   │   │   │   │   ├── rag_models.py
│   │   │   │   │   └── runtime_models.py
│   │   │   │   ├── ident_schema/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── _config.py
│   │   │   │   ├── _scheme.py
│   │   │   │   ├── app_config.py
│   │   │   │   ├── config_core.py
│   │   │   │   ├── env_expand.py
│   │   │   │   ├── evented_model.py
│   │   │   │   ├── ident_config.py
│   │   │   │   └── loader.py
│   │   │   ├── context/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── compressor.py
│   │   │   │   ├── context_store.py
│   │   │   │   └── views.py
│   │   │   ├── llm/
│   │   │   │   └── llm.py
│   │   │   ├── tools/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_tools.py
│   │   │   │   ├── tool_disclosure.py
│   │   │   │   └── tool_middleware.py
│   │   │   ├── __init__.py
│   │   │   ├── agent_hooks.py
│   │   │   ├── agent_runtime.py
│   │   │   ├── agent_session.py
│   │   │   ├── event_bus.py
│   │   │   ├── events.py
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
│   │   │   ├── exp8_rag_querying.py
│   │   │   └── exp9_agent_runtime.py
│   │   ├── mcp/
│   │   │   ├── client/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mcp_session_manager.py
│   │   │   │   ├── mcp_stdio_client.py
│   │   │   │   ├── mcp_streamable_session.py
│   │   │   │   └── mcp_tool_registry.py
│   │   │   ├── mcp_tools/
│   │   │   │   ├── __init__.py
│   │   │   │   └── rag_mcp_server.py
│   │   │   ├── server/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── context_inject.py
│   │   │   │   ├── mcp_stdio_server.py
│   │   │   │   └── mcp_streamable_http_server.py
│   │   │   └── __init__.py
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
│       │   ├── test_agent_runtime.py
│       │   ├── test_branch.py
│       │   ├── test_chat_session.py
│       │   ├── test_event_bus.py
│       │   ├── test_evented_model.py
│       │   ├── test_regenerate.py
│       │   ├── test_session_compressor.py
│       │   ├── test_session_store.py
│       │   └── test_session_tree.py
│       └── test_retriever.py
├── .env.example
├── .gitignore
├── .python-version
├── _diag_t.py
├── _diag_turn.py
├── compose.yml
├── gen_tree.py
├── pyproject.toml
├── README.md
└── uv.lock
```
