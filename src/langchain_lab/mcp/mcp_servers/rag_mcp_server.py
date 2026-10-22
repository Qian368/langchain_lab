"""RAG Knowledge Base MCP Server（路径 A：身份在连接层注入）

身份注入方式：
- stdio：进程启动时从环境变量 MCP_TENANT_ID / MCP_USER_ID / MCP_IS_ADMIN 读取
- HTTP：每个请求从 Header X-Tenant-ID / X-User-ID / X-Tenant-Admin 读取，中间件写入 contextvar

工具 schema 全部无用户/租户字段，LLM 只看到业务参数。
"""

import asyncio
from pathlib import Path  # noqa: F401

from mcp.server.mcpserver import MCPServer

from langchain_lab.core.config.config_core import get_config
from langchain_lab.mcp.server_encapsulation.context_inject import bind_exec_context, get_exec_context, setup_exec_context
from langchain_lab.rag.services import DocumentService, QueryService

document_service = DocumentService()
query_service = QueryService()

config = get_config()

ALLOWED_BASE_DIRS = config.rag.file.allowed_base_dirs


mcp = MCPServer("rag-kb")


@mcp.tool()
async def rag_query(question: str, top_k: int = 4) -> list[dict]:
    """纯向量语义检索知识库。
    适合语义相似问题，返回相关chunk列表。"""
    identity = get_exec_context()
    return await asyncio.to_thread(
        query_service.retrieve,
        question=question,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        is_tenant_admin=identity.is_tenant_admin,
        top_k=top_k,
    )


@mcp.tool()
async def rag_query_hybrid(question: str, top_k: int = 4) -> list[dict]:
    """混合检索：向量+关键字，RRF融合。适合专业词/缩写场景。"""
    identity = get_exec_context()
    return await asyncio.to_thread(
        query_service.hybrid_retrieve,
        question=question,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        is_tenant_admin=identity.is_tenant_admin,
        top_k=top_k,
    )


@mcp.tool()
def rag_list_documents(status: str | None = None, skip: int = 0, limit: int = 20) -> dict:
    """列出当前租户可见文档，支持分页和状态过滤（pending/indexed/failed）。
    返回 {"items": [...], "total": N}。"""
    identity = get_exec_context()
    limit = min(limit, 100)
    items, total = document_service.list_documents(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        is_tenant_admin=identity.is_tenant_admin,
        status=status,
        skip=skip,
        limit=limit,
    )
    return {"items": items, "total": total}


# region 不必要暴露的内容
# @mcp.tool()
# def rag_get_document(doc_id: str) -> dict:
#     """获取单个文档详情（含全文）。租户不匹配返回错误。"""
#     identity = get_exec_context()
#     doc = document_service.get_document_detail(doc_id=doc_id, tenant_id=identity.tenant_id)
#     if not doc:
#         return {"error": "document not found or no access"}
#     doc["_id"] = str(doc["_id"])
#     return doc


# @mcp.tool()
# async def rag_delete_document(doc_id: str) -> dict:
#     """删除文档及其所有chunk和向量。写操作，不可恢复，请确认后调用。"""
#     identity = get_exec_context()
#     ok = await asyncio.to_thread(document_service.delete_document, doc_id=doc_id, tenant_id=identity.tenant_id)
#     if not ok:
#         return {"error": "document not found or no access"}
#     return {"deleted": True, "doc_id": doc_id}


# @mcp.tool()
# async def rag_reindex_document(doc_id: str) -> dict:
#     """重建索引：删旧chunk/向量 -> 重新切分 -> 重嵌 -> 重存。
#     写操作，耗时较长，请确认后调用。"""
#     identity = get_exec_context()
#     doc = document_service.get_document_detail(doc_id=doc_id, tenant_id=identity.tenant_id)
#     if not doc:
#         return {"error": "document not found or no access"}
#     await asyncio.to_thread(document_service.reindex_document, doc_id=doc_id, tenant_id=identity.tenant_id)
#     return {"status": "reindexed", "doc_id": doc_id}


# @mcp.tool()
# async def rag_upload_file(path: str) -> dict:
#     """上传本地 .md/.txt/.markdown 文件到知识库（只入库不索引）。
#     返回 doc_id 后，请调用 rag_index_document 执行索引。"""
#     identity = get_exec_context()
#     file_path = Path(path)
#     if not file_path.is_file():
#         return {"error": f"file not found: {path}"}
#     if file_path.suffix.lower() not in {".md", ".txt", ".markdown"}:
#         return {"error": f"unsupported file type: {file_path.suffix}"}

#     doc_id = await asyncio.to_thread(
#         document_service.store_file,
#         file_path=file_path,
#         tenant_id=identity.tenant_id,
#         user_id=identity.user_id,
#     )
#     return {"doc_id": doc_id, "filename": file_path.name, "status": "stored"}


# @mcp.tool()
# async def rag_index_document(doc_id: str) -> dict:
#     """对已上传文档执行索引：切分chunk -> embedding -> 写入SQLite和Chroma。
#     对应 FastAPI 后台任务中的 index_document。"""
#     identity = get_exec_context()
#     doc = document_service.get_document_detail(doc_id=doc_id, tenant_id=identity.tenant_id)
#     if not doc:
#         return {"error": "document not found or no access"}
#     await asyncio.to_thread(
#         document_service.index_document,
#         doc_id=doc_id,
#         tenant_id=identity.tenant_id,
#     )
#     return {"status": "indexed", "doc_id": doc_id}


# @mcp.tool()
# async def rag_import_folder(folder: str) -> dict:
#     """批量导入文件夹内所有 md/txt/markdown 文件并索引。
#     写操作，耗时较长。folder 必须在 ALLOWED_BASE_DIRS 范围内。"""
#     identity = get_exec_context()
#     folder_path = Path(folder)
#     allowed = any(folder_path.resolve() == base.resolve() or folder_path.resolve().is_relative_to(base.resolve()) for base in ALLOWED_BASE_DIRS)
#     if not allowed:
#         return {"error": f"forbidden folder: {folder}"}

#     await asyncio.to_thread(
#         document_service.import_folder,
#         folder_path=folder_path,
#         tenant_id=identity.tenant_id,
#         user_id=identity.user_id,
#     )
#     return {"status": "imported", "folder": str(folder_path)}
# endregion


@mcp.tool()
def rag_health() -> dict:
    """健康检查：确认 MongoDB / Chroma / SQLite 连接状态。"""
    status = {"status": "ok"}
    try:
        document_service.mongo.client.admin.command("ping")
        status["mongo"] = "ok"
    except Exception as e:
        status["mongo"] = f"error: {e}"
    try:
        document_service.vector.ping()
        status["chroma"] = "ok"
    except Exception as e:
        status["chroma"] = f"error: {e}"
    try:
        document_service.meta.ping()
        status["sqlite"] = "ok"
    except Exception as e:
        status["sqlite"] = f"error: {e}"
    return status


# ===== 启动 mcp 服务 =====
async def run_single_mcp_session():
    token_wrapper = bind_exec_context()
    try:
        # 直接调用底层async方法，不再走 mcp.run()（它会新建anyio循环）
        await mcp.run_stdio_async()
    finally:
        token_wrapper.reset()


async def main():
    # ========== 服务全局模板初始化（整个进程仅一次） ==========
    setup_exec_context()
    await run_single_mcp_session()


if __name__ == "__main__":
    asyncio.run(main())
