"""
FastAPI 入口：提供文档管理和检索接口。
- POST /documents/upload          单个文件上传（同步存Mongo，后台索引）
- POST /documents/import-folder   批量导入文件夹（后台执行）
- GET  /documents                 文档列表
- GET  /documents/{doc_id}        文档详情
- DELETE /documents/{doc_id}      删除文档
- POST /documents/{doc_id}/reindex 重建索引
- POST /query                     纯向量检索
- POST /query/hybrid              混合检索（向量+关键字）
- GET  /health                    健康检查
"""

import os
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from pymongo.errors import DuplicateKeyError

from langchain_lab.core.config.config_core import get_config
from langchain_lab.rag.api.schemas import (
    DocumentItem,
    DocumentListResponse,
    ImportFolderResponse,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
    UploadResponse,
    UserIdentity,
)
from langchain_lab.rag.services import DocumentService, QueryService

app = FastAPI(title="RAG Knowledge Base API", version="0.1.0")
# 服务单例
document_service = DocumentService()
query_service = QueryService()

config = get_config()
UPLOAD_DIR = config.rag.file.upload_dir
session_ctx = config.project.session_ctx

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTS = {".md", ".txt", ".markdown"}

DEFAULT_TENANT_ID = session_ctx["tenant_id"]
DEFAULT_USER_ID = session_ctx["user_id"]


# ========== 模拟身份解析依赖，生产替换为JWT解析 ==========
def get_identity(
    tenant_id: str = DEFAULT_TENANT_ID,
    user_id: str = DEFAULT_USER_ID,
    is_tenant_admin: bool = False,
) -> UserIdentity:
    """
    模拟身份注入；生产环境改成从Authorization Header解析JWT。

    这里通过 Depends， 会自动把函数的参数当成请求参数去解析（比如 query /path，不会从 body）。
    """
    return UserIdentity(tenant_id=tenant_id, user_id=user_id, is_tenant_admin=is_tenant_admin)


# ---------- 文档管理 ----------
@app.post("/documents/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    identity: UserIdentity = Depends(get_identity),
    file: UploadFile = File(...),
):
    """
    上传单个文档文件（multipart/form-data）。
    流程：
    1. 同步：保存临时文件 → 存入 MongoDB（拿到 doc_id）
    2. 异步：后台切分索引
    """
    try:
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="上传文件缺少文件名")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

        temp_path = UPLOAD_DIR / Path(filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            f.flush()
            os.fsync(f.fileno())

        # 同步入库，传入租户身份
        try:
            doc_id = document_service.store_file(file_path=temp_path, tenant_id=identity.tenant_id, user_id=identity.user_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档入库失败: {e}")

        # 后台索引，传入tenant_id用于校验
        background_tasks.add_task(document_service.index_document, doc_id=doc_id, tenant_id=identity.tenant_id)
        return UploadResponse(
            filename=filename,
            status="accepted",
            message=f"文档已入库，doc_id={doc_id}，正在后台索引",
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="文件已存在，不允许重复上传")


@app.post("/documents/import-folder", response_model=ImportFolderResponse, status_code=202)
async def import_folder(
    folder: str,
    background_tasks: BackgroundTasks,
    identity: UserIdentity = Depends(get_identity),
):
    """
    批量导入文件夹。参数传递方式：JSON body {"folder": "D:/notes"}。
    校验文件夹路径在 ALLOWED_BASE_DIRS 范围内（原硬编码已收敛到 config.rag.file.allowed_base_dirs）。
    """
    folder_path = Path(folder)
    allowed = any(
        folder_path.resolve() == base.resolve() or folder_path.resolve().is_relative_to(base.resolve())
        for base in config.rag.file.allowed_base_dirs
    )
    if not allowed:
        raise HTTPException(status_code=403, detail=f"禁止访问该目录: {folder_path}")

    background_tasks.add_task(
        document_service.import_folder,
        folder_path=folder_path,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
    )
    return ImportFolderResponse(folder=str(folder_path))


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status: str | None = None,
    identity: UserIdentity = Depends(get_identity),
):
    """文档列表，支持分页和状态过滤，按角色自动过滤可见文档"""
    items, total = document_service.list_documents(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        is_tenant_admin=identity.is_tenant_admin,
        status=status,
        skip=skip,
        limit=limit,
    )
    return {"items": items, "total": total}


@app.get("/documents/{doc_id}", response_model=DocumentItem)
async def get_document(
    doc_id: str,
    identity: UserIdentity = Depends(get_identity),
):
    """查看单个文档详情（含全文），租户隔离校验"""
    doc = document_service.get_document_detail(doc_id=doc_id, tenant_id=identity.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在或无访问权限")
    doc["_id"] = str(doc["_id"])
    return doc


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    identity: UserIdentity = Depends(get_identity),
):
    """删除文档及其所有 chunk 和向量，租户隔离校验"""
    ok = document_service.delete_document(doc_id=doc_id, tenant_id=identity.tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在或无权限")
    return {"deleted": True, "doc_id": doc_id}


@app.post("/documents/{doc_id}/reindex", status_code=202)
async def reindex_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    identity: UserIdentity = Depends(get_identity),
):
    """重建索引，后台任务"""
    doc = document_service.get_document_detail(doc_id=doc_id, tenant_id=identity.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在或无访问权限")
    background_tasks.add_task(document_service.reindex_document, doc_id=doc_id, tenant_id=identity.tenant_id)
    return {"status": "accepted", "message": f"重建索引任务已受理: {doc_id}"}


# ---------- 检索 ----------
@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    identity: UserIdentity = Depends(get_identity),
):
    """纯向量检索"""
    result_dicts = query_service.retrieve(
        question=req.question, tenant_id=identity.tenant_id, user_id=identity.user_id, is_tenant_admin=identity.is_tenant_admin, top_k=req.top_k
    )
    results = [QueryResultItem.model_validate(item) for item in result_dicts]
    return QueryResponse(question=req.question, total=len(results), results=results)


@app.post("/query/hybrid", response_model=QueryResponse)
async def query_hybrid(
    req: QueryRequest,
    identity: UserIdentity = Depends(get_identity),
):
    """混合检索：向量 + 关键字"""
    result_dicts = query_service.hybrid_retrieve(
        question=req.question, tenant_id=identity.tenant_id, user_id=identity.user_id, is_tenant_admin=identity.is_tenant_admin, top_k=req.top_k
    )
    results = [QueryResultItem.model_validate(item) for item in result_dicts]
    return QueryResponse(question=req.question, total=len(results), results=results)


# ---------- 健康检查 ----------
@app.get("/health")
async def health():
    """健康检查：确认三个存储是否连接正常"""
    status = {"status": "ok"}
    # MongoDB
    try:
        document_service.mongo.client.admin.command("ping")
        status["mongo"] = "ok, connection alive"
    except Exception as e:
        status["mongo"] = f"error: {e}"
    # Chroma
    try:
        document_service.vector.ping()
        status["chroma"] = "ok, connection alive"
    except Exception as e:
        status["chroma"] = f"error: {e}"
    # SQLite
    try:
        document_service.meta.ping()
        status["sqlite"] = "ok, connection alive"
    except Exception as e:
        status["sqlite"] = f"error: {e}"

    return status


# ---------- 调试服务 ----------
@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    """新增 scalar 路由"""
    from scalar_fastapi import get_scalar_api_reference

    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)


# ---------- 启动服务 ----------
def serve():
    import uvicorn

    uvicorn.run("langchain_lab.rag.api.main_rag:app", reload=True)
