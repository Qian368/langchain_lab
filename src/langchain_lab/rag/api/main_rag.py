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

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pymongo.errors import DuplicateKeyError

from langchain_lab.config import UPLOAD_DIR
from langchain_lab.rag.api.schemas import (
    DocumentListResponse,
    ImportFolderResponse,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
    UploadResponse,
)
from langchain_lab.rag.services import DocumentService, QueryService

app = FastAPI(title="RAG Knowledge Base API", version="0.1.0")

# 服务实例（单例）
document_service = DocumentService()
query_service = QueryService()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".md", ".txt", ".markdown"}


# ---------- 文档管理 ----------


@app.post("/documents/upload", response_model=UploadResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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

        # 保存临时文件
        # temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        temp_path = UPLOAD_DIR / f"{filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            f.flush()
            os.fsync(f.fileno())

        # 同步入库，拿 doc_id
        try:
            doc_id = document_service.store_file(temp_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档入库失败: {e}")

        # 后台索引
        background_tasks.add_task(document_service.index_document, doc_id)

        return UploadResponse(
            filename=filename,
            status="accepted",
            message=f"文档已入库，doc_id={doc_id}，正在后台索引",
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="文件已存在，不允许重复上传")


@app.post("/documents/import-folder", response_model=ImportFolderResponse, status_code=202)
async def import_folder(folder: str, background_tasks: BackgroundTasks):
    """
    批量导入文件夹。参数传递方式：JSON body {"folder": "D:/notes"}。
    校验文件夹路径在 ALLOWED_BASE_DIRS 范围内。
    """
    folder_path = Path(folder)

    # 权限校验（提前拦截，避免后台任务报错）
    from langchain_lab.config import ALLOWED_BASE_DIRS

    allowed = any(folder_path.resolve() == base.resolve() or folder_path.resolve().is_relative_to(base.resolve()) for base in ALLOWED_BASE_DIRS)
    if not allowed:
        raise HTTPException(status_code=403, detail=f"禁止访问该目录: {folder_path}")

    background_tasks.add_task(document_service.import_folder, folder_path)

    return ImportFolderResponse(folder=str(folder_path))


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status: str | None = None,
):
    """文档列表，支持分页和状态过滤"""
    items = document_service.mongo.list_documents(skip=skip, limit=limit, status=status)
    return {"items": items, "total": len(items)}


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """查看单个文档详情（含全文）"""
    doc = document_service.mongo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    doc["_id"] = str(doc["_id"])
    return doc


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档及其所有 chunk 和向量"""
    ok = document_service.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": True, "doc_id": doc_id}


@app.post("/documents/{doc_id}/reindex", status_code=202)
async def reindex_document(doc_id: str, background_tasks: BackgroundTasks):
    """重建索引"""
    doc = document_service.mongo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    background_tasks.add_task(document_service.reindex_document, doc_id)
    return {"status": "accepted", "message": f"重建索引任务已受理: {doc_id}"}


# ---------- 检索 ----------


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """纯向量检索"""
    result_dicts = query_service.retrieve(question=req.question, top_k=req.top_k)
    results = [QueryResultItem.model_validate(item) for item in result_dicts]
    return QueryResponse(question=req.question, total=len(results), results=results)


@app.post("/query/hybrid", response_model=QueryResponse)
async def query_hybrid(req: QueryRequest):
    """混合检索：向量 + 关键字"""
    result_dicts = query_service.hybrid_retrieve(question=req.question, top_k=req.top_k)
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
        status["mongo"] = "ok"
    except Exception as e:
        status["mongo"] = f"error: {e}"

    # Chroma
    try:
        count = document_service.vector.count()
        status["chroma"] = f"ok, {count} vectors"
    except Exception as e:
        status["chroma"] = f"error: {e}"

    # SQLite
    try:
        count = document_service.meta.count_chunks()
        status["sqlite"] = f"ok, {count} chunks"
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
