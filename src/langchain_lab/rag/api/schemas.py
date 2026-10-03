"""FastAPI 的 Pydantic 模型定义"""

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """上传接口响应（异步任务已受理）"""

    filename: str
    status: str = "pending"
    message: str = "已受理，正在后台索引"


class ImportFolderResponse(BaseModel):
    """文件夹导入响应（异步任务已受理）"""

    folder: str
    status: str = "processing"
    message: str = "已提交批量导入，处理结束后可刷新文档列表查看"


class DocumentListResponse(BaseModel):
    """文档列表响应"""

    items: list[dict]
    total: int


class QueryRequest(BaseModel):
    """检索请求"""

    question: str
    top_k: int = 4


class QueryResultItem(BaseModel):
    """检索结果中的单条记录"""

    chunk_id: str
    doc_id: str
    score: float
    source: str
    section: str
    content: str


class QueryResponse(BaseModel):
    """检索响应——直接返回结果列表，不包含 LLM 回答"""

    question: str
    total: int
    results: list[QueryResultItem]
