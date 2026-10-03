"""全局配置"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DATA_DIR = PROJECT_ROOT / "data"

# =====用户侧配置=====
# MongoDB
MONGO_DB = "knowledge_base"
MONGO_COLLECTION = "kb_documents"
USER_MONGO_URI = f"mongodb://app_user:AppPass123456@localhost:27017/{MONGO_DB}?authSource={MONGO_DB}"


# SQLite
META_DB_PATH = DATA_DIR / "sqlite/meta.db"

# Chroma
CHROMA_PATH = DATA_DIR / "chroma"
CHROMA_COLLECTION = "kb_chunks"

# 允许 import-folder 接口访问的根目录
ALLOWED_BASE_DIRS = [
    Path("D:/notes"),  # 个人笔记目录
]

# 上传临时目录
UPLOAD_DIR = DATA_DIR / "uploads"


# =====运维配置=====
# 备份目录
BACKUP_DIR = DATA_DIR / "backup"

# MongoDB
ADMIN_MONGO_URI = "mongodb://root:862697055@localhost:27017/?authSource=admin"


if __name__ == "__main__":
    from pymongo import MongoClient

    client = MongoClient(USER_MONGO_URI)
    db = client[MONGO_DB]
    print(db.list_collection_names())
