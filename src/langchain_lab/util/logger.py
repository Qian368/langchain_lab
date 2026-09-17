"""
统一日志配置模块

用法：
    from langchain_lab.util.logger import logger

    logger.info("普通信息")
    logger.debug("调试信息")
    logger.warning("警告")
    logger.error("错误")
    logger.exception("异常，自动带堆栈")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 项目根目录下的 logs 文件夹
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志格式：时间 | 级别 | 模块名 | 消息
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 轮转配置
MAX_BYTES = 1 * 1024 * 1024  # 单个文件最大 1MB
BACKUP_COUNT = 3  # 保留 5 个备份文件


def _setup_logger() -> logging.Logger:
    """配置并返回全局 logger 实例"""
    logger = logging.getLogger("langchain_lab")
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 handler（模块被多次导入时）
    if logger.handlers:
        return logger

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)

    # 文件输出（所有级别）- 带轮转
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    # 错误单独一个文件（只记 ERROR 及以上）- 带轮转
    error_handler = RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(error_handler)

    return logger


# 模块级单例，整个项目共享同一个实例
logger = _setup_logger()
