import time
from pathlib import Path

from langchain_lab.core.config._config import BACKUP_DIR, UPLOAD_DIR
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.OPERATE)


def clean_older_dir(backup_dir: str | Path, keep_days: int = 30, dry_run: bool = True):
    """
    清理备份文件夹，删除修改时间超过 keep_days 天的文件

    :param backup_dir: 备份目录路径
    :param keep_days: 保留多少天
    :param dry_run: True=预览，不删除；False=真实删除
    """
    root = Path(backup_dir)
    if not root.exists():
        logger.warning(f"目录不存在：{backup_dir}")
        return

    cutoff = time.time() - keep_days * 86400
    deleted_count = 0

    # 遍历目录下所有文件（不递归子文件夹，如需递归换成 rglob("*")）
    for f in root.glob("*"):
        if not f.is_file():
            continue
        mtime = f.stat().st_mtime
        if mtime < cutoff:
            if dry_run:
                logger.info(f"[dry-run] 将删除: {f}")
            else:
                f.unlink()
                logger.info(f"已删除: {f}")
            deleted_count += 1

    logger.info(f"扫描完成，总共标记删除 {deleted_count} 个文件")


if __name__ == "__main__":
    # ========== 改这里配置 ==========
    RETAIN_DAYS = 30
    DRY_RUN = False  # 先跑一次预览，确认无误再改成False
    # ==================================
    clean_older_dir(BACKUP_DIR, keep_days=RETAIN_DAYS, dry_run=DRY_RUN)
    clean_older_dir(UPLOAD_DIR, keep_days=RETAIN_DAYS, dry_run=DRY_RUN)
