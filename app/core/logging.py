"""日志配置：每条日志都带 trace_id，方便把一次请求的所有日志串起来。

为什么要有 trace_id？
一次对话会经过 接入 → 记忆 → 检索 → 编排 → 模型 五步，每一步都打日志。
没有 trace_id，日志就是一锅粥，出了问题你只能靠猜。
有了它，grep 一个 id 就能看到这次请求的完整经过。
"""

import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

from app.core.config import ROOT_DIR, settings

# 当前请求的 trace_id。用 ContextVar 而不是全局变量：
# 这样多个请求并发时，各自的 trace_id 互不干扰。
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> str:
    return _trace_id.get()


def _patch(record) -> None:
    """往每条日志里塞入 trace_id 字段。"""
    record["extra"]["trace_id"] = _trace_id.get()


def setup_logging() -> None:
    """初始化日志。在 main.py 启动时调用一次。"""
    logger.remove()  # 去掉 loguru 默认的 handler

    # 控制台：带颜色，人类看的
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
            "<cyan>{extra[trace_id]}</cyan> | <level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件：完整信息，出问题回溯用的
    log_dir: Path = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "app.log",
        level="DEBUG",
        rotation="10 MB",     # 单文件超过 10MB 就切
        retention="7 days",   # 只留 7 天
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {extra[trace_id]} | {name}:{line} | {message}",
    )

    logger.configure(patcher=_patch)


__all__ = ["logger", "setup_logging", "set_trace_id", "get_trace_id"]
