"""后台任务管理。

为什么需要这个文件？
记忆抽取要额外调用一次模型（慢），绝不能让你等着。所以主流程先把回复返回，
抽取丢到后台慢慢跑。

但后台任务有两个坑，这里都处理了：
  1. asyncio 不会自动保存任务的引用，任务可能在跑完前被垃圾回收 → 用集合持有引用
  2. 测试需要等"记忆真的写完了"才能断言 → 提供 wait_all()

面试可讲：这就是"延迟敏感路径"与"非关键路径"的分离。
用户感知的延迟只算生成回答的时间，写记忆不在关键路径上。
"""

import asyncio

from app.core.logging import logger

# 持有正在运行的后台任务引用，防止被 GC 回收
_running: set[asyncio.Task] = set()


def spawn(coro, *, name: str = "background") -> asyncio.Task:
    """把一个协程丢到后台执行，立即返回，不等待。"""
    task = asyncio.create_task(coro, name=name)
    _running.add(task)

    def _done(t: asyncio.Task) -> None:
        _running.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            # 后台任务出错不能影响主流程，只记日志
            logger.error(f"后台任务 {name} 执行失败：{type(exc).__name__}: {exc}")

    task.add_done_callback(_done)
    return task


async def wait_all(timeout: float = 30.0) -> None:
    """等所有后台任务跑完。测试和优雅关闭时用。"""
    if not _running:
        return
    tasks = list(_running)
    logger.debug(f"等待 {len(tasks)} 个后台任务完成…")
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        logger.warning(f"还有 {len(pending)} 个后台任务未在 {timeout}s 内完成")


def pending_count() -> int:
    return len(_running)
