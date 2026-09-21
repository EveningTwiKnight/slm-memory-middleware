"""数据库连接管理。

用 SQLAlchemy 2.0 异步接口 + SQLite。选 SQLite 的理由：
- 零部署，一个文件就是一个库，改起来不用运维；
- 个人项目的数据量（几万条对话）它完全扛得住；
- 换成 PostgreSQL 只需要改这一个文件里的连接串 —— 这是分层的价值。

面试可讲："为什么不用 MySQL？因为这个规模用 SQLite 是正确取舍，
瓶颈不在数据库；真到多实例部署时，改 DATABASE_URL 就能换 PG。"
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import logger
from app.memory.models import Base

# SQLite 的异步驱动是 aiosqlite
DATABASE_URL = f"sqlite+aiosqlite:///{settings.db_file.as_posix()}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # True 会打印所有 SQL，调试时开
    future=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,   # commit 后对象仍可访问属性，避免多余查询
)


async def init_db() -> None:
    """建表 + 补齐缺失的列。服务启动时调一次。

    顺序很重要：先 create_all 建缺失的表，再 auto_migrate 给已有的表补列。
    反过来的话，新建的表会被 auto_migrate 当成"已存在"而跳过；
    实际上 auto_migrate 只处理已存在的表，所以必须在 create_all 之后跑。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"数据库就绪：{settings.db_file}")

    # SQLAlchemy 的 create_all 不会修改已存在的表 ——
    # 所以给 models 加字段后，老库需要自动补列（见 migrate.py）
    from app.memory.migrate import auto_migrate

    await auto_migrate(engine)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖注入用：每个请求一个会话，用完自动关闭。"""
    async with SessionLocal() as session:
        yield session


async def close_db() -> None:
    await engine.dispose()
