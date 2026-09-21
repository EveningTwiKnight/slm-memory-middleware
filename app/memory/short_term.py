"""短期记忆：最近几轮对话的原文。

这是最"土"但最关键的一层。没有它，模型每轮都是失忆的。
Day 4~6 会在它前面加"检索"，但最近几轮的原文永远直接给，
因为对话的连贯性（"刚才说的那个事"）靠检索是找不回来的。
"""

from sqlalchemy import delete, func, select

from app.core.config import settings
from app.core.logging import logger
from app.llm.base import Message
from app.memory.database import SessionLocal
from app.memory.models import Conversation


async def append(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
) -> None:
    """写入一条消息。role 只能是 user 或 assistant。"""
    if role not in ("user", "assistant"):
        raise ValueError(f"role 只能是 user/assistant，收到 {role}")

    async with SessionLocal() as db:
        db.add(
            Conversation(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
            )
        )
        await db.commit()


async def get_recent(session_id: str, limit: int | None = None) -> list[Message]:
    """取最近 N 轮对话，按时间正序返回（最早的在前面）。

    为什么要"先倒序取、再正序返回"？
    因为数据库里我们要的是"最近的 N 条"，所以要按时间倒序 LIMIT N；
    但发给模型时必须按时间正序，否则模型看到的是倒着说的对话。
    """
    limit = limit or settings.history_limit
    async with SessionLocal() as db:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.id.desc())
            .limit(limit)
        )
        rows = list(result.scalars().all())

    rows.reverse()  # 恢复成时间正序
    return [Message(role=r.role, content=r.content) for r in rows]


async def count(session_id: str) -> int:
    """这个会话一共有多少条消息。"""
    async with SessionLocal() as db:
        result = await db.execute(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.session_id == session_id)
        )
        return int(result.scalar_one())


async def list_sessions(user_id: str) -> list[dict]:
    """列出某个用户的所有会话及各自的消息数。调试和演示用。"""
    async with SessionLocal() as db:
        result = await db.execute(
            select(
                Conversation.session_id,
                func.count().label("n"),
                func.min(Conversation.created_at).label("started"),
                func.max(Conversation.created_at).label("last"),
            )
            .where(Conversation.user_id == user_id)
            .group_by(Conversation.session_id)
            .order_by(func.max(Conversation.created_at).desc())
        )
        return [
            {
                "session_id": r.session_id,
                "messages": r.n,
                "started_at": r.started.isoformat() if r.started else None,
                "last_at": r.last.isoformat() if r.last else None,
            }
            for r in result.all()
        ]


async def clear_session(session_id: str) -> int:
    """删掉一个会话的全部消息，返回删除条数。"""
    async with SessionLocal() as db:
        result = await db.execute(
            delete(Conversation).where(Conversation.session_id == session_id)
        )
        await db.commit()
        n = result.rowcount or 0
    logger.info(f"清空会话 {session_id}，删除 {n} 条消息")
    return n
