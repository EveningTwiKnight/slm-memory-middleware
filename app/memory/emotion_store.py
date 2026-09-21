"""情绪状态的持久化：读写 user_emotions 表。

为什么单独一个模块，而不是塞进 long_term.py？
因为职责不同：
  long_term     管"记住的事实"（离散的、一条条的）
  emotion_store 管"此刻的状态"（连续的、随时间变化的）
混在一起会让 long_term 越来越臃肿，而且两者更新频率和时机都不一样。
"""

from datetime import datetime

from sqlalchemy import delete, desc, select

from app.core.logging import logger
from app.memory.database import SessionLocal
from app.memory.emotion import (
    EmotionScore,
    EmotionState,
    label_for_state,
    update_state,
)
from app.memory.models import Memory, UserEmotion

# 状态自然回落：多久没说话，即时状态就往中性回一点。
# 为什么要这个？用户三天前很生气，今天再来时系统不该还处于"他很生气"的状态 ——
# 情绪是有时效的，但长期倾向要保留。
DECAY_AFTER_HOURS = 6
DECAY_RATE = 0.5  # 每超过一个衰减周期，即时状态向中性收拢一半


def _apply_time_decay(row: UserEmotion) -> None:
    """按离线时长让即时状态向中性回落，长期倾向衰减更慢。"""
    if row.updated_at is None:
        return
    hours = (datetime.now() - row.updated_at).total_seconds() / 3600.0
    if hours < DECAY_AFTER_HOURS:
        return

    periods = hours / DECAY_AFTER_HOURS
    factor = DECAY_RATE**periods  # 指数收拢，越久越接近中性

    row.current_valence = round(row.current_valence * factor, 4)
    row.current_arousal = round(row.current_arousal * factor, 4)
    # 长期倾向保留得更多：它是"这个人整体什么样"，不该因为两天没聊就归零
    row.stable_valence = round(row.stable_valence * (factor * 0.5 + 0.5), 4)
    row.stable_arousal = round(row.stable_arousal * (factor * 0.5 + 0.5), 4)
    logger.debug(
        f"情绪状态时间衰减：离线 {hours:.1f}h，即时状态收拢到 {row.current_valence:+.3f}"
    )


def _row_to_state(row: UserEmotion) -> EmotionState:
    return EmotionState(
        valence=row.current_valence,
        arousal=row.current_arousal,
        label=label_for_state(row.current_valence, row.current_arousal),
        stable_valence=row.stable_valence,
        stable_arousal=row.stable_arousal,
        turns=row.turns,
        recent_low=(
            row.current_valence <= -0.25
            and row.stable_valence <= -0.25
            and row.turns >= 2
        ),
    )


async def get_state(user_id: str) -> EmotionState:
    """读用户当前情绪状态。没有记录就返回中性状态。"""
    async with SessionLocal() as db:
        row = (
            await db.execute(select(UserEmotion).where(UserEmotion.user_id == user_id))
        ).scalar_one_or_none()

        if row is None:
            return EmotionState()

        _apply_time_decay(row)
        await db.commit()
        return _row_to_state(row)


async def update_from_score(user_id: str, score: EmotionScore) -> EmotionState:
    """用一轮对话的情绪打分更新用户状态，返回更新后的状态。"""
    async with SessionLocal() as db:
        row = (
            await db.execute(select(UserEmotion).where(UserEmotion.user_id == user_id))
        ).scalar_one_or_none()

        if row is None:
            row = UserEmotion(user_id=user_id)
            db.add(row)
            await db.flush()

        _apply_time_decay(row)

        state = _row_to_state(row)
        state = update_state(state, score)

        row.current_valence = state.valence
        row.current_arousal = state.arousal
        row.stable_valence = state.stable_valence
        row.stable_arousal = state.stable_arousal
        row.turns = state.turns

        await db.commit()
        state.label = label_for_state(state.valence, state.arousal)
        return state


async def get_trajectory(user_id: str, limit: int = 20) -> list[dict]:
    """取用户情绪的历史轨迹（从记忆里反推）。

    数据来源是 memories 表里带情绪的记忆 —— 每条记忆记录了当时的情绪，
    按时间排起来就是一条情绪曲线。演示时画成折线图非常直观。
    """
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Memory)
                .where(Memory.user_id == user_id, Memory.valence != 0.0)
                .order_by(desc(Memory.created_at))
                .limit(limit)
            )
        ).scalars().all()

    rows.reverse()
    return [
        {
            "at": m.created_at.isoformat() if m.created_at else "",
            "valence": m.valence,
            "arousal": m.arousal,
            "label": m.emotion_label,
            "content": m.content[:40],
        }
        for m in rows
    ]


async def clear_user(user_id: str) -> int:
    """删除用户的情绪状态（合规要求）。"""
    async with SessionLocal() as db:
        r = await db.execute(delete(UserEmotion).where(UserEmotion.user_id == user_id))
        await db.commit()
    return r.rowcount or 0
