"""检索与打分：从记忆库里挑出"这一轮最该给模型看的那几条"。

Day 4 的做法很笨：取重要度最高的 5 条。
问题很明显 —— 你问"我上次说的副业怎么开始"，它却端出"你叫小明"。

本模块做三件事：
  1. **候选召回**：用向量找出语义上相关的记忆（比"最重要"精准得多）
  2. **复合打分**：相关性 × 时间新鲜度 × 重要度 综合排序
  3. **语义去重**：意思相同但措辞不同的记忆合并，避免记忆库被同一件事灌满

关于时间衰减，这里用的是艾宾浩斯遗忘曲线的简化形式：
    新鲜度 = 0.5 ^ (记忆年龄 / 半衰期)
意思是"30 天前发生的事，权重降到一半"。
为什么用指数衰减而不是线性？因为遗忘的规律本来就是先快后慢，
三天前和一周前的差别，远大于三个月前和四个月前的差别。
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

from app.core.config import settings
from app.core.logging import logger
from app.memory import embeddings, vector_store
from app.memory.database import SessionLocal
from app.memory.models import Memory

# 打分权重。四项加起来是 1.0，方便解释"谁在主导排序"。
#
# 为什么相似度从 0.55 降到 0.45？
# 因为要腾出 0.20 给情绪一致性。这不是拍脑袋 —— 设计意图是：
# 相关性仍然主导（没有相关性，其他三项加再多也是错的记忆），
# 但在同样相关的候选里，状态匹配的那条应该优先浮出来。
WEIGHT_SIMILARITY = 0.45   # 语义相关性 —— 仍是主导项
WEIGHT_RECENCY = 0.20      # 时间新鲜度
WEIGHT_IMPORTANCE = 0.15   # 记忆本身的重要度
WEIGHT_EMOTION = 0.20      # 情绪一致性（Day5 情感模块新增）

# 时间半衰期（天）：年龄等于它的记忆，新鲜度降到 0.5
RECENCY_HALF_LIFE_DAYS = 30.0


def emotion_consistency(memory: Memory, state) -> float:
    """算一条记忆和用户当前情绪状态的一致程度，0~1。

    公式设计：consistency = 1 - |记忆效价 - 当前效价| / 2

    为什么用"差异"而不是"同向"？
    因为我们想要的是：用户现在焦虑（效价 -0.5），就优先端出"上次也焦虑时聊的事"
    （效价 -0.4，差异 0.1，一致性 0.95），而不是端出"三周前聊得很开心的事"
    （效价 +0.6，差异 1.1，一致性 0.45）。

    注意：**状态接近中性时这个指标会失效** —— 用户心情平静时，
    所有记忆的差异都不大，这一项就退化成常数，不影响排序。
    这是可以接受的：平静时本来就不需要情绪加权。
    """
    if state is None:
        return 0.5  # 没有状态信息时给中间值，不干扰排序

    m_val = float(memory.valence or 0.0)
    diff = abs(m_val - float(state.valence))
    return max(0.0, 1.0 - diff / 2.0)


def recency_score(created_at: datetime | None, now: datetime | None = None) -> float:
    """时间新鲜度，0~1。越新越接近 1。"""
    if created_at is None:
        return 0.5
    now = now or datetime.now()
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


@dataclass
class ScoredMemory:
    """一条被打分后的记忆。把四个分项都留着，便于调试和做消融实验。"""

    memory: Memory
    similarity: float
    recency: float
    importance: float
    emotion: float
    score: float
    breakdown: dict = field(default_factory=dict)

    @property
    def id(self) -> int:
        return self.memory.id


async def retrieve_memories(
    user_id: str,
    query: str,
    *,
    top_k: int | None = None,
    candidate_k: int = 20,
    emotion_state=None,
    use_emotion: bool = True,
) -> list[ScoredMemory]:
    """检索最相关的记忆。

    为什么先召回 20 条再挑 5 条？
    向量检索给的是"语义最近"，但"最近"不等于"最该给模型看" ——
    三天前的重要目标，语义相似度可能不如一条一模一样的废话。
    所以要拿更多候选出来，再叠加时间、重要度、情绪重新排序。
    这跟重排序（rerank）是同一个思路，只是这里用规则代替了模型。

    use_emotion 参数用于消融实验：关掉它就能对比"情感加权到底有没有用"。
    """
    top_k = top_k or settings.memory_top_k

    if not query.strip():
        return []

    # 1. 召回候选
    try:
        query_vector = embeddings.embed_query(query)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"查询向量化失败，退回按重要度取记忆：{e}")
        return await _fallback_by_importance(user_id, top_k)

    hits = vector_store.search_memories(user_id, query_vector, top_k=candidate_k)

    # 向量库是空的（比如索引还没建），退回按重要度取
    if not hits:
        return await _fallback_by_importance(user_id, top_k)

    # 2. 取完整记录（向量库只有索引，完整信息在 SQLite）
    ids = [h["memory_id"] for h in hits if h.get("memory_id") is not None]
    sim_map = {h["memory_id"]: h["similarity"] for h in hits if h.get("memory_id") is not None}

    async with SessionLocal() as db:
        from sqlalchemy import select

        rows = (
            await db.execute(select(Memory).where(Memory.id.in_(ids)))
        ).scalars().all()
    by_id = {m.id: m for m in rows}

    # 3. 复合打分
    now = datetime.now()
    scored: list[ScoredMemory] = []
    for mid in ids:
        mem = by_id.get(mid)
        if mem is None:
            continue  # 向量库里有、SQLite 里没有：可能是删过，跳过
        sim = sim_map.get(mid, 0.0)
        rec = recency_score(mem.created_at, now)
        imp = float(mem.importance or 0.5)
        emo = emotion_consistency(mem, emotion_state) if use_emotion else 0.5

        if use_emotion:
            score = (
                WEIGHT_SIMILARITY * sim
                + WEIGHT_RECENCY * rec
                + WEIGHT_IMPORTANCE * imp
                + WEIGHT_EMOTION * emo
            )
        else:
            # 消融模式：把情绪那一项的权重还给相似度，保证总分可比
            score = (
                (WEIGHT_SIMILARITY + WEIGHT_EMOTION) * sim
                + WEIGHT_RECENCY * rec
                + WEIGHT_IMPORTANCE * imp
            )

        scored.append(
            ScoredMemory(
                memory=mem,
                similarity=sim,
                recency=rec,
                importance=imp,
                emotion=emo,
                score=score,
                breakdown={
                    "similarity": round(sim, 4),
                    "recency": round(rec, 4),
                    "importance": round(imp, 4),
                    "emotion": round(emo, 4),
                    "use_emotion": use_emotion,
                },
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    result = scored[:top_k]

    if result:
        logger.debug(
            "检索结果 | "
            + " ; ".join(
                f"{s.memory.content[:14]}(sim={s.similarity:.2f},"
                f"emo={s.emotion:.2f},score={s.score:.2f})"
                for s in result
            )
        )
    return result


async def _fallback_by_importance(user_id: str, top_k: int) -> list[ScoredMemory]:
    """降级方案：检索不可用时，退回按重要度取。

    为什么要专门写这个？
    嵌入模型加载失败、向量库损坏都可能发生。这时宁可给出"不那么精准但可用"的
    结果，也不能让整个对话失败 —— 这就是降级链的思想。
    """
    from app.memory import long_term

    memories = await long_term.get_memories(user_id, limit=top_k)
    now = datetime.now()
    out = []
    for m in memories:
        rec = recency_score(m.created_at, now)
        imp = float(m.importance or 0.5)
        out.append(
            ScoredMemory(
                memory=m,
                similarity=0.0,
                recency=rec,
                importance=imp,
                emotion=0.5,
                score=WEIGHT_RECENCY * rec + WEIGHT_IMPORTANCE * imp,
                breakdown={"fallback": True},
            )
        )
    return out


async def retrieve_knowledge(query: str, *, top_k: int = 5) -> list[dict]:
    """检索知识库。返回 [{content, similarity, source}, ...]"""
    if not query.strip():
        return []
    try:
        qv = embeddings.embed_query(query)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"知识库检索失败：{e}")
        return []
    return vector_store.search_knowledge(qv, top_k=top_k)


# ------------------------------------------------------------ 语义去重


def find_duplicate(
    user_id: str,
    content: str,
    *,
    threshold: float = 0.92,
) -> tuple[int | None, float]:
    """检查一句话是不是已经有一条意思相同的记忆了。

    返回 (已存在记忆的 id 或 None, 最高相似度)

    为什么需要这个？
    Day 4 的实测发现：同一件事被不同措辞存了两遍 ——
        会话 B 抽出"用户担心自己坚持不下来"
        会话 C 抽出"用户担心自己无法坚持副业计划"
    精确文本去重拦不住这种，必须用向量算语义相似度。
    """
    try:
        vec = embeddings.embed_documents([content])[0]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"语义去重跳过（向量化失败）：{e}")
        return None, 0.0

    hits = vector_store.search_memories(user_id, vec, top_k=3)
    if not hits:
        return None, 0.0

    best = max(hits, key=lambda h: h["similarity"])
    if best["similarity"] >= threshold:
        return best.get("memory_id"), best["similarity"]
    return None, best["similarity"]
