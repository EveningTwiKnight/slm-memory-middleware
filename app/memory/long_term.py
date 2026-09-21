"""长期记忆：memories 表的读写。

和 short_term.py（存原文流水账）的分工：
  short_term  最近 N 轮对话原文，会话级，直接全量给模型
  long_term   提炼后的结论，用户级、永久保存、需要检索才给模型

本模块负责：写入去重、按重要度/时间读取、档案项冲突消解。
"""

from datetime import UTC, datetime

from sqlalchemy import delete, desc, func, select

from app.core.config import settings
from app.core.logging import logger
from app.memory.database import SessionLocal
from app.memory.extractor import MemoryItem
from app.memory.models import Memory, ProfileFact

# 哪些类型的记忆应该同步写进"用户档案"
# 档案会全量注入提示词，所以只放最核心、最确定的信息
PROFILE_TYPES = {"fact"}


async def save_memories(
    user_id: str,
    items: list[MemoryItem],
    *,
    source_session: str | None = None,
    dedupe: bool = True,
    semantic_dedupe: bool = True,
) -> list[Memory]:
    """写入一批记忆，返回真正新增的那些。

    三重防污染，一层比一层严格：
      1. 空内容/超短内容直接丢（在 extractor 里已处理）
      2. 完全相同的文本不重复写（精确去重）
      3. 语义相同但措辞不同的合并（语义去重，需要向量库）

    第 3 层是 Day 5 加的，解决实测发现的问题：
      "用户担心自己坚持不下来" 和 "用户担心自己无法坚持副业计划"
      是同一件事，精确去重拦不住。

    另外每次写入都会同步建立向量索引，保证"存进去的就能被检索到"。
    """
    if not items:
        return []

    # 延迟导入，避免模块级循环依赖（retrieval -> memory -> retrieval）
    from app.memory import embeddings, vector_store
    from app.retrieval import memory_retriever

    saved: list[Memory] = []
    stats = {"exact_skip": 0, "semantic_skip": 0, "new": 0}

    async with SessionLocal() as db:
        existing: set[str] = set()
        if dedupe:
            rows = await db.execute(select(Memory.content).where(Memory.user_id == user_id))
            existing = {r[0].strip() for r in rows.all()}

        for item in items:
            key = item.content.strip()

            # --- 第 2 层：精确去重 ---
            if dedupe and key in existing:
                stats["exact_skip"] += 1
                logger.debug(f"精确重复，跳过：{key[:30]}")
                continue

            # --- 第 3 层：语义去重 ---
            if semantic_dedupe:
                dup_id, sim = memory_retriever.find_duplicate(
                    user_id, key, threshold=settings.semantic_dedupe_threshold
                )
                if dup_id is not None:
                    stats["semantic_skip"] += 1
                    logger.info(
                        f"语义重复（相似度 {sim:.3f}，已有记忆 #{dup_id}），跳过：{key[:40]}"
                    )
                    continue

            mem = Memory(
                user_id=user_id,
                content=key,
                type=item.type,
                importance=item.importance,
                valence=item.valence,
                arousal=item.arousal,
                emotion_label=item.emotion_label,
                source_session=source_session,
            )
            db.add(mem)
            existing.add(key)
            saved.append(mem)
            stats["new"] += 1

        await db.commit()

        # commit 后主键才有值，这时才能建立向量索引
        if saved:
            try:
                vectors = embeddings.embed_documents([m.content for m in saved])
                for m, vec in zip(saved, vectors, strict=False):
                    vector_store.add_memory(
                        m.id,
                        user_id,
                        m.content,
                        vec,
                        importance=m.importance,
                        mtype=m.type,
                    )
            except Exception as e:  # noqa: BLE001
                # 建索引失败不该丢数据：记忆已在 SQLite 里，检索会走降级方案
                logger.error(f"建立记忆向量索引失败（记忆已入库，检索将降级）：{e}")

    # 档案同步
    if saved:
        await _sync_profile(user_id, [i for i in items if i.type in PROFILE_TYPES])

    logger.info(
        f"记忆写入 user={user_id} 新增={stats['new']} "
        f"精确重复={stats['exact_skip']} 语义重复={stats['semantic_skip']}"
    )
    return saved


async def _sync_profile(user_id: str, fact_items: list[MemoryItem]) -> None:
    """把事实类记忆同步到用户档案。

    冲突消解：同一个 key 的新事实会取代旧事实，但旧记录保留（is_active=0），
    这样能回答"用户的职业是怎么变化的"这类问题，也便于排查抽取错误。

    Day 5 修复：模型抽出的 fact 经常是"用户叫小林，是一名后端工程师，在杭州工作"
    这种一句话包含多个信息点的形式。早期版本把整句话塞进"姓名"字段，
    导致档案里姓名长得像一段简介。
    现在先把句子按逗号拆成小句，每个小句单独归类，各进各的字段。
    """
    for item in fact_items:
        for clause in split_fact_clauses(item.content):
            key = infer_key(clause)
            if not key:
                continue
            async with SessionLocal() as db:
                # 把同 key 的旧记录标记为失效
                old = (
                    await db.execute(
                        select(ProfileFact).where(
                            ProfileFact.user_id == user_id,
                            ProfileFact.key == key,
                            ProfileFact.is_active == 1,
                        )
                    )
                ).scalars().all()

                if any(o.value.strip() == clause for o in old):
                    continue  # 值没变，不用动

                for o in old:
                    o.is_active = 0

                db.add(
                    ProfileFact(
                        user_id=user_id,
                        key=key,
                        value=clause,
                        is_active=1,
                    )
                )
                await db.commit()


# 拆句用的分隔符
CLAUSE_SEPS = "，,；;。"


def split_fact_clauses(content: str) -> list[str]:
    """把一句话拆成若干小句。

    "用户叫小林，是一名后端工程师，在杭州工作"
      → ["用户叫小林", "是一名后端工程师", "在杭州工作"]

    为什么要拆？
    因为档案是"一件事一条"，而抽取器给出的一句话经常包含多个信息点。
    不拆的话，档案会变成一段话，既占 token 又无法单独更新
    （比如用户换了工作，整句话都得重写）。
    """
    parts = [content]
    for sep in CLAUSE_SEPS:
        next_parts: list[str] = []
        for p in parts:
            next_parts.extend(p.split(sep))
        parts = next_parts
    out = [p.strip() for p in parts if len(p.strip()) >= 3]
    return out or [content.strip()]


# 档案 key 的推断规则。按顺序匹配，命中即返回。
#
# 为什么不让模型直接给 key？
# 因为 key 必须稳定 —— 同一个含义每次都要映射到同一个 key。
# 模型措辞多变，会产出"姓名"/"名字"/"用户姓名"三个不同的 key，档案就乱了。
#
# 排序的讲究（踩过的坑）：
#   "在杭州工作" 这句话里同时有"工作"（职业关键词）和"杭州"（地点）。
#   早期版本职业规则在前，结果地点被误判成职业。
#   现在把"所在城市"放在职业前面，并且职业关键词里去掉了"工作"这个词
#   （因为"在XX工作"是地点表达，"是XX工程师"才是职业表达）。
KEY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("姓名", ("叫", "名字", "姓名")),
    ("所在城市", ("住在", "城市", "老家", "在上海", "在北京", "在杭州", "在深圳", "在广州", "在成都", "在南京", "在武汉", "在西安")),
    ("职业", ("职业", "工程师", "程序员", "设计师", "产品经理", "学生", "就读", "岗位", "后端", "前端", "全栈", "算法", "测试")),
    ("目标", ("想做", "目标是", "打算", "计划", "副业", "创业", "准备考")),
    ("偏好", ("喜欢", "讨厌", "偏好", "习惯")),
]


def infer_key(content: str) -> str | None:
    """从一句记忆里推断档案项名称。"""
    for key, words in KEY_RULES:
        if any(w in content for w in words):
            return key
    return None


async def get_profile(user_id: str) -> list[dict]:
    """取用户当前有效的档案项。"""
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(ProfileFact)
                .where(ProfileFact.user_id == user_id, ProfileFact.is_active == 1)
                .order_by(ProfileFact.key)
            )
        ).scalars().all()
    return [{"key": r.key, "value": r.value} for r in rows]


async def get_memories(
    user_id: str,
    *,
    limit: int = 20,
    min_importance: float = 0.0,
) -> list[Memory]:
    """按重要度取记忆。

    注意：**先用重要度取**（而不是时间），因为这是"最值得记住的事"。
    Day 5 会换成"语义 + 重要度 + 情绪"的加权检索。
    """
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Memory)
                .where(Memory.user_id == user_id, Memory.importance >= min_importance)
                .order_by(desc(Memory.importance), desc(Memory.created_at))
                .limit(limit)
            )
        ).scalars().all()
    return list(rows)


async def touch(memory_ids: list[int]) -> None:
    """记录这些记忆最近被召回的时间。

    用途：Day 5 算"记忆强度"——经常被用到的记忆衰减更慢（类似艾宾浩斯复习）。
    """
    if not memory_ids:
        return
    now = datetime.now(UTC).replace(tzinfo=None)
    async with SessionLocal() as db:
        rows = (
            await db.execute(select(Memory).where(Memory.id.in_(memory_ids)))
        ).scalars().all()
        for r in rows:
            r.last_access_at = now
        await db.commit()


async def count(user_id: str) -> int:
    async with SessionLocal() as db:
        result = await db.execute(
            select(func.count()).select_from(Memory).where(Memory.user_id == user_id)
        )
        return int(result.scalar_one())


async def clear_user(user_id: str) -> dict:
    """删除某个用户的全部记忆与档案（含向量索引）。合规要求，必须提供。"""
    from app.memory import vector_store

    async with SessionLocal() as db:
        m = await db.execute(delete(Memory).where(Memory.user_id == user_id))
        f = await db.execute(delete(ProfileFact).where(ProfileFact.user_id == user_id))
        await db.commit()

    # 向量索引同步清理，否则会出现"SQLite 删了但还能被检索到"的幽灵记忆
    vector_store.delete_user_memories(user_id)

    logger.info(f"清空用户 {user_id} 的记忆：memory={m.rowcount}, fact={f.rowcount}")
    return {"memories_deleted": m.rowcount or 0, "facts_deleted": f.rowcount or 0}
