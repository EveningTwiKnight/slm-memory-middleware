"""回答缓存：把重复问题挡在模型调用之前。

两类缓存，价值递增：

  1. **精确缓存**：完全一样的问题 → 直接返回上次的答案。
     实现最简单（一个字典），但命中率低 —— 用户很少一字不差地重复提问。

  2. **语义缓存**：意思相近的问题 → 复用答案。
     "什么是向量数据库" 和 "向量数据库是什么" 应该命中同一条。
     用嵌入向量算相似度，超过阈值就算命中。

为什么值得做？
  - 模型调用是整条链路里最慢、最贵的一步（实测 400~13000ms，波动很大）
  - 缓存命中后延迟能压到几十毫秒
  - 常见问答场景（比如产品说明、FAQ）重复率其实很高

两个必须注意的坑：
  1. **缓存键要包含"上下文指纹"**，不能只用问题文字。
     同一个问题在不同用户、不同记忆状态下，正确答案可能不同
     （"我该怎么办" 对甲和乙的答案是两回事）。
     这里用 user_id + 问题 + 注入记忆 id 集合 做键。
  2. **语义缓存的阈值要保守**。
     把"我的副业该怎么办"和"我的副业还要不要做"判成同一个问题是很危险的 ——
     它们情绪倾向完全不同。宁可少命中，不可错命中。

缓存容量用简单的 LRU 控制，避免内存无限增长。
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.core.logging import logger

# 精确缓存的容量与存活时间
EXACT_MAX_SIZE = 500
EXACT_TTL_SECONDS = 3600

# 语义缓存的相似度阈值。
# 定得比较保守（比记忆去重的 0.80 更高），因为答案复用的风险大于记错一条记忆。
SEMANTIC_THRESHOLD = 0.95
SEMANTIC_MAX_SIZE = 200
SEMANTIC_TTL_SECONDS = 1800


@dataclass
class CacheEntry:
    answer: str
    created_at: float
    hits: int = 0
    embedding: list[float] | None = None
    meta: dict = field(default_factory=dict)

    def expired(self, ttl: int) -> bool:
        return (time.time() - self.created_at) > ttl


@dataclass
class CacheStats:
    exact_hit: int = 0
    semantic_hit: int = 0
    miss: int = 0
    stored: int = 0

    @property
    def total(self) -> int:
        return self.exact_hit + self.semantic_hit + self.miss

    @property
    def hit_rate(self) -> float:
        return (self.exact_hit + self.semantic_hit) / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"精确命中 {self.exact_hit} / 语义命中 {self.semantic_hit} / "
            f"未命中 {self.miss}，命中率 {self.hit_rate:.1%}"
        )


def make_key(user_id: str, question: str, memory_ids: list[int]) -> str:
    """生成精确缓存键。

    为什么要把 memory_ids 拼进来？
    因为"同一个问题在不同记忆状态下，正确答案可能不同"。
    比如用户先说"我不喜欢猫"，后来说"我喜欢猫"，
    这两次问"我养什么宠物好"的答案应该不一样。
    把注入的记忆 id 作为指纹，能避免返回过期答案。
    """
    raw = f"{user_id}|{question.strip()}|{','.join(str(i) for i in sorted(memory_ids))}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class ResponseCache:
    """两级回答缓存。进程内实现（重启即失效）。

    为什么不做成 Redis？
    起步阶段没必要。进程内缓存已经能覆盖大部分收益，而且没有额外依赖。
    真要上多实例时，把这里换成 Redis 客户端即可 —— 接口不变。
    """

    def __init__(self) -> None:
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._semantic: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()

    # ---------------- 精确缓存 ----------------

    def get_exact(self, key: str) -> str | None:
        entry = self._exact.get(key)
        if entry is None:
            return None
        if entry.expired(EXACT_TTL_SECONDS):
            self._exact.pop(key, None)
            return None
        entry.hits += 1
        self._exact.move_to_end(key)  # LRU：命中的移到末尾
        self.stats.exact_hit += 1
        return entry.answer

    def put_exact(self, key: str, answer: str, meta: dict | None = None) -> None:
        self._exact[key] = CacheEntry(answer=answer, created_at=time.time(), meta=meta or {})
        self._exact.move_to_end(key)
        while len(self._exact) > EXACT_MAX_SIZE:
            self._exact.popitem(last=False)  # 淘汰最久未用的
        self.stats.stored += 1

    # ---------------- 语义缓存 ----------------

    def get_semantic(self, user_id: str, question: str) -> str | None:
        """在语义缓存里找相近问题。

        注意：这里**只比较同一用户**的缓存。
        不同用户的相似问题不该互相命中 —— 他们的记忆和情绪状态不同。
        """
        if not self._semantic:
            return None
        try:
            from app.memory import embeddings

            qv = embeddings.embed_query(question)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"语义缓存跳过（向量化失败）：{e}")
            return None

        best_key = None
        best_sim = 0.0
        for key, entry in self._semantic.items():
            if entry.expired(SEMANTIC_TTL_SECONDS):
                continue
            if entry.meta.get("user_id") != user_id:
                continue
            if entry.embedding is None:
                continue
            sim = embeddings.similarity(qv, entry.embedding)
            if sim > best_sim:
                best_sim, best_key = sim, key

        if best_key and best_sim >= SEMANTIC_THRESHOLD:
            entry = self._semantic[best_key]
            entry.hits += 1
            self._semantic.move_to_end(best_key)
            self.stats.semantic_hit += 1
            logger.debug(f"语义缓存命中（相似度 {best_sim:.3f}）：{question[:20]}")
            return entry.answer
        return None

    def put_semantic(self, user_id: str, question: str, answer: str) -> None:
        if len(question) < 6:
            return  # 太短的问题（"嗯"/"好的"）不做语义缓存
        try:
            from app.memory import embeddings

            vec = embeddings.embed_query(question)
        except Exception:  # noqa: BLE001
            return

        key = hashlib.sha256(f"{user_id}|{question}".encode()).hexdigest()[:32]
        self._semantic[key] = CacheEntry(
            answer=answer,
            created_at=time.time(),
            embedding=vec,
            meta={"user_id": user_id, "question": question},
        )
        self._semantic.move_to_end(key)
        while len(self._semantic) > SEMANTIC_MAX_SIZE:
            self._semantic.popitem(last=False)

    # ---------------- 维护 ----------------

    def note_miss(self) -> None:
        self.stats.miss += 1

    def clear(self) -> None:
        self._exact.clear()
        self._semantic.clear()

    def info(self) -> dict:
        return {
            "exact_size": len(self._exact),
            "semantic_size": len(self._semantic),
            "stats": {
                "exact_hit": self.stats.exact_hit,
                "semantic_hit": self.stats.semantic_hit,
                "miss": self.stats.miss,
                "hit_rate": round(self.stats.hit_rate, 4),
            },
        }


# 全局单例
_cache = ResponseCache()


def get_cache() -> ResponseCache:
    return _cache
