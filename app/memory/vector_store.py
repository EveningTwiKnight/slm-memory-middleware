"""向量库封装（Chroma）。

为什么要把向量单独存一份，而不是每次都现场算？
  记忆有几千条时，每条都重新编码一遍太慢。
  向量库预先存好向量，并且用近似最近邻算法（HNSW）快速找最相似的几条。

为什么记忆同时存 SQLite 和 Chroma？
  分工不同：
    SQLite  存**完整信息 + 结构化字段**（时间、重要度、情绪），是"事实来源"
    Chroma  只存**向量 + 少量元数据**，是"索引"
  这是标准做法：向量库负责"找得到"，业务库负责"存得全"。
  冲突时以 SQLite 为准，向量库随时可以重建。

两个 collection 严格分开：
  memories      用户的长期记忆
  knowledge     上传的知识库文档
为什么要分开？面试会问"你怎么区分事实记忆和知识检索"——分开存就是答案。
"""

from functools import lru_cache

import chromadb

from app.core.config import settings
from app.core.logging import logger

COLLECTION_MEMORIES = "memories"
COLLECTION_KNOWLEDGE = "knowledge"


@lru_cache(maxsize=1)
def get_client():
    """Chroma 本地持久化客户端。数据落在 settings.chroma_dir。"""
    logger.info(f"初始化向量库：{settings.chroma_dir}")
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _collection(name: str):
    # hnsw:space=cosine 表示用余弦距离。我们的向量已归一化，配合起来最自然
    return get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ------------------------------------------------------------ 写入


def add_memory(
    memory_id: int,
    user_id: str,
    content: str,
    vector: list[float],
    *,
    importance: float = 0.5,
    mtype: str = "fact",
) -> None:
    """把一条记忆的向量写进索引。id 用 SQLite 的主键，两边靠它对上。"""
    _collection(COLLECTION_MEMORIES).upsert(
        ids=[f"m_{memory_id}"],
        documents=[content],
        embeddings=[vector],
        metadatas=[
            {
                "user_id": user_id,          # 用于按用户过滤，必须隔离
                "memory_id": memory_id,
                "importance": float(importance),
                "type": mtype,
            }
        ],
    )


def add_knowledge(
    doc_id: str,
    content: str,
    vector: list[float],
    *,
    source: str = "",
    chunk_index: int = 0,
) -> None:
    """把知识库的一个文本块写进索引。"""
    _collection(COLLECTION_KNOWLEDGE).upsert(
        ids=[doc_id],
        documents=[content],
        embeddings=[vector],
        metadatas=[{"source": source, "chunk_index": chunk_index}],
    )


# ------------------------------------------------------------ 检索


def search_memories(
    user_id: str,
    query_vector: list[float],
    *,
    top_k: int = 10,
) -> list[dict]:
    """在某个用户的记忆里做语义检索。

    返回 [{memory_id, content, similarity, importance, type}, ...]
    similarity 由余弦距离换算：similarity = 1 - distance
    """
    col = _collection(COLLECTION_MEMORIES)
    if col.count() == 0:
        return []

    try:
        res = col.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(col.count(), 1)),
            where={"user_id": user_id},   # 关键：用户之间绝不能串数据
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"记忆检索失败：{type(e).__name__}: {e}")
        return []

    out: list[dict] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for i, _id in enumerate(ids):
        meta = metas[i] or {}
        out.append(
            {
                "memory_id": meta.get("memory_id"),
                "content": docs[i] if i < len(docs) else "",
                "similarity": 1.0 - float(dists[i]) if i < len(dists) else 0.0,
                "importance": float(meta.get("importance", 0.5)),
                "type": meta.get("type", "fact"),
            }
        )
    return out


def search_knowledge(query_vector: list[float], *, top_k: int = 5) -> list[dict]:
    """在知识库里做语义检索。"""
    col = _collection(COLLECTION_KNOWLEDGE)
    if col.count() == 0:
        return []

    try:
        res = col.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"知识库检索失败：{type(e).__name__}: {e}")
        return []

    out: list[dict] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        out.append(
            {
                "content": doc,
                "similarity": 1.0 - float(dists[i]) if i < len(dists) else 0.0,
                "source": (meta or {}).get("source", ""),
            }
        )
    return out


# ------------------------------------------------------------ 维护


def delete_memories(memory_ids: list[int]) -> None:
    if not memory_ids:
        return
    try:
        _collection(COLLECTION_MEMORIES).delete(ids=[f"m_{i}" for i in memory_ids])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"删除记忆向量失败：{e}")


def delete_user_memories(user_id: str) -> None:
    """按用户删光向量。合规要求，必须提供。"""
    try:
        _collection(COLLECTION_MEMORIES).delete(where={"user_id": user_id})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"按用户删除向量失败：{e}")


def clear_knowledge() -> None:
    """清空知识库（重建索引时用）。"""
    try:
        get_client().delete_collection(COLLECTION_KNOWLEDGE)
        logger.info("知识库已清空")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"清空知识库时忽略：{e}")


def stats() -> dict:
    """看看两个库里各有多少条。调试用。"""
    mem = _collection(COLLECTION_MEMORIES)
    kb = _collection(COLLECTION_KNOWLEDGE)
    return {"memories": mem.count(), "knowledge": kb.count()}
