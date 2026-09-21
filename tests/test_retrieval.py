"""Day 5 验证：语义检索、语义去重、知识库 RAG。

这是全项目最能出"数字"的一份测试，因为它做的是**改动前后的对比**：
  改动前（Day 4）：取重要度最高的 5 条记忆
  改动后（Day 5）：用当前问题做语义检索，再叠加时间与重要度重排

用法：
    .venv\\Scripts\\python.exe -m tests.test_retrieval
    .venv\\Scripts\\python.exe -m tests.test_retrieval --offline   # 只跑不需要网络的
"""

import asyncio
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from app.core.config import settings  # noqa: E402
from app.memory import embeddings, long_term, vector_store  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.retrieval import memory_retriever  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []
metrics: dict[str, object] = {}


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# 一组刻意设计的记忆：既有"重要但跟问题无关"的，也有"不太重要但正好相关"的
# 用来验证检索是不是真的按相关性在排
SEED_MEMORIES: list[tuple[str, str, float]] = [
    ("用户叫小明", "fact", 0.95),
    ("用户是前端工程师", "fact", 0.90),
    ("用户在杭州工作", "fact", 0.60),
    ("用户想做宠物相关的副业", "goal", 0.85),
    ("用户担心自己坚持不下来，容易半途而废", "emotion", 0.70),
    ("用户喜欢用 React 和 TypeScript", "preference", 0.55),
    ("用户养了一只叫豆豆的柯基", "fact", 0.50),
    ("用户上周去看了牙医，补了两颗牙", "event", 0.40),
    ("用户的妈妈最近身体不太好", "fact", 0.75),
    ("用户计划今年考驾照", "goal", 0.45),
]


async def test_embeddings() -> None:
    print("\n" + "=" * 66)
    print("[1] 嵌入模型：让「意思相近」可以被计算")
    print("=" * 66)

    pairs = [
        ("用户担心自己坚持不下来", "用户担心自己无法坚持副业计划", "同义不同措辞", "high"),
        ("用户想做一个宠物相关的副业", "用户计划开展宠物领域的副业项目", "同义不同措辞", "high"),
        ("用户是前端工程师", "用户在做牙医手术", "完全无关", "low"),
        ("用户喜欢用 React", "用户偏好使用 React 框架", "同义不同措辞", "high"),
    ]

    print(f"\n  {'句子A':<22} {'句子B':<24} {'相似度':>8}  判定")
    print("  " + "-" * 70)
    for a, b, label, expect in pairs:
        va = embeddings.embed_documents([a])[0]
        vb = embeddings.embed_documents([b])[0]
        sim = embeddings.similarity(va, vb)
        print(f"  {a[:20]:<22} {b[:22]:<24} {sim:>8.4f}  {label}")
        if expect == "high":
            check(f"同义句相似度应偏高（{sim:.3f}）", sim >= 0.7, f"{sim:.4f}")

    # 记录一对"day4 发现的问题"的实际相似度，这就是阈值的依据
    v1 = embeddings.embed_documents(["用户担心自己坚持不下来"])[0]
    v2 = embeddings.embed_documents(["用户担心自己无法坚持副业计划"])[0]
    sim = embeddings.similarity(v1, v2)
    metrics["dedupe_pair_similarity"] = round(sim, 4)
    print(f"\n  ★ Day4 发现的重复对，实际相似度 = {sim:.4f}")
    print(f"    （当前去重阈值 {settings.semantic_dedupe_threshold}）")


async def test_dedupe() -> None:
    print("\n" + "=" * 66)
    print("[2] 语义去重：Day4 的缺陷是否修复")
    print("=" * 66)

    user_id = f"u_dedupe_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    # 第一条
    await long_term.save_memories(
        user_id,
        [MemoryItem("用户担心自己坚持不下来", "emotion", 0.7, valence=-0.4, arousal=0.5)],
    )
    n1 = await long_term.count(user_id)

    # 第二条：意思相同、措辞不同 —— 应该被拦下
    saved = await long_term.save_memories(
        user_id,
        [MemoryItem("用户担心自己无法坚持副业计划", "emotion", 0.7, valence=-0.4, arousal=0.5)],
    )
    n2 = await long_term.count(user_id)

    print(f"  写入第一条后：{n1} 条")
    print(f"  写入同义第二条后：{n2} 条（新增 {len(saved)} 条）")
    check("语义重复被拦下", len(saved) == 0 and n2 == n1, f"{n1} → {n2}")

    # 第三条：只是相关但不是同一件事 —— 必须能写进去，不能误杀
    saved2 = await long_term.save_memories(
        user_id,
        [MemoryItem("用户想做宠物相关的副业", "goal", 0.85)],
    )
    n3 = await long_term.count(user_id)
    print(f"  写入相关但不同的第三条后：{n3} 条")
    check("不同内容没被误杀", len(saved2) == 1 and n3 == n2 + 1, f"{n2} → {n3}")

    metrics["dedupe"] = {"before": n1, "after_dup": n2, "after_new": n3}
    await long_term.clear_user(user_id)


async def test_retrieval_quality() -> None:
    print("\n" + "=" * 66)
    print("[3] 检索质量：按重要度取 vs 语义检索（核心对比）")
    print("=" * 66)

    user_id = f"u_retr_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    items = [
        MemoryItem(content=c, type=t, importance=imp, valence=0.0, arousal=0.0)
        for c, t, imp in SEED_MEMORIES
    ]
    await long_term.save_memories(user_id, items, semantic_dedupe=False)
    print(f"  已植入 {await long_term.count(user_id)} 条记忆作为测试集")

    # 每个问题标注"正确答案"—— 期望被召回的记忆关键词
    cases = [
        ("我那个副业该怎么起步？", ["宠物", "副业"], "问副业"),
        ("我平时写代码用什么技术栈比较好？", ["React", "TypeScript"], "问技术栈"),
        ("我家狗狗最近怎么样？", ["柯基", "豆豆"], "问宠物"),
        ("我妈妈的事你有什么建议吗？", ["妈妈"], "问家人"),
        ("我之前说的那个担心是什么？", ["坚持"], "问担忧"),
    ]

    print(f"\n  {'问题':<24} {'改动前(按重要度)':<26} {'改动后(语义检索)'}")
    print("  " + "-" * 76)

    better = same = worse = 0
    for question, keywords, label in cases:
        # 改动前的做法：按重要度取 Top-5
        old = await long_term.get_memories(user_id, limit=5)
        old_hit = any(any(k in m.content for k in keywords) for m in old)

        # 改动后的做法：语义检索
        new = await memory_retriever.retrieve_memories(user_id, question, top_k=5)
        new_top = new[0].memory.content if new else ""
        new_hit = any(any(k in s.memory.content for k in keywords) for s in new)

        old_mark = "命中" if old_hit else "未命中"
        new_mark = "命中" if new_hit else "未命中"

        if new_hit and not old_hit:
            better += 1
            delta = "↑ 改善"
        elif old_hit and not new_hit:
            worse += 1
            delta = "↓ 变差"
        else:
            same += 1
            delta = "= 相同"

        print(f"  {question[:22]:<24} {old_mark:<26} {new_mark}  {delta}")
        print(f"  {'':<24} └ 旧 Top1: {(old[0].content[:20] if old else '')}")
        print(f"  {'':<24} └ 新 Top1: {new_top[:20]} (sim={new[0].similarity:.3f})" if new else "")

    total = len(cases)

    # 改动前的命中率：按重要度取 Top-5
    old_total = 0
    for _q, kws, _label in cases:
        old = await long_term.get_memories(user_id, limit=5)
        if any(any(k in m.content for k in kws) for m in old):
            old_total += 1

    # 改动后的命中率：语义检索 Top-5
    new_total = 0
    for q, kws, _label in cases:
        res = await memory_retriever.retrieve_memories(user_id, q, top_k=5)
        if any(any(k in s.memory.content for k in kws) for s in res):
            new_total += 1

    print(f"\n  ★ 命中率：改动前 {old_total}/{total} → 改动后 {new_total}/{total}")
    metrics["recall_at_5"] = {
        "before_importance_only": f"{old_total}/{total}",
        "after_semantic": f"{new_total}/{total}",
        "improved_cases": better,
        "regressed_cases": worse,
    }

    check(
        "语义检索命中率不低于按重要度取",
        new_total >= old_total,
        f"{old_total}/{total} → {new_total}/{total}",
    )
    check("语义检索命中率达到 5/5", new_total == total, f"实际 {new_total}/{total}")

    # 记录一个具体的排序例子，面试时可以讲
    print("\n  【具体例子】问「我那个副业该怎么起步？」时，检索给出的排序：")
    res = await memory_retriever.retrieve_memories(user_id, "我那个副业该怎么起步？", top_k=5)
    for i, s in enumerate(res, 1):
        print(
            f"    {i}. score={s.score:.3f} (sim={s.similarity:.3f} 新鲜={s.recency:.3f} "
            f"重要={s.importance:.2f}) {s.memory.content}"
        )

    await long_term.clear_user(user_id)


async def test_recency() -> None:
    print("\n" + "=" * 66)
    print("[4] 时间衰减：旧记忆的权重如何变化")
    print("=" * 66)

    now = datetime.now()
    print(f"\n  {'年龄':<12} {'新鲜度':>8}")
    print("  " + "-" * 24)
    samples = [0, 1, 3, 7, 30, 90, 180, 365]
    vals = []
    for days in samples:
        r = memory_retriever.recency_score(now - timedelta(days=days), now)
        vals.append(r)
        print(f"  {str(days) + ' 天':<12} {r:>8.4f}")

    check("新鲜度单调递减", all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)))
    check("30 天衰减到约一半", abs(vals[4] - 0.5) < 0.02, f"实际 {vals[4]:.4f}")
    check("一年前接近于 0 但不为 0", 0 < vals[-1] < 0.01, f"实际 {vals[-1]:.4f}")
    metrics["recency_curve"] = {f"{d}天": round(v, 4) for d, v in zip(samples, vals, strict=False)}


async def test_knowledge() -> None:
    print("\n" + "=" * 66)
    print("[5] 知识库检索（RAG）")
    print("=" * 66)

    # 造两条临时知识
    docs = [
        (
            "kb_test_vec",
            "向量数据库是专门用于存储和检索高维向量的数据库，核心能力是近似最近邻搜索（ANN）。"
            "常见实现有 Chroma、Qdrant、Milvus、FAISS 等。选型时主要看数据规模、过滤能力和部署成本。",
            "技术笔记.md",
        ),
        (
            "kb_test_rag",
            "检索增强生成（RAG）的基本流程是：先把文档切块并向量化存入向量库，"
            "用户提问时用同样方式把问题向量化，检索出最相似的若干文本块，"
            "再把这些文本块作为参考资料拼进提示词，让模型基于资料回答。",
            "技术笔记.md",
        ),
        (
            "kb_test_unrelated",
            "红烧肉的做法：五花肉切块冷水下锅焯水，加冰糖炒糖色，加生抽老抽料酒，"
            "小火慢炖四十分钟，最后大火收汁。",
            "菜谱.md",
        ),
    ]
    for doc_id, content, src in docs:
        vec = embeddings.embed_documents([content])[0]
        vector_store.add_knowledge(doc_id, content, vec, source=src)

    print(f"  已植入 3 条测试知识（向量库现有 {vector_store.stats()['knowledge']} 块）")

    queries = [
        ("什么是向量数据库？", "向量数据库", "应该命中向量库那条"),
        ("RAG 是怎么工作的？", "检索增强生成", "应该命中 RAG 那条"),
        ("红烧肉怎么做？", "红烧肉", "应该命中菜谱那条"),
    ]

    for q, expect_kw, note in queries:
        res = await memory_retriever.retrieve_knowledge(q, top_k=2)
        top = res[0] if res else None
        hit = bool(top and expect_kw in top["content"])
        print(f"\n  问：{q}")
        if top:
            print(f"    Top1（相似度 {top['similarity']:.3f}，来源 {top['source']}）：{top['content'][:50]}…")
        check(f"{note}", hit, "命中" if hit else "未命中")

    # 清理测试知识
    for doc_id, _, _ in docs:
        try:
            vector_store._collection("knowledge").delete(ids=[doc_id])
        except Exception:  # noqa: BLE001
            pass


async def test_fallback() -> None:
    print("\n" + "=" * 66)
    print("[6] 降级链：检索不可用时不能整个对话失败")
    print("=" * 66)

    user_id = f"u_fb_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)
    await long_term.save_memories(
        user_id,
        [MemoryItem("用户叫小明", "fact", 0.95)],
        semantic_dedupe=False,
    )

    # 模拟向量库故障：直接查一个不存在的用户，向量库会返回空
    res = await memory_retriever.retrieve_memories("u_never_exist_xyz", "测试", top_k=3)
    check("没有数据时返回空列表而不是报错", res == [], f"实际 {len(res)} 条")

    # 模拟查询向量化失败
    original = embeddings.embed_query
    embeddings.embed_query = lambda text: (_ for _ in ()).throw(RuntimeError("模拟嵌入服务不可用"))
    try:
        res2 = await memory_retriever.retrieve_memories(user_id, "我叫什么", top_k=3)
        check("向量化失败时降级为按重要度取", len(res2) >= 1, f"实际 {len(res2)} 条")
        if res2:
            print(f"    降级取到：{res2[0].memory.content}（breakdown={res2[0].breakdown}）")
    finally:
        embeddings.embed_query = original

    await long_term.clear_user(user_id)


async def main() -> int:
    offline_only = "--offline" in sys.argv
    await init_db()

    await test_embeddings()
    await test_dedupe()
    await test_retrieval_quality()
    await test_recency()
    if not offline_only:
        await test_knowledge()
    await test_fallback()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)

    print("\n" + "=" * 66)
    print("本次实测数据（可直接用于简历/实验报告）")
    print("=" * 66)
    r = metrics.get("recall_at_5", {})
    if r:
        print(f"  Recall@5：{r['before_importance_only']} → {r['after_semantic']}")
        print(f"  改善 {r['improved_cases']} 例，变差 {r['regressed_cases']} 例")
    if "dedupe_pair_similarity" in metrics:
        print(f"  重复记忆对的相似度：{metrics['dedupe_pair_similarity']}")
    if "dedupe" in metrics:
        d = metrics["dedupe"]  # type: ignore[assignment]
        print(f"  语义去重：写入重复后条数 {d['before']} → {d['after_dup']}（未增长）")

    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Day 5 完成！语义检索、语义去重、知识库 RAG 全部正常。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
