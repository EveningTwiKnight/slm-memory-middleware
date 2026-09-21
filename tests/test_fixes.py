"""Day 5 补充验证：档案拆分与知识库门槛（两个修复的效果确认）。

这两个修复都是被实测数据逼出来的：
  1. infer_key 把一整句话塞进"姓名"字段 → 改成按小句拆分归类
  2. 知识库不过滤相关度，无关问题也注入 → 加相似度门槛

用法：
    .venv\\Scripts\\python.exe -m tests.test_fixes
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

from app.memory import long_term, vector_store  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.retrieval import memory_retriever  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


def test_split() -> None:
    print("\n" + "=" * 66)
    print("[1] 档案拆分：一句话包含多个信息点时要拆开归类")
    print("=" * 66)

    cases = [
        (
            "用户叫小林，是一名后端工程师，在杭州工作",
            {"姓名", "职业", "所在城市"},
        ),
        (
            "用户叫小明，是个前端工程师，最近想做宠物相关的副业，喜欢用 React",
            {"姓名", "职业", "目标", "偏好"},
        ),
        ("用户很焦虑", set()),
    ]

    for content, expect_keys in cases:
        clauses = long_term.split_fact_clauses(content)
        print(f"\n  原句：{content}")
        print(f"  拆成：{clauses}")

        keys = {long_term.infer_key(c) for c in clauses}
        keys.discard(None)

        if expect_keys:
            missing = expect_keys - keys
            check(
                f"归类覆盖 {expect_keys}",
                not missing,
                f"实际 {keys}" + (f"，缺 {missing}" if missing else ""),
            )
        else:
            check("情绪类不进档案（没有匹配的 key）", not keys, f"实际 {keys}")

    # 关键回归：姓名不该是一整句话
    clauses = long_term.split_fact_clauses("用户叫小林，是一名后端工程师，在杭州工作")
    name_clause = next((c for c in clauses if long_term.infer_key(c) == "姓名"), None)
    print(f"\n  姓名对应的内容：{name_clause!r}")
    check(
        "姓名不再是整段简介",
        name_clause is not None and len(name_clause) < 20,
        f"长度 {len(name_clause) if name_clause else 0}",
    )


async def test_profile_db() -> None:
    print("\n" + "=" * 66)
    print("[2] 档案落库：多个字段各自独立")
    print("=" * 66)

    user_id = f"u_prof_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    await long_term.save_memories(
        user_id,
        [
            MemoryItem(
                "用户叫小林，是一名后端工程师，在杭州工作",
                "fact",
                0.95,
            )
        ],
        semantic_dedupe=False,
    )

    profile = await long_term.get_profile(user_id)
    print("\n  档案内容：")
    for p in profile:
        print(f"    {p['key']}：{p['value']}")

    keys = {p["key"] for p in profile}
    check("生成了多个档案项", len(profile) >= 2, f"{len(profile)} 条：{keys}")
    check("没有把整句话当姓名", all(len(p["value"]) < 30 for p in profile),
          f"最长 {max((len(p['value']) for p in profile), default=0)} 字")

    # 用户换工作：旧职业失效，新职业生效，且旧值保留
    await long_term.save_memories(
        user_id,
        [MemoryItem("用户已经转岗做全栈工程师", "fact", 0.9)],
        semantic_dedupe=False,
    )
    profile2 = await long_term.get_profile(user_id)
    jobs = [p["value"] for p in profile2 if p["key"] == "职业"]
    print(f"\n  换工作后职业字段：{jobs}")
    check("职业只剩一条有效值", len(jobs) == 1, f"实际 {jobs}")

    await long_term.clear_user(user_id)


async def test_knowledge_threshold() -> None:
    print("\n" + "=" * 66)
    print("[3] 知识库门槛：相关问题注入，无关问题不注入")
    print("=" * 66)

    if vector_store.stats()["knowledge"] == 0:
        print(f"  {FAIL} 知识库为空，先跑：scripts/ingest_kb --dir docs/kb_sample --reset")
        return

    from app.core.config import settings

    print(f"  当前门槛：相似度 >= {settings.knowledge_min_similarity}\n")

    cases = [
        ("token 预算超了应该先裁剪哪部分？", True, "与文档强相关"),
        ("记忆的衰减半衰期是多少？", True, "与文档相关"),
        ("红烧肉怎么做？", False, "与文档无关"),
        ("今天天气怎么样？", False, "与文档无关"),
    ]

    for q, expect_knowledge, note in cases:
        hits = await memory_retriever.retrieve_knowledge(q, top_k=settings.knowledge_top_k)
        passed = [h for h in hits if h["similarity"] >= settings.knowledge_min_similarity]
        got = len(passed) > 0
        top_sim = hits[0]["similarity"] if hits else 0.0
        print(
            f"  {q[:26]:<28} 最高相似度 {top_sim:.3f} → "
            f"注入 {len(passed)} 条  {'（期望：注入）' if expect_knowledge else '（期望：不注入）'}"
        )
        check(f"{note}", got == expect_knowledge, f"实际 {'注入' if got else '不注入'}")


async def main() -> int:
    await init_db()
    test_split()
    await test_profile_db()
    await test_knowledge_threshold()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} 两个修复都生效了。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
