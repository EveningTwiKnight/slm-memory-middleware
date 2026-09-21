"""验证意图判断：无关问题不检索记忆，相关问题照常检索。

这个测试针对的是一个真实反馈：
用户问天气、附近美食时，界面一直显示"注入了 5 条记忆"，让人以为
中间件在乱存东西。实际原因是**每轮都做检索**，跟问题相不相关都注入。

修法是先判断"这句话是否关于用户本人"，无关就不检索。

用法：
    .venv\\Scripts\\python.exe -m tests.test_intent
"""

import asyncio
import sys

sys.path.insert(0, ".")

from app.core.config import settings  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.orchestration import prompt_builder as pb  # noqa: E402
from app.orchestration.intent import is_about_user  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# (问题, 是否应该被判定为"关于用户本人")
CASES = [
    # ---- 应该判定为相关（需要记忆）----
    ("我叫什么？", True),
    ("你还记得我是谁吗", True),
    ("我最近在做什么项目", True),
    ("我平时喜欢怎么推进工作", True),
    ("我主要担心什么", True),
    ("我用什么技术栈", True),
    ("我上次说的那个想法", True),
    ("我最近状态怎么样", True),
    ("我的目标是什么", True),
    ("我之前提过什么", True),
    # ---- 应该判定为无关（不需要记忆）----
    ("今天天气怎么样", False),
    ("附近有什么好吃的", False),
    ("推荐一部电影", False),
    ("介绍一下宋朝历史", False),
    ("1+1 等于几", False),
    ("现在几点了", False),
    ("帮我把这段话翻译成英文", False),
    ("什么是向量数据库", False),
    ("帮我写个快排", False),
    ("怎么缓解焦虑", False),
    ("红烧肉怎么做", False),
]

# 已知的边界情况：记录下来而不是假装准确率 100%。
KNOWN_EDGE_CASES = {
    "怎么缓解焦虑": "命中通用问法「怎么」判为不检索。若用户本意是聊自己的焦虑，"
                    "这里会少给一点背景 —— 属于通用问法与个人问法的天然模糊地带。",
}


def test_rules() -> None:
    print("=" * 74)
    print("一、规则判断准确性")
    print("=" * 74)

    correct = 0
    false_positives: list[str] = []
    false_negatives: list[str] = []
    for q, expect in CASES:
        got, reason = is_about_user(q)
        ok = got == expect
        if ok:
            correct += 1
        elif got and not expect:
            false_positives.append(q)
        else:
            false_negatives.append(q)
        tag = "需检索" if expect else "不检索"
        mark = OK if ok else FAIL
        print(f"  {mark} {q:<24} → {'需检索' if got else '不检索':<6} [{tag}]  {reason}")

    total = len(CASES)
    print()
    check(
        f"规则准确率 {correct}/{total}",
        correct == total,
        f"实际错 {total - correct} 例",
    )
    # 关键：不能有"漏判"——把本该检索的判成不检索，等于失忆
    check(
        "没有漏判（相关句全部判为需要检索）",
        not false_negatives,
        f"漏了：{false_negatives}" if false_negatives else "无漏判",
    )
    # 反向：不能有误判（把无关问题判成需要检索，会白烧 token 并干扰模型）
    check(
        "没有误判（无关句全部判为不需要检索）",
        not false_positives,
        f"误判：{false_positives}" if false_positives else "无误判",
    )

    if KNOWN_EDGE_CASES:
        print()
        print("  边界情况说明：")
        for q, note in KNOWN_EDGE_CASES.items():
            print(f"    · {q}：{note}")


async def test_pipeline() -> None:
    print()
    print("=" * 74)
    print("二、接入流程后的实际效果")
    print("=" * 74)

    await init_db()
    uid = "u_intent_test"
    await long_term.clear_user(uid)
    await long_term.save_memories(
        uid,
        [
            MemoryItem("林知远是算法工程师，专做推荐系统方向", "fact", 0.95),
            MemoryItem("用户正在开发一个面向中小商家的选品推荐工具", "goal", 0.90),
            MemoryItem("用户因为项目进度慢而焦虑", "emotion", 0.75),
            MemoryItem("用户喜欢用 Python 和 PyTorch", "preference", 0.50),
        ],
        semantic_dedupe=False,
    )

    print(f"\n  {'问题':<24}{'参考记忆':>10}{'档案':>8}{'prompt':>10}   判断")
    print("  " + "-" * 68)

    rows = []
    for q, expect_retrieval in CASES:
        bundle = await pb.assemble(
            user_id=uid,
            user_input=q,
            history=[],
            use_emotion=False,
        )
        rows.append((q, expect_retrieval, bundle.memories_used, bundle.est_prompt_tokens))
        print(
            f"  {q:<24}{bundle.memories_used:>10}{bundle.profile_used:>8}"
            f"{bundle.est_prompt_tokens:>10}   "
            f"{'检索' if bundle.memories_used else '未检索'}"
        )

    print()
    # 关键断言：无关问题的记忆注入必须是 0
    bad = [
        (q, n)
        for q, expect, n, _ in rows
        if not expect and n > 0
    ]
    check(
        "无关问题的记忆注入为 0",
        not bad,
        f"仍有注入：{bad}" if bad else "全部为 0",
    )

    # 相关问题必须注入到
    missed = [
        (q, n)
        for q, expect, n, _ in rows
        if expect and n == 0
    ]
    check(
        "相关问题的记忆正常注入",
        not missed,
        f"漏了：{missed}" if missed else "全部有注入",
    )

    # token 节省
    with_gate = sum(t for q, e, n, t in rows if not e)
    without_gate = sum(t for q, e, n, t in rows if not e and True)
    related_tokens = [t for q, e, n, t in rows if e]
    unrelated_tokens = [t for q, e, n, t in rows if not e]
    print()
    print(f"  相关问题的平均 prompt：{sum(related_tokens) // max(len(related_tokens), 1)} token")
    print(f"  无关问题的平均 prompt：{sum(unrelated_tokens) // max(len(unrelated_tokens), 1)} token")
    if related_tokens and unrelated_tokens:
        saved = 1 - (sum(unrelated_tokens) / len(unrelated_tokens)) / (
            sum(related_tokens) / len(related_tokens)
        )
        print(f"  → 无关问题比相关问题省 {saved:.0%} 的 prompt token")

    await long_term.clear_user(uid)


async def main() -> int:
    print()
    test_rules()
    await test_pipeline()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print()
    print("=" * 74)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} 意图判断生效：无关问题不再乱注入记忆。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
