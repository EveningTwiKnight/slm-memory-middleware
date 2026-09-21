"""Day 3 验证：记忆抽取器。

分两部分：
  A. 离线测试（不需要 Key）：三层 JSON 解析兜底、去重、档案冲突消解
  B. 在线测试（需要 Key）：真实对话抽取，看抽出来的记忆像不像样

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m tests.test_extractor
    .venv\\Scripts\\python.exe -m tests.test_extractor --offline   # 只跑离线部分
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

from app.core import tasks  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.llm.openai_compat import get_llm  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.extractor import MemoryExtractor, infer_emotion_label, parse_memories  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# ============================================================ A. 离线测试
def test_parsing() -> None:
    print("\n" + "=" * 66)
    print("A. 离线测试：三层 JSON 解析兜底（不需要 API Key）")
    print("=" * 66)

    # --- 1. 标准输出 ---
    print("\n[1] 标准 JSON")
    r = parse_memories('{"memories":[{"content":"用户叫小明","type":"fact","importance":0.95}]}')
    check("解析出 1 条", len(r) == 1, f"实际 {len(r)}")
    check("内容正确", r and r[0].content == "用户叫小明")
    check("重要度正确", r and abs(r[0].importance - 0.95) < 1e-6)

    # --- 2. 带 markdown 代码块（小模型最常见的毛病）---
    print("\n[2] 包在 ```json 代码块里 —— 必须能剥掉")
    wrapped = '```json\n{"memories":[{"content":"用户是前端工程师","type":"fact","importance":0.9}]}\n```'
    r = parse_memories(wrapped)
    check("剥掉围栏后解析出 1 条", len(r) == 1, f"实际 {len(r)}")

    # --- 3. 前后一堆废话 ---
    print("\n[3] 前后有废话 —— 必须能抠出 JSON")
    noisy = (
        "好的，我已经分析了这轮对话，以下是我的提取结果：\n"
        '{"memories":[{"content":"用户想做宠物AI助手副业","type":"goal","importance":0.85}]}\n'
        "希望对你有帮助！如果需要我还可以继续分析。"
    )
    r = parse_memories(noisy)
    check("从废话里抠出 1 条", len(r) == 1, f"实际 {len(r)}")
    check("内容正确", r and "宠物" in r[0].content)

    # --- 4. 内容里含花括号（括号配对算法的考验）---
    print("\n[4] content 里含花括号 —— 简单正则会被骗，括号配对才对")
    tricky = '{"memories":[{"content":"用户的代码里用了 {a:1} 这种写法","type":"fact","importance":0.5}]}'
    r = parse_memories(tricky)
    check("正确解析出 1 条", len(r) == 1, f"实际 {len(r)}: {[i.content for i in r]}")

    # --- 5. 完全不是 JSON ---
    print("\n[5] 完全不是 JSON —— 必须安全返回空列表，不能抛异常")
    r = parse_memories("我无法完成这个任务。")
    check("返回空列表而不是报错", r == [], f"实际 {r}")

    # --- 6. 空输出 ---
    print("\n[6] 空输出")
    check("空字符串返回空列表", parse_memories("") == [])
    check("None 输入也安全", parse_memories(None) == [])

    # --- 7. 空记忆数组（正常情况：这轮没值得记的）---
    print("\n[7] 模型判断'没什么值得记的'")
    r = parse_memories('{"memories":[]}')
    check("返回空列表", r == [], f"实际 {r}")

    # --- 8. 模型返回裸数组（不听话的另一种形式）---
    print("\n[8] 返回裸数组而不是 {memories:[...]}")
    r = parse_memories('[{"content":"用户住在杭州","type":"fact","importance":0.7}]')
    check("也能解析出来", len(r) == 1, f"实际 {len(r)}")

    # --- 9. 越界数值与非法类型 ---
    print("\n[9] 越界数值与非法类型 —— 要夹回合法区间，而不是丢弃整条")
    r = parse_memories(
        '{"memories":[{"content":"用户很兴奋","type":"心情",'
        '"importance":1.8,"valence":-3,"arousal":"0.9"}]}'
    )
    check("解析出 1 条（没被丢）", len(r) == 1, f"实际 {len(r)}")
    if r:
        check("importance 被夹到 1.0", r[0].importance == 1.0, f"实际 {r[0].importance}")
        check("valence 被夹到 -1.0", r[0].valence == -1.0, f"实际 {r[0].valence}")
        check("非法 type 兜回 fact", r[0].type == "fact", f"实际 {r[0].type}")
        check("字符串 arousal 被转成数字", r[0].arousal == 0.9, f"实际 {r[0].arousal}")

    # --- 10. 内容太短要丢掉 ---
    print("\n[10] 空白/超短内容要丢掉")
    r = parse_memories('{"memories":[{"content":"","importance":0.9},{"content":"啊","importance":0.9}]}')
    check("空与单字内容被过滤", r == [], f"实际 {[i.content for i in r]}")

    # --- 11. 情绪标签推断（用数值反推，保证标签体系稳定）---
    print("\n[11] 情绪标签推断（效价-唤醒二维模型）")
    cases = [
        (-0.7, 0.7, "焦虑"),
        (-0.7, 0.95, "愤怒"),
        (-0.7, 0.2, "低落"),
        (0.6, 0.8, "兴奋"),
        (0.6, 0.1, "平静"),
        (0.0, 0.3, "中性"),
    ]
    for v, a, expect in cases:
        got = infer_emotion_label(v, a)
        check(f"valence={v} arousal={a} → {expect}", got == expect, f"实际 {got}")


async def test_storage() -> None:
    print("\n" + "=" * 66)
    print("A2. 离线测试：写入去重与档案冲突消解")
    print("=" * 66)

    user_id = f"u_store_{uuid.uuid4().hex[:6]}"

    from app.memory.extractor import MemoryItem

    print("\n[12] 重复写入同一条记忆 —— 不该产生两条")
    items = [MemoryItem(content="用户叫小明", type="fact", importance=0.9)]
    s1 = await long_term.save_memories(user_id, items)
    s2 = await long_term.save_memories(user_id, items)
    check("第一次写入 1 条", len(s1) == 1, f"实际 {len(s1)}")
    check("第二次因重复被跳过", len(s2) == 0, f"实际 {len(s2)}")
    check("库里确实只有 1 条", await long_term.count(user_id) == 1)

    print("\n[13] 档案冲突消解：职业变了，旧值失效但保留")
    await long_term.save_memories(
        user_id, [MemoryItem(content="用户的职业是前端工程师", type="fact", importance=0.9)]
    )
    await long_term.save_memories(
        user_id, [MemoryItem(content="用户已经转岗做后端工程师", type="fact", importance=0.9)]
    )
    profile = await long_term.get_profile(user_id)
    check("档案里只有 1 条有效职业", len([p for p in profile if p["key"] == "职业"]) == 1,
          f"实际 {profile}")

    print("\n[14] 低重要度过滤")
    await long_term.save_memories(user_id, [MemoryItem(content="用户明天可能下雨带伞", type="other", importance=0.1)])
    all_m = await long_term.get_memories(user_id, min_importance=0.0)
    high_m = await long_term.get_memories(user_id, min_importance=0.5)
    check("低分记忆默认也存着", len(all_m) >= 3, f"实际 {len(all_m)}")
    check("可以按重要度筛掉", len(high_m) < len(all_m), f"高分 {len(high_m)} < 全部 {len(all_m)}")

    print("\n[15] 按重要度排序")
    contents = [m.content for m in all_m]
    if all_m:
        check("第一条是最高分", all_m[0].importance >= all_m[-1].importance,
              f"{all_m[0].importance} >= {all_m[-1].importance}")

    await long_term.clear_user(user_id)


# ============================================================ B. 在线测试
async def test_live() -> None:
    print("\n" + "=" * 66)
    print("B. 在线测试：真实对话抽取（需要 API Key）")
    print("=" * 66)

    if not settings.llm_api_key:
        print(f"  {FAIL} 没有 API Key，跳过在线测试")
        checks.append(("在线抽取", False, "缺 API Key"))
        return

    extractor = MemoryExtractor(get_llm())

    # 应该抽出记忆的对话
    cases = [
        (
            "自我介绍",
            "你好，我叫小明，是个前端工程师，现在在杭州",
            "你好小明！前端工程师在杭州挺多的，那边互联网氛围不错。",
            True,
        ),
        (
            "目标与情绪",
            "我最近特别焦虑，想做个宠物相关的副业但一直没动手，怕自己坚持不下来",
            "焦虑很正常，说明你在意这件事。要不先定一个这周就能完成的小目标？",
            True,
        ),
        (
            "纯闲聊（不该抽出任何记忆）",
            "今天天气真好啊",
            "是的，适合出去走走，晒晒太阳心情会好很多。",
            False,
        ),
        (
            "问常识（不该抽出任何记忆）",
            "什么是向量数据库？",
            "向量数据库是专门存储和检索向量数据的数据库，支持相似度搜索，常用于 RAG 场景。",
            False,
        ),
    ]

    user_id = f"u_live_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    for name, umsg, amsg, expect_any in cases:
        print(f"\n[在线] {name}")
        print(f"  用户：{umsg}")
        items = await extractor.extract(umsg, amsg, min_importance=0.0)
        if items:
            for it in items:
                print(
                    f"    → [{it.type}] 重要度{it.importance:.2f} "
                    f"情绪={it.emotion_label}({it.valence:+.2f}) {it.content}"
                )
        else:
            print("    → （没有抽取到记忆）")

        if expect_any:
            check(f"{name}：抽取到了记忆", len(items) >= 1, f"实际 {len(items)} 条")
        else:
            check(f"{name}：正确地没有硬凑记忆", len(items) == 0, f"实际 {len(items)} 条：{[i.content for i in items]}")

        if items:
            await long_term.save_memories(user_id, items)

    # 去重效果
    print("\n[在线] 去重：把同一段对话再抽一遍，不该新增")
    before = await long_term.count(user_id)
    items = await extractor.extract(cases[0][1], cases[0][2], min_importance=0.0)
    await long_term.save_memories(user_id, items)
    after = await long_term.count(user_id)
    check("重复抽取没有新增记忆", after == before, f"{before} → {after}")

    print(f"\n用户 {user_id} 最终记忆库：{after} 条")
    for m in await long_term.get_memories(user_id, limit=20):
        print(f"    [{m.type}] {m.importance:.2f} {m.content}")

    # 清理
    await long_term.clear_user(user_id)


async def main() -> int:
    offline_only = "--offline" in sys.argv
    await init_db()

    test_parsing()
    await test_storage()

    if not offline_only:
        await test_live()
    await tasks.wait_all()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Day 3 完成！记忆抽取的解析兜底、去重、档案冲突消解全部正常。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
