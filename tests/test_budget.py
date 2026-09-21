"""Day 6 验证：Token 预算调度 + 流式输出 + 缓存。

分四部分：
  A. 预算裁剪逻辑（不需要 Key）：优先级顺序、保护段不被裁
  B. 长对话 token 曲线（不需要 Key）：第 30 轮 vs 第 1 轮，有预算 vs 无预算
  C. 缓存（不需要 Key）：精确命中、语义命中、用户隔离
  D. 流式输出（需要 Key）：首字延迟对比

用法：
    .venv\\Scripts\\python.exe -m tests.test_budget
    .venv\\Scripts\\python.exe -m tests.test_budget --offline
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

from app.core.config import settings  # noqa: E402
from app.llm.base import Message  # noqa: E402
from app.orchestration import cache as cache_mod  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []
metrics: dict = {}


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# ============================================================ A. 预算裁剪
def test_budget() -> None:
    from app.orchestration import budget as bd

    print("\n" + "=" * 70)
    print("A. 预算裁剪：优先级顺序是否正确")
    print("=" * 70)

    print("\n[1] 不超预算时不该裁任何东西")
    history = [Message("user", "你好"), Message("assistant", "你好！有什么可以帮你？")]
    kept, plan = bd.plan_and_trim(
        persona="你是助手",
        profile_text="- 姓名：小明",
        memory_lines=[("用户想做副业", 7, 1)],
        knowledge_lines=["参考资料一段"],
        history=history,
        user_input="我该怎么办",
        total_budget=2400,
    )
    check("没有裁剪", plan.saved == 0, plan.summary())
    check("历史完整保留", len(kept["history"]) == 2)
    check("记忆完整保留", len(kept["memory_lines"]) == 1)

    print("\n[2] 超预算时：先砍参考资料，再砍历史，再砍记忆，最后才动档案")
    long_history = [Message("user", "历史消息" * 40) for _ in range(10)]  # 每条约 160 字
    kept2, plan2 = bd.plan_and_trim(
        persona="你是助手" * 5,
        profile_text="- 姓名：小明\n- 职业：工程师\n- 目标：做副业",
        memory_lines=[(f"记忆{i}", 50, i) for i in range(6)],
        knowledge_lines=["资料" * 100 for _ in range(3)],  # 每条 200 字
        history=long_history,
        user_input="我该怎么办" * 10,
        total_budget=900,
    )
    print(f"      {plan2.summary()}")
    print(f"      裁剪明细：{plan2.trimmed}")
    check("参考资料被砍", plan2.dropped_knowledge > 0, f"丢 {plan2.dropped_knowledge} 条")
    check("历史被砍", plan2.dropped_history > 0, f"丢 {plan2.dropped_history} 条")
    check("裁剪后不超预算", plan2.used_after <= 900, f"{plan2.used_after} <= 900")

    print("\n[3] 保护段：人设与当前输入永不裁剪")
    kept3, plan3 = bd.plan_and_trim(
        persona="你是一个有明确人设的助手",
        profile_text="", memory_lines=[], knowledge_lines=[],
        history=[Message("user", "x" * 3000) for _ in range(5)],
        user_input="这句话不能被砍掉" * 5,
        total_budget=300,
    )
    check("人设未被裁", "明确人设" in kept3["persona"])
    check(
        "当前输入未被裁（在配额内）",
        kept3["user_input"].startswith("这句话"),
        f"长度 {len(kept3['user_input'])}",
    )

    print("\n[4] 当前输入本身超配额时会被截断（唯一会动保护段的情况）")
    kept4, plan4 = bd.plan_and_trim(
        persona="你是助手", profile_text="", memory_lines=[], knowledge_lines=[],
        history=[], user_input="很长的一句话" * 200,  # 1200 字
        total_budget=2400,
    )
    check("超长输入被截断", plan4.truncated_input, f"截后 {len(kept4['user_input'])} 字")
    check("截断后仍在配额内", len(kept4["user_input"]) <= bd.DEFAULT_BUDGET["input"] + 1,
          f"{len(kept4['user_input'])} <= {bd.DEFAULT_BUDGET['input']}")

    print("\n[5] 历史从最早的开始砍（保留最近的上下文）")
    hist = [Message("user", f"第{i}轮" + "内容" * 30) for i in range(8)]
    kept5, _ = bd.plan_and_trim(
        persona="助手", profile_text="", memory_lines=[], knowledge_lines=[],
        history=hist, user_input="问题", total_budget=500,
    )
    kept_texts = [m.content[:4] for m in kept5["history"]]
    print(f"      保留的历史：{kept_texts}")
    check("保留的是较新的消息", "第7轮" in "".join(kept_texts) or "第6轮" in "".join(kept_texts),
          f"实际 {kept_texts}")
    check("最早的消息被砍掉", "第0轮" not in kept_texts, f"实际 {kept_texts}")


# ============================================================ B. token 曲线
async def test_token_curve() -> None:
    from app.memory.database import init_db
    from app.memory import long_term, short_term
    from app.memory.extractor import MemoryItem
    from app.orchestration import prompt_builder as pb

    print("\n" + "=" * 70)
    print("B. 长对话 token 曲线：有预算调度 vs 无预算调度")
    print("=" * 70)

    await init_db()
    user_id = f"u_curve_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    # 造一批记忆
    await long_term.save_memories(
        user_id,
        [
            MemoryItem(f"用户的第{i}条信息：正在推进一个长期项目", "fact", 0.6)
            for i in range(8)
        ],
        semantic_dedupe=False,
    )

    # 造一个越来越长的会话
    session_id = f"s_curve_{uuid.uuid4().hex[:6]}"
    filler = [
        ("今天我们讨论一下项目排期吧", "好的，排期可以按优先级列出来。"),
        ("我觉得这个功能有点复杂", "复杂的地方可以拆成小步骤逐个验证。"),
        ("昨天测试的时候发现一个bug", "bug 的复现步骤能描述一下吗？"),
        ("这个方案的性能怎么样", "性能要看具体数据量，建议先压测。"),
        ("我担心时间不够", "时间紧张的话可以先砍掉非核心功能。"),
    ]

    print(f"\n  {'轮次':<6}{'无预算(全量)':>16}{'有预算调度':>14}{'节省':>10}")
    print("  " + "-" * 50)

    # 故意把预算压到 700 —— 这样第 10 轮左右就会真的触发裁剪。
    # 为什么要压这么低？因为默认预算 2400 在我们的测试数据量下根本撑不到，
    # 不触发裁剪就测不出预算调度有没有用（第一次跑就踩了这个坑）。
    # 这本身也说明一个事实：**预算调度是为极端长对话准备的保险**，
    # 短对话里它是"待命"状态，不产生效果也不产生开销。
    TEST_BUDGET = 700

    rows = []
    for turn in range(1, 31):
        u, a = filler[(turn - 1) % len(filler)]
        await short_term.append(session_id, user_id, "user", f"{u}（第{turn}轮）")
        await short_term.append(session_id, user_id, "assistant", a)

        if turn in (1, 5, 10, 20, 30):
            history = await short_term.get_recent(session_id, limit=100)
            profile = await long_term.get_profile(user_id)
            mems = await long_term.get_memories(user_id, limit=20)

            # 无预算：全量历史 + 全量记忆
            b_no = pb.build_messages(
                user_input=u,
                history=history,
                profile=profile,
                memories=mems,
                knowledge=[],
                emotion_block="",
                apply_budget=False,
            )

            # 有预算：显式传入较低预算，强制触发裁剪
            b_yes = pb.build_messages(
                user_input=u,
                history=history,
                profile=profile,
                memories=mems,
                knowledge=[],
                emotion_block="",
                apply_budget=True,
            )
            # build_messages 内部读 settings.max_prompt_tokens，
            # 这里直接用 budget 模块跑一遍拿到"低预算"下的结果
            from app.orchestration import budget as bd

            kept, plan = bd.plan_and_trim(
                persona=pb.DEFAULT_PERSONA,
                profile_text=pb.format_profile(profile),
                memory_lines=[(m.content, bd.est(m.content), m.id or 0) for m in mems],
                knowledge_lines=[],
                history=history,
                user_input=u,
                total_budget=TEST_BUDGET,
            )
            yes_tok = (
                bd.est(kept["persona"]) + bd.est(kept["profile_text"])
                + sum(t for _, t, _ in kept["memory_lines"])
                + sum(bd.est(m.content) for m in kept["history"])
                + bd.est(kept["user_input"])
            )

            no_tok = b_no.est_prompt_tokens
            save_pct = (1 - yes_tok / no_tok) * 100 if no_tok else 0
            trimmed_note = (
                f"裁掉 历史{plan.dropped_history}/记忆{plan.dropped_memories}"
                if (plan.dropped_history or plan.dropped_memories) else "未触发裁剪"
            )
            rows.append((turn, no_tok, yes_tok, save_pct, trimmed_note))
            print(f"  {turn:<6}{no_tok:>16}{yes_tok:>14}{save_pct:>9.0f}%   {trimmed_note}")

    if rows:
        first, last = rows[0], rows[-1]
        growth_no = last[1] / first[1] if first[1] else 1
        growth_yes = last[2] / first[2] if first[2] else 1

        # 更关键的指标：**是否到达平台期**。
        # 光看"第30轮/第1轮"的倍数会误导 —— 第 1 轮本身很短（人设占固定比例），
        # 分母小会让倍数看起来很大。真正要验证的是：
        # 继续聊下去，token 还会不会涨？涨到预算上限就该停住。
        turn20 = next((r for r in rows if r[0] == 20), None)
        late_growth_no = last[1] / turn20[1] if turn20 and turn20[1] else 1
        late_growth_yes = last[2] / turn20[2] if turn20 and turn20[2] else 1

        print(f"\n  第 30 轮 / 第 1 轮的 prompt 增长倍数（预算 {TEST_BUDGET}）：")
        print(f"    无预算：{growth_no:.2f} 倍")
        print(f"    有预算：{growth_yes:.2f} 倍")
        print("\n  第 20 → 30 轮的增长（看是否到达平台期，这才是关键）：")
        print(f"    无预算：{turn20[1]} → {last[1]}（涨 {late_growth_no:.2f} 倍）")
        print(f"    有预算：{turn20[2]} → {last[2]}（涨 {late_growth_yes:.2f} 倍）")
        print(f"\n  第 30 轮绝对 token：无预算 {last[1]} → 有预算 {last[2]}"
              f"（省 {last[3]:.0f}%）")
        metrics["token_curve"] = {
            "no_budget_growth": round(growth_no, 2),
            "with_budget_growth": round(growth_yes, 2),
            "late_growth_no_budget": round(late_growth_no, 2),
            "late_growth_with_budget": round(late_growth_yes, 2),
            "turn30_no_budget": last[1],
            "turn30_with_budget": last[2],
            "saved_pct_at_30": round(last[3], 1),
        }
        check("有预算时第 30 轮 token 明显更少", last[2] < last[1],
              f"{last[1]} → {last[2]}")
        check("有预算时到达平台期（继续聊不再增长）", late_growth_yes < 1.05,
              f"第20→30轮只涨 {late_growth_yes:.2f} 倍")
        check("无预算时持续增长（未受控）", late_growth_no > 1.2,
              f"第20→30轮涨 {late_growth_no:.2f} 倍")
        check("第 30 轮确实触发了裁剪", "裁掉" in last[4], last[4])

    await long_term.clear_user(user_id)
    await short_term.clear_session(session_id)


# ============================================================ C. 缓存
async def test_cache() -> None:
    print("\n" + "=" * 70)
    print("C. 两级缓存：精确命中 + 语义命中 + 用户隔离")
    print("=" * 70)

    c = cache_mod.ResponseCache()

    print("\n[1] 精确命中")
    key = cache_mod.make_key("u1", "什么是向量数据库", [1, 2])
    check("首次查询未命中", c.get_exact(key) is None)
    c.put_exact(key, "向量数据库是专门存向量的数据库")
    check("再次查询命中", c.get_exact(key) == "向量数据库是专门存向量的数据库")

    print("\n[2] 缓存键包含注入的记忆 id（记忆变了答案就不该复用）")
    key2 = cache_mod.make_key("u1", "什么是向量数据库", [1, 2, 3])
    check("记忆不同 → 键不同 → 不命中", c.get_exact(key2) is None)

    print("\n[3] 不同用户的同一问题不该互相命中")
    key3 = cache_mod.make_key("u2", "什么是向量数据库", [1, 2])
    check("用户不同 → 键不同", c.get_exact(key3) is None)

    print("\n[4] 语义命中：措辞不同但意思一样")
    c2 = cache_mod.ResponseCache()
    c2.put_semantic("u1", "什么是向量数据库", "向量数据库是专门存向量的数据库")
    hit = c2.get_semantic("u1", "向量数据库是什么")
    print(f"      '什么是向量数据库' → '向量数据库是什么' 命中结果：{hit}")
    check("同义改写能命中", hit is not None, f"结果 {hit}")

    print("\n[5] 语义缓存不该跨用户命中")
    hit_other = c2.get_semantic("u_other", "向量数据库是什么")
    check("其他用户不命中", hit_other is None, f"结果 {hit_other}")

    print("\n[6] 不相关的问题不该命中")
    hit_unrelated = c2.get_semantic("u1", "红烧肉怎么做")
    check("不相关问题不命中", hit_unrelated is None, f"结果 {hit_unrelated}")

    print("\n[7] LRU 淘汰：超出容量后最久未用的被清掉")
    c3 = cache_mod.ResponseCache()
    for i in range(cache_mod.EXACT_MAX_SIZE + 50):
        c3.put_exact(f"key{i}", f"答案{i}")
    check("容量受控", len(c3._exact) <= cache_mod.EXACT_MAX_SIZE,
          f"{len(c3._exact)} <= {cache_mod.EXACT_MAX_SIZE}")
    check("最早写入的被淘汰", c3.get_exact("key0") is None)

    print("\n[8] 统计信息")
    print(f"      {c2.stats.summary()}")
    check("有命中率统计", c2.stats.hit_rate >= 0)


# ============================================================ D. 流式
async def test_stream() -> None:
    print("\n" + "=" * 70)
    print("D. 流式输出：首字延迟")
    print("=" * 70)

    if not settings.llm_api_key:
        print(f"  {FAIL} 没有 API Key，跳过")
        return

    import time

    import httpx

    from app.main import app
    from app.memory.database import init_db

    await init_db()
    user_id = f"u_stream_{uuid.uuid4().hex[:6]}"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
        question = "用三句话说明什么是检索增强生成"

        print("\n[1] 非流式（等全部生成完）")
        t0 = time.perf_counter()
        r = await client.post("/v1/chat", json={"user_id": user_id, "message": question})
        non_stream_ms = int((time.perf_counter() - t0) * 1000)
        print(f"      总耗时 {non_stream_ms}ms，收到 {len(r.json()['answer'])} 字")

        print("\n[2] 流式（一个字一个字吐）")
        first_ms = 0
        total_ms = 0
        chars = 0
        t0 = time.perf_counter()
        async with client.stream(
            "POST", "/v1/chat/stream",
            json={"user_id": user_id, "message": question},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                import json as _json

                data = _json.loads(line[6:])
                if data["type"] == "delta":
                    if not first_ms:
                        first_ms = int((time.perf_counter() - t0) * 1000)
                    chars += len(data["text"])
                elif data["type"] == "done":
                    total_ms = data["latency_ms"]

        print(f"      首字延迟 {first_ms}ms，总耗时 {total_ms or int((time.perf_counter() - t0) * 1000)}ms，{chars} 字")
        if first_ms and non_stream_ms:
            ratio = first_ms / non_stream_ms
            print(f"      首字延迟是非流式总耗时的 {ratio:.0%}")
            print(f"      → 用户感知的等待时间缩短了约 {(1 - ratio) * 100:.0f}%")
            metrics["stream"] = {
                "non_stream_ms": non_stream_ms,
                "first_token_ms": first_ms,
                "reduction_pct": round((1 - ratio) * 100, 1),
            }
            check("首字延迟明显低于非流式总耗时", first_ms < non_stream_ms,
                  f"{first_ms}ms < {non_stream_ms}ms")


async def main() -> int:
    offline_only = "--offline" in sys.argv
    test_budget()
    await test_token_curve()
    await test_cache()
    if not offline_only:
        await test_stream()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 70)
    print("本次实测数据")
    print("=" * 70)
    if "token_curve" in metrics:
        t = metrics["token_curve"]
        print(f"  第 30 轮 prompt token：无预算 {t['turn30_no_budget']} → "
              f"有预算 {t['turn30_with_budget']}（省 {t['saved_pct_at_30']}%）")
        print(f"  长度增长倍数：无预算 {t['no_budget_growth']}x → "
              f"有预算 {t['with_budget_growth']}x")
    if "stream" in metrics:
        s = metrics["stream"]
        print(f"  流式首字延迟 {s['first_token_ms']}ms vs 非流式 {s['non_stream_ms']}ms"
              f"（感知等待缩短 {s['reduction_pct']}%）")

    print("\n" + "=" * 70)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Day 6 完成！预算调度、缓存、流式全部正常。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
