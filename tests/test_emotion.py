"""Day 5 后半验证：情感记忆模块。

四部分：
  A. 情绪打分（词典方法的准确性与边界）
  B. 情绪状态维护（EMA 平滑、时间衰减、趋势检测）
  C. 情感加权召回（消融对比：开/关情感加权，排序差异）
  D. 情绪驱动的回复姿态（真模型对比同一问题在不同情绪下的回答）

用法：
    .venv\\Scripts\\python.exe -m tests.test_emotion
    .venv\\Scripts\\python.exe -m tests.test_emotion --offline   # 跳过真模型部分
"""

import asyncio
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from app.memory import emotion as emo  # noqa: E402
from app.memory import emotion_store, long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.memory.models import Memory  # noqa: E402
from app.retrieval import memory_retriever as mr  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []
metrics: dict = {}


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# ============================================================ A. 情绪打分
def test_analyze() -> None:
    print("\n" + "=" * 68)
    print("A. 情绪打分：词典方法的准确性与边界")
    print("=" * 68)

    cases = [
        # (文本, 期望效价方向, 期望标签, 说明)
        ("我最近特别焦虑，怕做不成", "neg", "焦虑", "典型焦虑"),
        ("烦死了，什么都不想干", "neg", "焦虑", "烦躁"),
        ("今天心情不错，挺开心的", "pos", "兴奋", "中等偏强正面（两个正面词）"),
        ("我拿到 offer 了！太开心了！！", "pos", "兴奋", "高唤醒正面"),
        ("有点累，提不起劲", "neg", "低落", "低唤醒负面"),
        ("我好难过，感觉没希望了", "neg", None, "低落/绝望（标签只查方向）"),
        ("我太生气了，简直受不了", "neg", "愤怒", "愤怒"),
        ("今天天气不错", "pos_mild", "平静", "轻微正面陈述（不该判成强正面）"),
        ("嗯", "neutral", "中性", "极短输入"),
        ("我不知道该怎么办", "neutral", "中性", "无情绪词"),
    ]

    for text, direction, expect_label, note in cases:
        s = emo.analyze(text)
        ok = True
        detail = f"效价{s.valence:+.2f} 唤醒{s.arousal:.2f} 标签={s.label} 置信{s.confidence:.2f}"
        if direction == "neg":
            ok = s.valence < -0.1
        elif direction == "pos":
            ok = s.valence > 0.1
        elif direction == "pos_mild":
            # 轻微正面：方向对，但不能顶格
            ok = 0.05 < s.valence < 0.6
        elif direction == "neutral":
            ok = abs(s.valence) < 0.45
        # 标签断言：'焦虑'与'愤怒'的边界本来就是连续的，只在明确时校验
        if ok and expect_label:
            ok = s.label == expect_label
            detail += f"（期望标签 {expect_label}）"
        check(f"{note}：{text[:16]}", ok, detail)

    print("\n  【边界测试：否定词处理】")
    cases_neg = [
        ("我不开心", "neg", "「不开心」应判为负向"),
        ("我不焦虑", "pos_or_neutral", "「不焦虑」不该判为负向"),
    ]
    for text, expect, note in cases_neg:
        s = emo.analyze(text)
        if expect == "neg":
            ok = s.valence < 0
        else:
            ok = s.valence >= -0.1
        check(note, ok, f"效价{s.valence:+.2f} 命中{s.matched}")

    print("\n  【边界测试：程度副词】")
    weak = emo.analyze("我有点焦虑")
    strong = emo.analyze("我非常焦虑")
    print(f"    '有点焦虑' 唤醒={weak.arousal:.2f}   '非常焦虑' 唤醒={strong.arousal:.2f}")
    check("程度副词影响强度", strong.arousal > weak.arousal,
          f"{weak.arousal:.2f} < {strong.arousal:.2f}")

    print("\n  【已知局限：词典方法判不了的情况】")
    hard = [
        ("反讽：真是太好了，又加班到十二点", emo.analyze("真是太好了，又加班到十二点")),
        ("隐晦：最近睡得不太好", emo.analyze("最近睡得不太好")),
        ("长句夹杂：虽然项目延期了，但我觉得问题不大，就是有点担心进度", emo.analyze("虽然项目延期了，但我觉得问题不大，就是有点担心进度")),
    ]
    for note, s in hard:
        print(f"    {note[:24]:<28} → 效价{s.valence:+.2f} 标签={s.label} 置信{s.confidence:.2f}")
    print("    → 这些是词典方法的固有局限，也是为什么要有低置信度兜底：")
    print("      判不准时退回中性姿态，宁可不做姿态，也不用错误的姿态回应。")

    metrics["lexicon_notes"] = "反讽与隐晦表达判不准，靠低置信度兜底退回中性"


# ============================================================ B. 状态维护
def test_state() -> None:
    print("\n" + "=" * 68)
    print("B. 情绪状态维护：EMA 平滑与趋势检测")
    print("=" * 68)

    print("\n  [1] 单轮极端情绪不该让状态剧烈跳变")
    state = emo.EmotionState()
    extreme = emo.EmotionScore(valence=-1.0, arousal=1.0, label="愤怒", confidence=1.0)
    state = emo.update_state(state, extreme)
    print(f"    输入效价 -1.00，一轮后状态效价 = {state.valence:+.3f}")
    check("状态没有直接跳到 -1", state.valence > -0.5, f"实际 {state.valence:+.3f}")
    check("但确实往负向走了", state.valence < 0, f"实际 {state.valence:+.3f}")

    print("\n  [2] 连续负向会逐步累积（模拟持续低落）")
    state2 = emo.EmotionState()
    print(f"    {'轮次':<6}{'即时效价':>12}{'长期倾向':>12}{'持续低落?':>12}")
    track = []
    for i in range(10):
        s = emo.EmotionScore(valence=-0.5, arousal=0.6, label="焦虑", confidence=1.0)
        state2 = emo.update_state(state2, s)
        track.append(state2.valence)
        print(
            f"    {i + 1:<6}{state2.valence:>12.3f}{state2.stable_valence:>12.3f}"
            f"{('是' if state2.recent_low else '否'):>12}"
        )
    check("状态持续下降", track[-1] < track[0], f"首轮 {track[0]:.3f} → 末轮 {track[-1]:.3f}")
    check("连续负向后判定为持续低落", state2.recent_low, f"recent_low={state2.recent_low}")
    print("    说明：判定是有意保守的 —— 需要即时状态和长期倾向都越过阈值才报警，")
    print("          避免用户只抱怨一句就被系统标记为『情绪问题』。")
    metrics["sustained_negative_curve"] = [round(v, 3) for v in track]

    print("\n  [3] 低置信度输入不污染状态")
    state3 = emo.EmotionState(valence=-0.6, arousal=0.7, label="焦虑", turns=5)
    before = state3.valence
    neutral = emo.EmotionScore(valence=0.0, arousal=0.0, label="中性", confidence=0.0)
    state3 = emo.update_state(state3, neutral)
    print(f"    输入『嗯』（无情绪词，置信度 0）前后：{before:+.3f} → {state3.valence:+.3f}")
    check("状态保持不变", abs(state3.valence - before) < 1e-9,
          f"{before:+.3f} → {state3.valence:+.3f}")

    print("\n  [4] 情绪转折检测")
    drop = [
        emo.EmotionScore(valence=-0.1, arousal=0.4, confidence=1.0),
        emo.EmotionScore(valence=-0.1, arousal=0.4, confidence=1.0),
        emo.EmotionScore(valence=-0.2, arousal=0.4, confidence=1.0),
        emo.EmotionScore(valence=-0.8, arousal=0.7, confidence=1.0),
        emo.EmotionScore(valence=-0.9, arousal=0.8, confidence=1.0),
        emo.EmotionScore(valence=-0.85, arousal=0.8, confidence=1.0),
    ]
    flat = [emo.EmotionScore(valence=-0.3, arousal=0.4, confidence=1.0) for _ in range(6)]
    check("情绪恶化能被检出", emo.detect_shift(drop) is True)
    check("平稳对话不误报", emo.detect_shift(flat) is False)
    check("样本太少时不误报", emo.detect_shift(drop[:3]) is False)


# ============================================================ C. 情感加权
def test_emotion_weighting() -> None:
    print("\n" + "=" * 68)
    print("C. 情感加权召回：消融对比")
    print("=" * 68)

    now = datetime.now()
    # 造一批记忆：语义上都跟"副业"相关，但情绪色彩不同
    mems = [
        ("用户想做宠物相关的副业", 0.85, 0.5, now - timedelta(days=2), "正"),
        ("用户担心副业坚持不下来，很焦虑", 0.80, -0.6, now - timedelta(days=2), "负"),
        ("用户说做副业让他很有成就感，很开心", 0.70, 0.7, now - timedelta(days=2), "正"),
        ("用户因为副业进度慢被家人说过，很沮丧", 0.65, -0.5, now - timedelta(days=2), "负"),
    ]

    query = "我的副业现在该怎么办"

    def build(use_emotion: bool, state):
        out = []
        for content, imp, val, created, tag in mems:
            m = Memory(user_id="u", content=content, type="goal", importance=imp,
                       valence=val, arousal=0.5, emotion_label="测试")
            m.id = hash(content) % 10000
            m.created_at = created
            sim = 0.5  # 假设语义相似度相同，隔离出情绪因素
            emo_c = mr.emotion_consistency(m, state) if use_emotion else 0.5
            if use_emotion:
                score = (mr.WEIGHT_SIMILARITY * sim + mr.WEIGHT_RECENCY * mr.recency_score(created, now)
                         + mr.WEIGHT_IMPORTANCE * imp + mr.WEIGHT_EMOTION * emo_c)
            else:
                score = ((mr.WEIGHT_SIMILARITY + mr.WEIGHT_EMOTION) * sim
                         + mr.WEIGHT_RECENCY * mr.recency_score(created, now)
                         + mr.WEIGHT_IMPORTANCE * imp)
            out.append(mr.ScoredMemory(memory=m, similarity=sim,
                                       recency=mr.recency_score(created, now),
                                       importance=imp, emotion=emo_c, score=score))
        out.sort(key=lambda s: s.score, reverse=True)
        return out

    scenarios = [
        ("用户此刻很焦虑（效价 -0.6）", emo.EmotionState(valence=-0.6, arousal=0.7, turns=5), "负"),
        ("用户此刻很开心（效价 +0.7）", emo.EmotionState(valence=0.7, arousal=0.6, turns=5), "正"),
    ]

    for note, state, expects in scenarios:
        print(f"\n  【{note}】")
        off = build(False, state)
        on = build(True, state)
        print(f"    {'':<4}{'关闭情感加权':<34}{'开启情感加权'}")
        for i in range(len(off)):
            a = off[i].memory.content[:16]
            b = on[i].memory.content[:16]
            mark = "  ← 变化" if a != b else ""
            print(f"    {i + 1}. {a:<32}{b}{mark}")

        if expects == "负":
            check("焦虑时「焦虑」相关记忆排到第一",
                  "焦虑" in on[0].memory.content,
                  f"实际 Top1：{on[0].memory.content[:20]}")
            check("关闭时排序不同（证明加权有效）",
                  off[0].memory.content != on[0].memory.content,
                  f"关闭 Top1：{off[0].memory.content[:20]}")
        else:
            # 正向记忆的相对排名应该上升。
            # 注意：这里不该断言"正向记忆排第一" ——
            # "用户想做宠物相关的副业"本身就是相关性最高的那条（它是纯目标陈述），
            # 让它保住第一是正确的。真正该看的是正向记忆相对负向记忆的位次变化。
            def rank_of(target):
                return next(
                    (i for i, s in enumerate(on) if target in s.memory.content), 99
                )

            def rank_of_off(target):
                return next(
                    (i for i, s in enumerate(off) if target in s.memory.content), 99
                )

            pos_on, neg_on = rank_of("成就感"), rank_of("焦虑")
            pos_off, neg_off = rank_of_off("成就感"), rank_of_off("焦虑")
            print(f"    （正向记忆位次 {pos_off + 1} → {pos_on + 1}；"
                  f"负向记忆位次 {neg_off + 1} → {neg_on + 1}）")
            check("开心时正向记忆位次上升",
                  pos_on <= pos_off, f"{pos_off + 1} → {pos_on + 1}")
            check("开心时正向记忆排在负向记忆之前",
                  pos_on < neg_on, f"正 {pos_on + 1} vs 负 {neg_on + 1}")

    print("\n  【情绪一致性数值】")
    state_neg = emo.EmotionState(valence=-0.6, arousal=0.7, turns=5)
    for content, imp, val, created, tag in mems:
        m = Memory(user_id="u", content=content, valence=val)
        c = mr.emotion_consistency(m, state_neg)
        print(f"    记忆效价{val:+.1f} 对 当前效价{state_neg.valence:+.1f} → 一致性 {c:.3f}  ({tag})")
    neg_c = mr.emotion_consistency(Memory(user_id="u", content="", valence=-0.6), state_neg)
    pos_c = mr.emotion_consistency(Memory(user_id="u", content="", valence=0.7), state_neg)
    check("同向记忆一致性更高", neg_c > pos_c, f"{neg_c:.3f} > {pos_c:.3f}")
    metrics["emotion_consistency"] = {"同向": round(neg_c, 3), "反向": round(pos_c, 3)}


# ============================================================ D. 回复姿态
def test_strategy() -> None:
    print("\n" + "=" * 68)
    print("D. 情绪到回复姿态的映射")
    print("=" * 68)

    cases = [
        (emo.EmotionState(valence=-0.6, arousal=0.8, turns=3), "接情绪", "焦虑/愤怒 → 先接住情绪"),
        (emo.EmotionState(valence=-0.5, arousal=0.2, turns=3), "陪伴", "低落 → 温和陪伴"),
        (emo.EmotionState(valence=0.7, arousal=0.8, turns=3), "同在", "兴奋 → 一起放大"),
        (emo.EmotionState(valence=0.5, arousal=0.2, turns=3), "平稳", "平静正面 → 正常交流"),
        (emo.EmotionState(valence=0.0, arousal=0.1, turns=3), "平稳", "中性 → 正常交流"),
    ]
    for state, expect, note in cases:
        s = emo.pick_strategy(state)
        check(f"{note}", s.name == expect, f"实际 {s.name}")

    print("\n  【姿态指令内容示例】")
    st = emo.EmotionState(valence=-0.6, arousal=0.8, turns=3)
    strat = emo.pick_strategy(st)
    block = emo.format_emotion_block(st, strat)
    for line in block.split("\n"):
        print(f"    | {line}")


async def test_emotion_e2e() -> None:
    """真模型对比：同一个问题，在不同情绪状态下回答有什么不同。"""
    print("\n" + "=" * 68)
    print("E. 真模型对比：情绪姿态是否真的改变了回答")
    print("=" * 68)

    from app.core.config import settings
    from app.llm.openai_compat import get_llm
    from app.orchestration import prompt_builder as pb

    if not settings.llm_api_key:
        print(f"  {FAIL} 没有 API Key，跳过")
        return

    user_id = f"u_emo_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)
    await emotion_store.clear_user(user_id)

    # 造一批记忆，让"副业"相关的内容有正负两面
    await long_term.save_memories(
        user_id,
        [
            MemoryItem("用户在做宠物相关的副业，已经做了两周", "goal", 0.85,
                       valence=0.3, arousal=0.5),
            MemoryItem("用户担心这个副业最后会失败，之前有过半途而废的经历", "emotion", 0.8,
                       valence=-0.6, arousal=0.6),
            MemoryItem("用户说做出第一个功能时很有成就感", "event", 0.7,
                       valence=0.8, arousal=0.7),
        ],
        semantic_dedupe=False,
    )

    question = "我那个副业现在该怎么办？"
    llm = get_llm()

    async def ask(state, use_emotion: bool):
        bundle = await pb.assemble(
            user_id=user_id,
            user_input=question,
            history=[],
            emotion_state=state,
            use_emotion=use_emotion,
        )
        reply = await llm.chat(bundle.messages, temperature=0.5)
        return reply.text, bundle

    # 场景 1：焦虑状态，开启情感
    anxious = emo.EmotionState(valence=-0.6, arousal=0.75, turns=4, stable_valence=-0.4)
    text_anxious, b1 = await ask(anxious, True)
    print(f"\n  【焦虑状态 + 情感姿态开启】（输出 {len(text_anxious)} 字）")
    print(f"    {text_anxious[:300]}")
    print(f"    → 姿态段是否注入：{getattr(b1, 'emotion_used', False)}")

    # 场景 2：焦虑状态，关闭情感（消融）
    text_noemo, b2 = await ask(anxious, False)
    print(f"\n  【焦虑状态 + 情感姿态关闭】（输出 {len(text_noemo)} 字）")
    print(f"    {text_noemo[:300]}")

    # 场景 3：平静状态
    calm = emo.EmotionState(valence=0.1, arousal=0.2, turns=4)
    text_calm, b3 = await ask(calm, True)
    print(f"\n  【平静状态】（输出 {len(text_calm)} 字）")
    print(f"    {text_calm[:300]}")

    # 判定：有情感姿态时回答应该更短、更克制（先接情绪而不是长篇建议）
    print("\n  【量化对比】")
    print(f"    焦虑+姿态：{len(text_anxious)} 字")
    print(f"    焦虑无姿态：{len(text_noemo)} 字")
    print(f"    平静：{len(text_calm)} 字")
    check("情感姿态被注入到提示词", getattr(b1, "emotion_used", False))
    check("关闭时没有姿态段", not getattr(b2, "emotion_used", False))
    metrics["emotion_response_length"] = {
        "焦虑+姿态": len(text_anxious),
        "焦虑无姿态": len(text_noemo),
        "平静": len(text_calm),
    }

    # 检查焦虑姿态下的回答是否避免了"你应该"
    preachy = any(w in text_anxious for w in ("你应该", "其实你可以", "建议你"))
    print(f"    焦虑姿态下出现『你应该/其实你可以/建议你』：{'是（姿态未生效）' if preachy else '否（符合要求）'}")
    check("姿态要求被遵守（不急于说教）", not preachy, "回答里没有说教句式")

    await long_term.clear_user(user_id)
    await emotion_store.clear_user(user_id)


async def main() -> int:
    offline_only = "--offline" in sys.argv
    await init_db()

    test_analyze()
    test_state()
    test_emotion_weighting()
    test_strategy()
    if not offline_only:
        await test_emotion_e2e()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 68)
    print("本次实测数据")
    print("=" * 68)
    if "emotion_consistency" in metrics:
        c = metrics["emotion_consistency"]
        print(f"  情绪一致性：同向记忆 {c['同向']} vs 反向记忆 {c['反向']}")
    if "emotion_response_length" in metrics:
        r = metrics["emotion_response_length"]
        print(f"  回答长度：焦虑+姿态 {r['焦虑+姿态']} 字 / 焦虑无姿态 {r['焦虑无姿态']} 字 / 平静 {r['平静']} 字")
    if "sustained_negative_curve" in metrics:
        print(f"  持续负向时的状态曲线：{metrics['sustained_negative_curve']}")

    print("\n" + "=" * 68)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} 情感记忆模块完成！")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
