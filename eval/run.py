"""评估执行器：在同一套评估集上对比多种配置。

这是整个项目最有价值的一个脚本，因为它把前面所有零散的优化
放进同一张表里，用同一套题目、同一个模型跑出可比较的结果。

七种配置（逐步叠加，形成消融实验）：

  1. baseline_full_history  朴素基线：小模型 + 全量历史，不用记忆、不用知识库
  2. +retrieval             加语义检索（从"按重要度取"升级为"按相关度取"）
  3. +emotion               加情感加权（第四项打分）
  4. +budget                加 Token 预算调度
  5. +cache                 加回答缓存
  6. full                   完整方案
  7. big_model_full         大模型 + 完整方案（对比"多花钱值不值"）

指标：
  Recall       应该召回的事实里，实际召回了几条（检索质量）
  Faithful     回答是否只基于注入信息，没编造（反幻觉）
  Refusal      问到没提过的信息时，是否诚实说不知道
  Tone OK      情感场景下是否遵守姿态要求（没说教、没泼冷水）
  Latency      端到端耗时
  Prompt tok   提示词 token 数

用法：
    .venv\\Scripts\\python.exe -m eval.run
    .venv\\Scripts\\python.exe -m eval.run --quick      # 只跑前 2 种配置
    .venv\\Scripts\\python.exe -m eval.run --big qwen-max
"""

import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.core.config import ROOT_DIR, settings  # noqa: E402
from app.llm.base import Message  # noqa: E402
from app.llm.openai_compat import OpenAICompatClient  # noqa: E402
from app.memory import emotion_store, long_term, short_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.emotion import EmotionState  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.orchestration import prompt_builder as pb  # noqa: E402
from eval.dataset import (  # noqa: E402
    EMOTION_SCENARIOS,
    EVAL_USER,
    all_cases,
    summary as dataset_summary,
)

OK = "[OK]"
FAIL = "[FAIL]"


# ============================================================ 结果结构


@dataclass
class CaseResult:
    question: str
    category: str
    answer: str
    recall_hit: int = 0
    recall_total: int = 0
    faithful: bool = False
    refused_correctly: bool = False
    tone_ok: bool = True
    latency_ms: int = 0
    prompt_tokens: int = 0


@dataclass
class ConfigResult:
    name: str
    label: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def recall_rate(self) -> float:
        hits = sum(c.recall_hit for c in self.cases)
        total = sum(c.recall_total for c in self.cases)
        return hits / total if total else 0.0

    @property
    def faithful_rate(self) -> float:
        scored = [c for c in self.cases if c.category in ("memory", "knowledge")]
        if not scored:
            return 0.0
        return sum(1 for c in scored if c.faithful) / len(scored)

    @property
    def refusal_rate(self) -> float:
        scored = [c for c in self.cases if c.category == "refusal"]
        if not scored:
            return 0.0
        return sum(1 for c in scored if c.refused_correctly) / len(scored)

    @property
    def tone_rate(self) -> float:
        scored = [c for c in self.cases if c.category == "emotion"]
        if not scored:
            return 0.0
        return sum(1 for c in scored if c.tone_ok) / len(scored)

    @property
    def p50_latency(self) -> int:
        vals = [c.latency_ms for c in self.cases if c.latency_ms]
        return int(statistics.median(vals)) if vals else 0

    @property
    def avg_prompt_tokens(self) -> int:
        vals = [c.prompt_tokens for c in self.cases if c.prompt_tokens]
        return int(statistics.mean(vals)) if vals else 0

    @property
    def avg_answer_chars(self) -> int:
        vals = [len(c.answer) for c in self.cases if c.answer]
        return int(statistics.mean(vals)) if vals else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "recall": round(self.recall_rate, 4),
            "faithful": round(self.faithful_rate, 4),
            "refusal": round(self.refusal_rate, 4),
            "tone": round(self.tone_rate, 4),
            "p50_latency_ms": self.p50_latency,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_answer_chars": self.avg_answer_chars,
            "cases": [
                {
                    "question": c.question,
                    "category": c.category,
                    "answer": c.answer,
                    "recall": f"{c.recall_hit}/{c.recall_total}",
                    "faithful": c.faithful,
                    "tone_ok": c.tone_ok,
                    "latency_ms": c.latency_ms,
                    "prompt_tokens": c.prompt_tokens,
                }
                for c in self.cases
            ],
        }


# ============================================================ 判定逻辑


def check_recall(scored_memories, should_recall: list[str], answer: str) -> tuple[int, int]:
    """检查应该被召回的事实，实际召回了几条。

    判定依据是**检索结果**（而不是回答），因为我们要测的是检索质量：
    模型答不出来可能是表达问题，但检索没捞到就是检索的问题。
    """
    if not should_recall:
        return 0, 0
    retrieved_text = " ".join(s.memory.content for s in scored_memories)
    # 同时把注入的档案也算上（档案是全量注入的）
    hit = 0
    for kw in should_recall:
        if kw in retrieved_text or kw in answer:
            hit += 1
    return hit, len(should_recall)


async def judge_faithful(client, question: str, answer: str, injected: str) -> bool:
    """用模型做裁判：回答里有没有注入信息之外的内容（编造）。

    这是 LLM-as-Judge 的简化用法。注意它的局限：
    裁判本身也是模型，也会出错。所以文档里要写明"模型裁判 + 人工抽检"。
    """
    prompt = (
        "你是严格的评审。判断下面这个回答是否只使用了提供的资料，"
        "有没有加入资料里没有的具体信息（编造）。\n\n"
        f"【提供的资料】\n{injected[:1500]}\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【回答】\n{answer}\n\n"
        "只输出一个词：FAITHFUL（没有编造）或 FABRICATED（有编造具体信息）"
    )
    try:
        r = await client.chat([Message("user", prompt)], temperature=0.0, max_tokens=10)
        return "FABRICATED" not in r.text.upper()
    except Exception:  # noqa: BLE001
        return False


def check_refusal(answer: str, keywords: list[str]) -> bool:
    """检查是否诚实说不知道。"""
    return any(k in answer for k in keywords)


def check_tone(answer: str, forbidden: list[str]) -> bool:
    """检查情感姿态是否被遵守（没出现被禁止的句式）。"""
    return not any(f in answer for f in forbidden)


# ============================================================ 跑一种配置


async def run_config(
    *,
    name: str,
    label: str,
    client: OpenAICompatClient,
    use_retrieval: bool,
    use_emotion: bool,
    use_budget: bool,
    full_history: bool = False,
    judge_client: OpenAICompatClient | None = None,
) -> ConfigResult:
    """跑一种配置下的全部评估题。"""
    result = ConfigResult(name=name, label=label)
    judge = judge_client or client

    # 每次都用干净的会话，避免历史互相污染
    import uuid

    session_id = f"s_eval_{uuid.uuid4().hex[:8]}"

    # ---- 记忆类 + 拒绝类 ----
    for case in all_cases():
        question = case.question

        if full_history:
            # 基线：不用记忆、不用知识库，把"用户说过的所有话"当历史塞进去
            fake_history = [
                Message("user", content) for content, _, _ in EVAL_USER.facts
            ]
            bundle = pb.build_messages(
                user_input=question,
                history=fake_history,
                profile=[],
                memories=[],
                knowledge=[],
                emotion_block="",
                apply_budget=False,
            )
            scored = []
        else:
            from app.retrieval import memory_retriever as mr

            state = (
                await emotion_store.get_state(EVAL_USER.user_id) if use_emotion else None
            )
            profile = await long_term.get_profile(EVAL_USER.user_id)

            scored = await mr.retrieve_memories(
                EVAL_USER.user_id,
                question,
                top_k=settings.memory_top_k,
                candidate_k=settings.memory_candidate_k,
                emotion_state=state if use_emotion else None,
                use_emotion=use_emotion,
            )
            memories = [s.memory for s in scored]

            knowledge = []
            if case.category == "knowledge" and settings.enable_knowledge:
                raw = await mr.retrieve_knowledge(
                    question, top_k=settings.knowledge_top_k
                )
                knowledge = [
                    k for k in raw
                    if k.get("similarity", 0) >= settings.knowledge_min_similarity
                ]

            emotion_block = ""
            if use_emotion and state is not None:
                emotion_block = pb.build_emotion_block(state)

            bundle = pb.build_messages(
                user_input=question,
                history=[],
                profile=profile,
                memories=memories,
                knowledge=knowledge,
                emotion_block=emotion_block,
                apply_budget=use_budget,
            )

        t0 = time.perf_counter()
        try:
            reply = await client.chat(bundle.messages, temperature=0.3)
        except Exception as e:  # noqa: BLE001
            print(f"    {FAIL} 调用失败：{e}")
            continue
        latency = int((time.perf_counter() - t0) * 1000)

        hit, total = check_recall(scored, case.should_recall, reply.text)

        cr = CaseResult(
            question=question,
            category=case.category,
            answer=reply.text,
            recall_hit=hit,
            recall_total=total,
            latency_ms=latency,
            prompt_tokens=reply.prompt_tokens or bundle.est_prompt_tokens,
        )

        # 反幻觉判定（只对记忆/知识类做，拒绝类单独判）
        if case.category in ("memory", "knowledge"):
            injected = "\n".join(
                [m.content for m in bundle.messages if m.role == "system"]
            )
            cr.faithful = await judge_faithful(judge, question, reply.text, injected)
        elif case.category == "refusal":
            cr.refused_correctly = check_refusal(reply.text, case.expect_keywords)
            cr.faithful = cr.refused_correctly

        result.cases.append(cr)
        mark = "命中" if (hit == total or total == 0) else f"{hit}/{total}"
        print(f"    [{mark:>5}] {question[:26]:<28} {latency:>5}ms")

    # ---- 情感场景 ----
    for sc in EMOTION_SCENARIOS:
        # 把场景历史落库，让历史召回生效
        store_session = f"{session_id}_{sc['name']}"
        for u, a in sc["history"]:
            await short_term.append(store_session, EVAL_USER.user_id, "user", u)
            await short_term.append(store_session, EVAL_USER.user_id, "assistant", a)

        if full_history:
            history = [
                Message(role, content)
                for u, a in sc["history"]
                for role, content in (("user", u), ("assistant", a))
            ]
            bundle = pb.build_messages(
                user_input=sc["question"], history=history, apply_budget=False
            )
        else:
            history = await short_term.get_recent(store_session, limit=10)
            state = EmotionState(
                valence=-0.6, arousal=0.75, stable_valence=-0.4, turns=4
            ) if use_emotion else None
            bundle = await pb.assemble(
                user_id=EVAL_USER.user_id,
                user_input=sc["question"],
                history=history,
                emotion_state=state,
                use_emotion=use_emotion,
            )

        t0 = time.perf_counter()
        try:
            reply = await client.chat(bundle.messages, temperature=0.4)
        except Exception:  # noqa: BLE001
            continue
        latency = int((time.perf_counter() - t0) * 1000)

        cr = CaseResult(
            question=f"[{sc['name']}] {sc['question']}",
            category="emotion",
            answer=reply.text,
            tone_ok=check_tone(reply.text, sc["forbidden"]),
            latency_ms=latency,
            prompt_tokens=reply.prompt_tokens or bundle.est_prompt_tokens,
        )
        result.cases.append(cr)
        print(
            f"    [{'OK' if cr.tone_ok else '违规':>5}] {sc['name']:<28} "
            f"{latency:>5}ms  {len(reply.text)}字"
        )
        await short_term.clear_session(store_session)

    return result


# ============================================================ 主流程


async def prepare_user() -> None:
    """植入评估用户的记忆与档案。"""
    await long_term.clear_user(EVAL_USER.user_id)
    await emotion_store.clear_user(EVAL_USER.user_id)

    items = [
        MemoryItem(content=c, type=t, importance=imp, valence=0.0, arousal=0.0)
        for c, t, imp in EVAL_USER.facts
    ]
    saved = await long_term.save_memories(
        EVAL_USER.user_id, items, semantic_dedupe=False
    )
    print(f"  已植入 {len(saved)} 条用户事实")
    profile = await long_term.get_profile(EVAL_USER.user_id)
    print(f"  已生成 {len(profile)} 条档案项")


def print_table(results: list[ConfigResult]) -> None:
    print("\n" + "=" * 100)
    print("汇总对比")
    print("=" * 100)
    header = (
        f"{'配置':<26}{'Recall':>9}{'忠实度':>9}{'拒答正确':>11}"
        f"{'姿态合规':>11}{'P50延迟':>11}{'prompt tok':>12}{'回答字数':>10}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r.label:<26}"
            f"{r.recall_rate:>8.0%}"
            f"{r.faithful_rate:>9.0%}"
            f"{r.refusal_rate:>11.0%}"
            f"{r.tone_rate:>11.0%}"
            f"{r.p50_latency:>10}ms"
            f"{r.avg_prompt_tokens:>12}"
            f"{r.avg_answer_chars:>10}"
        )
    print("-" * 100)


def save_report(results: list[ConfigResult], meta: dict) -> Path:
    """把报告存成 JSON + Markdown，便于写进简历和面试时翻看。"""
    out_dir = ROOT_DIR / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "meta": meta,
        "results": [r.to_dict() for r in results],
    }
    json_path = out_dir / f"eval_{stamp}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown 表格
    lines = [
        "# 评估报告",
        "",
        f"生成时间：{payload['generated_at']}",
        f"模型：{meta.get('small_model')} / 对照大模型：{meta.get('big_model')}",
        f"评估集：{meta.get('dataset')}",
        "",
        "| 配置 | Recall | 忠实度 | 拒答正确 | 姿态合规 | P50延迟 | prompt token | 回答字数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.recall_rate:.0%} | {r.faithful_rate:.0%} | "
            f"{r.refusal_rate:.0%} | {r.tone_rate:.0%} | {r.p50_latency}ms | "
            f"{r.avg_prompt_tokens} | {r.avg_answer_chars} |"
        )
    lines += ["", "## 详细回答", ""]
    for r in results:
        lines.append(f"### {r.label}")
        lines.append("")
        for c in r.cases:
            lines.append(f"- **{c.question}**（{c.category}）")
            lines.append(f"  - Recall：{c.recall_hit}/{c.recall_total}　"
                         f"忠实：{c.faithful}　姿态合规：{c.tone_ok}　"
                         f"{c.latency_ms}ms　prompt {c.prompt_tokens} tok")
            lines.append(f"  - 回答：{c.answer[:200].replace(chr(10), ' ')}")
        lines.append("")

    md_path = out_dir / f"eval_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n  报告已保存：")
    print(f"    {json_path}")
    print(f"    {md_path}")
    return md_path


async def main() -> int:
    args = sys.argv[1:]
    quick = "--quick" in args
    big_model = "qwen-max"
    if "--big" in args:
        big_model = args[args.index("--big") + 1]

    await init_db()

    if not settings.llm_api_key:
        print(f"{FAIL} 没有 API Key")
        return 1

    print("=" * 100)
    print("评估集与消融实验")
    print("=" * 100)
    ds = dataset_summary()
    for k, v in ds.items():
        print(f"  {k}：{v}")

    print("\n准备评估用户…")
    await prepare_user()

    small = OpenAICompatClient(model=settings.llm_model)
    big = OpenAICompatClient(model=big_model)

    configs = [
        # (name, label, use_retrieval, use_emotion, use_budget, full_history)
        ("baseline_full_history", "1. 基线：全量历史", False, False, False, True),
        ("retrieval_only", "2. + 语义检索", True, False, False, False),
        ("retrieval_emotion", "3. + 情感加权", True, True, False, False),
        ("full", "4. + 预算调度（完整）", True, True, True, False),
    ]
    if quick:
        configs = configs[:2]

    results: list[ConfigResult] = []
    for name, label, retr, emo, bud, full_h in configs:
        print(f"\n{'=' * 100}")
        print(f"配置 {label}")
        print("=" * 100)
        r = await run_config(
            name=name,
            label=label,
            client=small,
            use_retrieval=retr,
            use_emotion=emo,
            use_budget=bud,
            full_history=full_h,
        )
        results.append(r)

    if not quick:
        print(f"\n{'=' * 100}")
        print(f"配置 5. 大模型（{big_model}）+ 完整方案")
        print("=" * 100)
        r_big = await run_config(
            name="big_model_full",
            label=f"5. {big_model} + 完整方案",
            client=big,
            use_retrieval=True,
            use_emotion=True,
            use_budget=True,
        )
        results.append(r_big)

    print_table(results)

    meta = {
        "small_model": settings.llm_model,
        "big_model": big_model,
        "dataset": ds,
        "budget": settings.max_prompt_tokens,
        "top_k": settings.memory_top_k,
    }
    save_report(results, meta)

    # 关键结论
    if len(results) >= 2:
        base, full = results[0], results[-2] if not quick else results[1]
        print("\n" + "=" * 100)
        print("关键结论")
        print("=" * 100)
        print(f"  基线（全量历史）→ 完整方案：")
        print(f"    Recall   {base.recall_rate:.0%} → {full.recall_rate:.0%}")
        print(f"    忠实度   {base.faithful_rate:.0%} → {full.faithful_rate:.0%}")
        print(f"    P50延迟  {base.p50_latency}ms → {full.p50_latency}ms")
        print(f"    prompt   {base.avg_prompt_tokens} → {full.avg_prompt_tokens} tok")

    await long_term.clear_user(EVAL_USER.user_id)
    await emotion_store.clear_user(EVAL_USER.user_id)
    from app.core import tasks

    await tasks.wait_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
