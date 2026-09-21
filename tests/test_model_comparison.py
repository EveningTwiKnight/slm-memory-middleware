"""模型对照实验：小模型 + 中间件 vs 大模型裸跑。

这个实验回答一个面试必问的问题：
「你为什么用小模型？大模型不是更好吗？」

做法：让**同一批问题、同一套中间件**分别走小模型和大模型，对比四个维度：
  1. 记忆类问题的正确率（能不能答出用户信息）
  2. 知识库问题的忠实度（有没有编造）
  3. 端到端延迟
  4. token 消耗（成本代理指标）

三种配置对比：
  A. 小模型 + 全量历史（朴素做法，作为基线）
  B. 小模型 + 中间件（我们的方案）
  C. 大模型 + 中间件（多花钱能不能换来更好效果？）

用法：
    # 先确认 .env 里配的是小模型（qwen-flash）
    .venv\\Scripts\\python.exe -m tests.test_model_comparison

    # 指定对比的大模型
    .venv\\Scripts\\python.exe -m tests.test_model_comparison --big qwen-max

前置：知识库要有内容
    .venv\\Scripts\\python.exe -m scripts.ingest_kb --dir docs/kb_sample --reset
"""

import asyncio
import statistics
import sys
import time
import uuid

sys.path.insert(0, ".")

from app.core import tasks  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.llm.base import Message  # noqa: E402
from app.llm.openai_compat import OpenAICompatClient  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.memory.database import SessionLocal, init_db  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.memory.models import Conversation  # noqa: E402
from app.orchestration import prompt_builder  # noqa: E402
from app.memory import short_term  # noqa: E402

SMALL_MODEL = settings.llm_model
BIG_MODEL = "qwen-max"

# ---- 测试用的人物设定，刻意做成"只有中间件才知道"的信息 ----
PERSONA_FACTS = [
    ("用户叫沈知远，是一名做工业视觉的算法工程师", "fact", 0.95),
    ("用户在上海工作，所在团队只有五个人", "fact", 0.75),
    ("用户正在把一条缺陷检测产线从人工复检改造成全自动", "goal", 0.90),
    ("用户很担心新模型上线后误检率上升导致产线停线", "emotion", 0.85),
    ("用户习惯把问题拆成小步骤逐个验证，不喜欢一次性大改动", "preference", 0.60),
]

# ---- 测试问题与判定关键词 ----
MEMORY_QUESTIONS = [
    ("你还记得我是谁、做什么的吗？", ["沈知远", "算法"], "记忆：身份"),
    ("我最近在推进什么项目？", ["缺陷", "检测"], "记忆：项目"),
    ("我主要担心什么？", ["误检", "停线"], "记忆：担忧"),
    ("我平时喜欢怎么推进工作？", ["小步骤", "逐个"], "记忆：偏好"),
]

KB_QUESTIONS = [
    ("token 预算超了应该按什么顺序裁剪？", ["历史", "低分"], "知识库：裁剪顺序"),
    ("记忆的衰减半衰期是多久？", ["30"], "知识库：半衰期"),
]

# 反幻觉测试：知识库里没有答案的问题，模型应该说"不知道"，而不是编造
HALLUCINATION_QUESTIONS = [
    ("我们公司今年的营收目标是多少？", "知识库没有的信息"),
]


async def build_history(user_id: str, session_id: str, n_turns: int) -> list[Message]:
    """造一段较长的对话历史，用来测长上下文下的表现。"""
    filler = [
        ("今天上午开了个会，讨论了下季度的排期", "排期的事可以列个清单，按优先级来。"),
        ("产线上的相机好像有点偏色，我调了下白平衡", "偏色会影响检测结果，建议固定光源后重新标定。"),
        ("中午跟同事吃了顿拉面，味道一般", "下次可以换一家试试。"),
        ("我把标注规范又细化了一版", "细化规范能减少标注歧义，值得做。"),
        ("下午测了下新模型的召回率", "召回率变化大吗？如果波动大要看是不是数据分布有变化。"),
    ]
    msgs: list[Message] = []
    for i in range(n_turns):
        u, a = filler[i % len(filler)]
        msgs.append(Message("user", f"{u}（第{i + 1}轮）"))
        msgs.append(Message("assistant", a))
    return msgs


async def run_config(
    name: str,
    client: OpenAICompatClient,
    user_id: str,
    *,
    use_middleware: bool,
    history_turns: int = 0,
) -> dict:
    """用指定模型和配置跑一遍全部测试问题，返回统计结果。"""
    print(f"\n{'=' * 70}")
    print(f"配置 {name}")
    print(f"{'=' * 70}")

    session_id = f"s_{uuid.uuid4().hex[:8]}"
    history = await build_history(user_id, session_id, history_turns) if history_turns else []

    if history:
        # 把自己造成的历史也落库，模拟"聊了很久"
        for m in history:
            await short_term.append(session_id, user_id, m.role, m.content)

    results = {
        "name": name,
        "correct": 0,
        "total": 0,
        "latencies": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "details": [],
    }

    all_questions = MEMORY_QUESTIONS + KB_QUESTIONS

    for question, keywords, label in all_questions:
        if use_middleware:
            bundle = await prompt_builder.assemble(
                user_id=user_id,
                user_input=question,
                history=history[-settings.history_limit :] if history else [],
            )
            messages = bundle.messages
            est = bundle.est_prompt_tokens
        else:
            # 朴素做法：人设 + 全量历史 + 当前问题，不做记忆注入、不做检索
            messages = [
                Message("system", "你是一个耐心、务实的助手。"),
                *history,
                Message("user", question),
            ]
            est = sum(len(m.content) for m in messages)

        t0 = time.perf_counter()
        try:
            reply = await client.chat(messages, temperature=0.3)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {label}：调用出错 {type(e).__name__}: {e}")
            results["total"] += 1
            continue
        elapsed = (time.perf_counter() - t0) * 1000

        hit = any(k in reply.text for k in keywords)
        results["total"] += 1
        if hit:
            results["correct"] += 1
        results["latencies"].append(elapsed)
        results["prompt_tokens"].append(reply.prompt_tokens or est)
        results["completion_tokens"].append(reply.completion_tokens)
        results["details"].append(
            {
                "label": label,
                "hit": hit,
                "answer": reply.text[:60].replace("\n", " "),
                "ms": int(elapsed),
                "ptok": reply.prompt_tokens,
            }
        )
        mark = "命中" if hit else "未命中"
        print(
            f"  [{mark}] {label}  {int(elapsed)}ms  "
            f"prompt={reply.prompt_tokens} completion={reply.completion_tokens}"
        )
        print(f"          → {reply.text[:70].replace(chr(10), ' ')}")

    return results


async def run_hallucination_test(client: OpenAICompatClient, name: str) -> dict:
    """反幻觉测试：知识库里没有的信息，模型该说不知道。"""
    print(f"\n  【反幻觉测试 · {name}】")
    out = []
    for question, note in HALLUCINATION_QUESTIONS:
        bundle = await prompt_builder.assemble(
            user_id="u_nonexist_halluc",
            user_input=question,
            history=[],
        )
        reply = await client.chat(bundle.messages, temperature=0.3)
        # 编造的典型特征：出现了具体数字
        import re

        has_number = bool(re.search(r"\d+(\.\d+)?\s*(亿|万|元|%)", reply.text))
        out.append({"question": question, "answer": reply.text, "fabricated": has_number})
        print(f"    {note}：{'⚠️ 疑似编造具体数字' if has_number else '✅ 没有编造'}")
        print(f"      → {reply.text[:90].replace(chr(10), ' ')}")
    return {"details": out}


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("汇总对比")
    print("=" * 78)
    header = f"{'配置':<26}{'正确率':>10}{'P50延迟':>12}{'平均prompt':>14}{'平均输出':>12}"
    print(header)
    print("-" * 78)
    for r in rows:
        acc = f"{r['correct']}/{r['total']}"
        p50 = f"{int(statistics.median(r['latencies']))}ms" if r["latencies"] else "-"
        ptok = f"{int(statistics.mean(r['prompt_tokens']))}" if r["prompt_tokens"] else "-"
        ctok = f"{int(statistics.mean(r['completion_tokens']))}" if r["completion_tokens"] else "-"
        print(f"{r['name']:<26}{acc:>10}{p50:>12}{ptok:>14}{ctok:>12}")
    print("-" * 78)


async def main() -> int:
    args = sys.argv[1:]
    big_model = BIG_MODEL
    if "--big" in args:
        big_model = args[args.index("--big") + 1]

    await init_db()

    if not settings.llm_api_key:
        print("[FAIL] 没有 API Key")
        return 1

    print("=" * 78)
    print("模型对照实验：小模型 + 中间件 vs 大模型")
    print("=" * 78)
    print(f"  小模型：{SMALL_MODEL}")
    print(f"  大模型：{big_model}")
    print(f"  测试项：记忆 {len(MEMORY_QUESTIONS)} 题 + 知识库 {len(KB_QUESTIONS)} 题")

    small = OpenAICompatClient(model=SMALL_MODEL)
    big = OpenAICompatClient(model=big_model)

    user_id = f"u_cmp_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)
    items = [
        MemoryItem(content=c, type=t, importance=imp, valence=0.0, arousal=0.0)
        for c, t, imp in PERSONA_FACTS
    ]
    saved = await long_term.save_memories(user_id, items, semantic_dedupe=False)
    print(f"\n  已为测试用户植入 {len(saved)} 条记忆")

    rows = []

    # 配置 A：小模型 + 全量历史（朴素做法）
    rows.append(
        await run_config(
            f"A. {SMALL_MODEL} + 全量历史",
            small,
            user_id,
            use_middleware=False,
            history_turns=10,
        )
    )

    # 配置 B：小模型 + 中间件
    rows.append(
        await run_config(
            f"B. {SMALL_MODEL} + 中间件",
            small,
            user_id,
            use_middleware=True,
            history_turns=10,
        )
    )

    # 配置 C：大模型 + 中间件
    rows.append(
        await run_config(
            f"C. {big_model} + 中间件",
            big,
            user_id,
            use_middleware=True,
            history_turns=10,
        )
    )

    # 反幻觉
    await run_hallucination_test(small, SMALL_MODEL)
    await run_hallucination_test(big, big_model)

    print_table(rows)

    # 结论
    a, b, c = rows
    print("\n" + "=" * 78)
    print("怎么读这张表")
    print("=" * 78)
    print(f"  · 中间件的价值：A → B 正确率 {a['correct']}/{a['total']} → {b['correct']}/{b['total']}")
    print(f"    （全量历史里其实有答案，但小模型在长上下文里看不出来）")
    print(f"  · 大模型多花的钱：B → C 正确率 {b['correct']}/{b['total']} → {c['correct']}/{c['total']}")
    print(f"    （如果差距不大，说明这个任务上小模型 + 中间件已经够用）")
    if b["prompt_tokens"] and c["prompt_tokens"]:
        print(f"  · token 量级：B 平均 {int(statistics.mean(b['prompt_tokens']))} vs "
              f"C 平均 {int(statistics.mean(c['prompt_tokens']))}")

    await long_term.clear_user(user_id)
    await tasks.wait_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
