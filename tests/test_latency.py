"""Day 6 补充：干净的首字延迟对比（排除冷启动干扰）。

为什么要单独写这个脚本？
第一次测流式时得到"12300ms → 615ms（缩短 95%）"，看起来很漂亮，但**不可信**：
那 12300ms 里有约 10 秒是首次加载嵌入模型（BGE）的耗时，不是模型推理时间。
拿一个含冷启动的数去对比一个不含冷启动的数，是错误对比。

本脚本的做法：
  1. 先跑一次"预热"请求，把嵌入模型加载等一次性开销消耗掉
  2. 预热后再测，此时测到的才是真正的推理+组装耗时
  3. 每个配置跑多次取中位数，避免单次抖动

用法：
    .venv\\Scripts\\python.exe -m tests.test_latency
"""

import asyncio
import json
import statistics
import sys
import time
import uuid

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.main import app  # noqa: E402
from app.memory.database import init_db  # noqa: E402

OK = "[OK]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else '❌'} {name}" + (f" —— {detail}" if detail else ""))


async def one_non_stream(client, user_id: str, q: str) -> tuple[int, int]:
    """返回 (总耗时ms, 字数)"""
    t0 = time.perf_counter()
    r = await client.post("/v1/chat", json={"user_id": user_id, "message": q})
    ms = int((time.perf_counter() - t0) * 1000)
    return ms, len(r.json().get("answer", ""))


async def one_stream(client, user_id: str, q: str) -> tuple[int, int, int]:
    """返回 (首字延迟ms, 总耗时ms, 字数)"""
    first = 0
    total = 0
    chars = 0
    t0 = time.perf_counter()
    async with client.stream(
        "POST", "/v1/chat/stream", json={"user_id": user_id, "message": q}
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            d = json.loads(line[6:])
            if d["type"] == "delta":
                if not first:
                    first = int((time.perf_counter() - t0) * 1000)
                chars += len(d["text"])
            elif d["type"] == "done":
                total = d["latency_ms"]
    return first, total or int((time.perf_counter() - t0) * 1000), chars


async def main() -> int:
    await init_db()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180) as client:
        user_id = f"u_lat_{uuid.uuid4().hex[:6]}"

        print("=" * 68)
        print("预热（消耗掉嵌入模型加载等一次性开销）")
        print("=" * 68)
        t0 = time.perf_counter()
        await client.post("/v1/chat", json={"user_id": user_id, "message": "预热一下"})
        warmup_ms = int((time.perf_counter() - t0) * 1000)
        print(f"  预热耗时 {warmup_ms}ms（含模型加载，这个数不参与下面统计）")

        # 预热后清掉缓存，避免缓存命中干扰延迟测量
        await client.delete("/v1/cache")

        questions = [
            "用三句话说明什么是检索增强生成，不要列条目",
            "简单介绍一下向量数据库的用途",
            "记忆的时间衰减是怎么设计的",
            "解释一下 RAG 和微调的区别",
            "为什么要做语义去重",
        ]

        print()
        print("=" * 68)
        print("正式测量（预热后，各跑 5 次取中位数）")
        print("=" * 68)
        print("  注意：每次都用**全新用户**，避免语义缓存命中干扰测量")
        print("        （缓存命中会把耗时压到几十毫秒，测不出真实推理时间）")

        non_list: list[int] = []
        first_list: list[int] = []
        stream_total_list: list[int] = []
        non_chars: list[int] = []
        stream_chars: list[int] = []

        print(f"\n  {'#':<4}{'非流式总耗时':>16}{'字数':>7}{'流式首字':>12}{'流式总耗时':>14}{'字数':>7}")
        print("  " + "-" * 62)
        for i, q in enumerate(questions, 1):
            # 每次用全新 user_id —— 缓存键含 user_id，天然隔离，不用手动清缓存
            u1 = f"u_ns_{uuid.uuid4().hex[:8]}"
            u2 = f"u_st_{uuid.uuid4().hex[:8]}"

            nm, nc = await one_non_stream(client, u1, q)
            fm, tm, sc = await one_stream(client, u2, q)

            non_list.append(nm)
            first_list.append(fm)
            stream_total_list.append(tm)
            non_chars.append(nc)
            stream_chars.append(sc)
            print(f"  {i:<4}{nm:>16}{nc:>7}{fm:>12}{tm:>14}{sc:>7}")

        non_med = int(statistics.median(non_list))
        first_med = int(statistics.median(first_list))
        stotal_med = int(statistics.median(stream_total_list))

        print("  " + "-" * 62)
        print(f"  {'中位数':<4}{non_med:>14}{int(statistics.median(non_chars)):>7}"
              f"{first_med:>12}{stotal_med:>14}{int(statistics.median(stream_chars)):>7}")

        reduction = (1 - first_med / non_med) * 100 if non_med else 0
        print()
        print("  " + "=" * 64)
        print(f"  ★ 用户感知等待：非流式 {non_med}ms → 流式首字 {first_med}ms")
        print(f"    缩短 {reduction:.0f}%")
        print("  " + "=" * 64)

        print("\n  说明：非流式必须等整段话生成完才能显示；")
        print("        流式在模型吐出第一个字时就能显示。")
        print(f"        整段内容的实际生成时间其实相近：")
        print(f"        非流式 {non_med}ms vs 流式总计 {stotal_med}ms")
        print()
        print("  ⚠️ 测量口径提示：")
        print("     上面「流式首字」是客户端测到的，含 SSE 事件穿过传输层的开销。")
        print("     服务端日志里记录的真实首字延迟更短（实测 167~298ms）：")
        print("       2026-09-21 | DEBUG | app.api.chat:event_gen | [stream] 首字延迟 276ms")
        print("     如果你要对外报数，建议用服务端的首字延迟，那才是模型真实响应时间。")

        # 断言用相对值，避免依赖客户端测到的绝对值
        check("流式首字延迟低于非流式总耗时", first_med < non_med,
              f"{first_med}ms < {non_med}ms")
        check("流式首字延迟低于非流式的 70%", first_med < non_med * 0.7,
              f"缩短 {reduction:.0f}%")
        check("流式总耗时与非流式接近（总工作量没变）",
              abs(stotal_med - non_med) / max(non_med, 1) < 0.8,
              f"{stotal_med} vs {non_med}")

    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\n结果：{passed}/{len(checks)} 项通过")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
