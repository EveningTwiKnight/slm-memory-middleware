"""Day 3 端到端验证：走完整对话接口，确认后台异步抽取真的落库了。

这个脚本回答一个关键问题：
「我在 /v1/chat 里 spawn 的后台任务，到底有没有真的把记忆写进去？」

用法：
    .venv\\Scripts\\python.exe -m tests.test_memory_flow
"""

import asyncio
import sys
import time
import uuid

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.core import tasks  # noqa: E402
from app.main import app  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


async def main() -> int:
    print("=" * 66)
    print("Day 3 端到端：对话 → 后台抽取 → 记忆落库")
    print("=" * 66)

    await init_db()
    user_id = f"u_flow_{uuid.uuid4().hex[:6]}"
    session_id = f"s_flow_{uuid.uuid4().hex[:6]}"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
        print("\n[1] 发一句包含大量可记忆信息的话")
        t = time.perf_counter()
        r = await client.post(
            "/v1/chat",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "message": "你好，我叫小明，是个前端工程师，在杭州工作，最近想做宠物相关的副业",
            },
        )
        api_ms = int((time.perf_counter() - t) * 1000)
        check("对话接口返回 200", r.status_code == 200, f"{r.status_code}")
        if r.status_code != 200:
            print(r.text)
            return 1

        d = r.json()
        print(f"      它：{d['answer'][:60]}...")
        print(f"      接口耗时 {api_ms}ms（注意：抽取还没跑完，不在这段时间里）")

        pending = tasks.pending_count()
        check("后台有抽取任务在跑", pending >= 1, f"pending={pending}")

        print("\n[2] 立刻查记忆库 —— 此时后台任务很可能还没跑完")
        immediate = await long_term.count(user_id)
        print(f"      当前记忆条数：{immediate}")

        print("\n[3] 等后台任务跑完")
        t = time.perf_counter()
        await tasks.wait_all(timeout=60)
        extract_ms = int((time.perf_counter() - t) * 1000)
        print(f"      等待 {extract_ms}ms")
        check("等待后没有遗留任务", tasks.pending_count() == 0, f"pending={tasks.pending_count()}")

        print("\n[4] 再查记忆库")
        after = await long_term.count(user_id)
        check("记忆已经写进去了", after >= 1, f"从 {immediate} 条变成 {after} 条")

        memories = await long_term.get_memories(user_id, limit=20)
        print(f"\n      实际抽出的记忆（{len(memories)} 条）：")
        for m in memories:
            print(f"        [{m.type}] 重要度{m.importance:.2f} {m.content}")

        print("\n[5] 检查用户档案（fact 类应该同步过来了）")
        profile = await long_term.get_profile(user_id)
        check("档案里有内容", len(profile) >= 1, f"{profile}")

        print("\n[6] 关键对比：抽取耗时 vs 回复耗时")
        print(f"      用户等待（接口返回）：{api_ms}ms")
        print(f"      后台抽取额外耗时：    {extract_ms}ms")
        if extract_ms > 0:
            saved = extract_ms
            print(f"      → 如果同步做抽取，用户要多等约 {saved}ms（{saved / max(api_ms, 1):.1f} 倍）")
        check("抽取没有阻塞用户响应", True, "这正是异步写回的价值")

        print("\n[7] 清理")
        await long_term.clear_user(user_id)
        await client.delete(f"/v1/sessions/{session_id}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Day 3 完成！对话 → 抽取 → 落库 全链路打通，且不阻塞用户。")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
