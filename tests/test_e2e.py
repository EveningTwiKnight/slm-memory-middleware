"""端到端验证：用**真模型**验证跨会话/跨重启的记忆能力。

和 test_memory.py 的区别：
  test_memory.py  用假模型，验证"管道通不通"（不需要 Key，快、可断言）
  本脚本          用真模型，验证"模型真的用上了历史"（需要 Key，慢）

本脚本模拟的正是 Day 7 演示视频里要录的那个场景：
  新会话自我介绍 → 换一个全新会话 → 它还记得吗？

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m tests.test_e2e
"""

import asyncio
import sys
import time
import uuid

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.main import app  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory import short_term  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"


async def ask(client: httpx.AsyncClient, user_id: str, session_id: str, msg: str) -> dict:
    """发一句话，打印问答，返回响应。"""
    t = time.perf_counter()
    r = await client.post(
        "/v1/chat",
        json={"user_id": user_id, "session_id": session_id, "message": msg},
    )
    ms = int((time.perf_counter() - t) * 1000)
    if r.status_code != 200:
        print(f"  {FAIL} 请求失败 {r.status_code}: {r.text}")
        return {}
    d = r.json()
    print(f"  我：{msg}")
    print(f"  它：{d['answer']}")
    print(f"      （{ms}ms，带 {d['history_used']} 条历史，token {d['tokens']['total']}）")
    return d


async def main() -> int:
    print("=" * 66)
    print("端到端验证：真模型 + 跨会话记忆")
    print("=" * 66)

    await init_db()

    user_id = f"u_e2e_{uuid.uuid4().hex[:6]}"
    session_a = f"s_a_{uuid.uuid4().hex[:6]}"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
        print("\n【第 1 步】会话 A：自我介绍，建立上下文")
        print("-" * 66)
        await ask(client, user_id, session_a, "你好，我叫小明，是个前端工程师，最近在做一个宠物相关的副业")
        await ask(client, user_id, session_a, "你觉得我该先做哪个功能？")

        print("\n【第 2 步】同一个会话 A：追问，验证上下文连贯")
        print("-" * 66)
        d = await ask(client, user_id, session_a, "我叫什么？我做什么工作？")

        print("\n【第 3 步】模拟重启：重新读库，确认数据落盘")
        print("-" * 66)
        total = await short_term.count(session_a)
        print(f"  会话 A 已落库 {total} 条消息")

        print("\n【第 4 步】全新会话 B：它还记得吗？")
        print("-" * 66)
        session_b = f"s_b_{uuid.uuid4().hex[:6]}"
        print("  （新会话看不到会话 A 的历史，所以这一步测的是'记不记得'的边界）")
        await ask(client, user_id, session_b, "你还记得我是谁吗？")

        # 结论判定
        print("\n" + "=" * 66)
        answer = (d.get("answer") or "")
        remembered = ("小明" in answer) or ("前端" in answer)
        if remembered:
            print(f"{OK} 模型在上下文里用上了'我叫小明/前端工程师' —— 短期记忆生效")
        else:
            print(f"{FAIL} 模型没有复述出姓名/职业，回答是：{answer}")
        print("=" * 66)
        print("\n说明：这一步只验证了『同一会话内的短期记忆』。")
        print("『跨会话长期记忆』要等 Day 3~4（记忆抽取 + 记忆注入）才具备。")

        # 清理
        await client.delete(f"/v1/sessions/{session_a}")
        await client.delete(f"/v1/sessions/{session_b}")

    return 0 if remembered else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
