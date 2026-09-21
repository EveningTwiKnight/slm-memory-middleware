"""Day 2 验证脚本：短期记忆是否真的存下来、取出来、拼进提示词。

特点：**不需要 API Key**。
它用一个"回声假模型"替换掉真模型 —— 假模型会把它收到的历史告诉
我们，于是"历史有没有被正确拼进提示词"这件事变成了可以直接断言的事实。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m tests.test_memory
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.api import chat as chat_module  # noqa: E402
from app.llm.fake import EchoLLMClient  # noqa: E402
from app.main import app  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory import short_term  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    mark = OK if passed else FAIL
    print(f"  {mark} {name}" + (f" —— {detail}" if detail else ""))


async def main() -> int:
    print("=" * 66)
    print("Day 2 验证：短期记忆（存历史 / 取历史 / 拼进提示词）")
    print("=" * 66)

    await init_db()

    # 用假模型替换真模型：不需要 Key，且输出可预测
    stub = EchoLLMClient()
    chat_module.get_llm = lambda: stub  # type: ignore[assignment]

    user_id = f"u_test_{uuid.uuid4().hex[:6]}"
    session_id = f"s_test_{uuid.uuid4().hex[:6]}"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # ---------- 测试 1：第一轮对话，应该没有历史 ----------
        print("\n[1] 第一轮对话（此前没有任何历史）")
        r1 = await client.post(
            "/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": "我叫小明，是个前端工程师"},
        )
        check("接口返回 200", r1.status_code == 200, f"实际 {r1.status_code}")
        if r1.status_code != 200:
            print(r1.text)
            return 1
        d1 = r1.json()
        print(f"      模型看到：{d1['answer']}")
        check("首轮没有历史", d1["history_used"] == 0, f"history_used={d1['history_used']}")

        # ---------- 测试 2：第二轮，应该能看到 2 条历史 ----------
        print("\n[2] 第二轮对话（应该看到上一轮的一问一答）")
        r2 = await client.post(
            "/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": "我叫什么？"},
        )
        d2 = r2.json()
        print(f"      模型看到：{d2['answer']}")
        check("带上了 2 条历史", d2["history_used"] == 2, f"history_used={d2['history_used']}")
        check(
            "历史里确实有'我叫小明'",
            "我叫小明" in d2["answer"],
            "假模型把看到的历史复述出来了" if "我叫小明" in d2["answer"] else "没找到！",
        )

        # ---------- 测试 3：第三轮，历史继续累积 ----------
        print("\n[3] 第三轮对话（历史继续累积）")
        r3 = await client.post(
            "/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": "我上周说想做个副业"},
        )
        d3 = r3.json()
        check("带上了 4 条历史", d3["history_used"] == 4, f"history_used={d3['history_used']}")

        # ---------- 测试 4：数据库里真的存了 ----------
        print("\n[4] 检查数据库落盘")
        total = await short_term.count(session_id)
        check("数据库里有 6 条消息（3 轮 × 2）", total == 6, f"实际 {total}")

        # ---------- 测试 5：会话隔离 ----------
        print("\n[5] 会话隔离：新会话不该看到旧会话的历史")
        other_session = f"s_other_{uuid.uuid4().hex[:6]}"
        r5 = await client.post(
            "/v1/chat",
            json={"user_id": user_id, "session_id": other_session, "message": "你好"},
        )
        d5 = r5.json()
        check("新会话历史为 0", d5["history_used"] == 0, f"history_used={d5['history_used']}")

        # ---------- 测试 6：模拟"重启服务"后仍能读到历史 ----------
        print("\n[6] 模拟服务重启：新开一个进程级的数据库连接，再取历史")
        # 这里直接走 repo 层重新查询，等价于重启后重新读库
        reloaded = await short_term.get_recent(session_id, limit=50)
        check("重启后仍能读到 6 条历史", len(reloaded) == 6, f"实际 {len(reloaded)}")
        first = reloaded[0].content if reloaded else ""
        check("最早那条是'我叫小明'", "我叫小明" in first, f"实际首条：{first[:20]}")

        # ---------- 测试 7：历史顺序正确（时间正序） ----------
        print("\n[7] 历史顺序：必须是时间正序，否则模型看到的是倒着说的对话")
        roles = [m.role for m in reloaded]
        check(
            "顺序为 user,assistant,user,assistant...",
            roles == ["user", "assistant"] * 3,
            f"实际 {roles}",
        )

        # ---------- 测试 8：history_limit 能限制条数 ----------
        print("\n[8] history_limit 参数生效")
        r8 = await client.post(
            "/v1/chat",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "message": "只带 2 条历史",
                "history_limit": 2,
            },
        )
        d8 = r8.json()
        check("只带了 2 条历史", d8["history_used"] == 2, f"history_used={d8['history_used']}")

        # ---------- 测试 9：历史接口可查 ----------
        print("\n[9] 历史查询接口 GET /v1/sessions/{id}/messages")
        r9 = await client.get(f"/v1/sessions/{session_id}/messages")
        check("接口返回 200", r9.status_code == 200)
        d9 = r9.json()
        check("接口返回的消息数与库一致", d9["total"] == 8, f"实际 {d9['total']}")

        # ---------- 测试 10：清理 ----------
        print("\n[10] 清理测试数据")
        r10 = await client.delete(f"/v1/sessions/{session_id}")
        check("删除成功", r10.status_code == 200, r10.text)
        after = await short_term.count(session_id)
        check("删除后条数为 0", after == 0, f"实际 {after}")
        await client.delete(f"/v1/sessions/{other_session}")

    # ---------- 汇总 ----------
    passed = sum(1 for _, ok, _ in checks if ok)
    total_checks = len(checks)
    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total_checks} 项通过")
    if passed == total_checks:
        print(f"{OK} Day 2 完成！短期记忆的存、取、拼装全部正常。")
    else:
        print(f"{FAIL} 有 {total_checks - passed} 项失败：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 66)
    return 0 if passed == total_checks else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
