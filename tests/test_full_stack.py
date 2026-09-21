"""Day 5 全链路验证：记忆注入 + 知识库 RAG 同时生效。

这个脚本回答一个终极问题：
「中间件把所有零件都装好了，真模型能不能同时用上"关于你的记忆"和"知识库资料"？」

前置条件：先跑一次入库
    .venv\\Scripts\\python.exe -m scripts.ingest_kb --dir docs/kb_sample --reset

用法：
    .venv\\Scripts\\python.exe -m tests.test_full_stack
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.core import tasks  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.memory import long_term, vector_store  # noqa: E402
from app.memory.database import init_db  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


async def ask(client, user_id, session_id, msg, show=True):
    r = await client.post(
        "/v1/chat",
        json={"user_id": user_id, "session_id": session_id, "message": msg},
    )
    if r.status_code != 200:
        print(f"  {FAIL} {r.status_code} {r.text[:200]}")
        return {}
    d = r.json()
    if show:
        print(f"  我：{msg}")
        print(f"  它：{d['answer']}")
        print(
            f"      （历史{d['history_used']} 档案{d['profile_used']} "
            f"记忆{d['memories_used']} 资料{d['knowledge_used']} "
            f"prompt≈{d['est_prompt_tokens']}token {d['latency_ms']}ms）"
        )
    return d


async def main() -> int:
    print("=" * 66)
    print("Day 5 全链路：长期记忆 + 知识库 RAG")
    print("=" * 66)

    await init_db()

    kb = vector_store.stats()["knowledge"]
    if kb == 0:
        print(f"\n{FAIL} 知识库是空的，请先执行：")
        print("    .venv\\Scripts\\python.exe -m scripts.ingest_kb --dir docs/kb_sample --reset")
        return 1
    print(f"\n知识库已就绪：{kb} 块")

    if not settings.llm_api_key:
        print(f"{FAIL} 没有 API Key")
        return 1

    user_id = f"u_full_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
        # ---------- 建立记忆 ----------
        print("\n【1】建立长期记忆（自我介绍）")
        print("-" * 66)
        session_a = f"s_a_{uuid.uuid4().hex[:6]}"
        await ask(
            client, user_id, session_a,
            "我叫小林，是个后端工程师，正在做一个给视障人士用的导航类项目，最近压力有点大。",
        )
        await tasks.wait_all(timeout=60)
        n = await long_term.count(user_id)
        print(f"      （后台抽取出 {n} 条记忆）")
        check("记忆已建立", n >= 1, f"{n} 条")

        # ---------- 跨会话记忆 ----------
        print("\n【2】全新会话：它记得我吗？")
        print("-" * 66)
        session_b = f"s_b_{uuid.uuid4().hex[:6]}"
        d2 = await ask(client, user_id, session_b, "你还记得我是谁、在做什么吗？")
        ans = d2.get("answer", "")
        check("说出姓名", "小林" in ans, "说了" if "小林" in ans else "没说")
        check("说出职业", "后端" in ans, "说了" if "后端" in ans else "没说")
        check("说出项目", ("导航" in ans or "视障" in ans), "说了" if ("导航" in ans or "视障" in ans) else "没说")

        # ---------- 知识库 RAG ----------
        print("\n【3】再开新会话：问知识库里才有答案的问题")
        print("-" * 66)
        session_c = f"s_c_{uuid.uuid4().hex[:6]}"
        d3 = await ask(
            client, user_id, session_c,
            "我的中间件在做 token 预算调度时，超预算了应该按什么顺序裁剪？裁剪时哪些段落不能动？",
        )
        ans3 = d3.get("answer", "")
        check("注入了知识库资料", d3.get("knowledge_used", 0) >= 1, f"{d3.get('knowledge_used')} 条")
        check("答出了裁剪顺序（先历史再低分记忆）",
              ("历史" in ans3 or "对话" in ans3),
              "提到了" if ("历史" in ans3 or "对话" in ans3) else "没提")
        check("答出了人设与当前输入不裁剪",
              ("人设" in ans3 and ("当前输入" in ans3 or "输入" in ans3)),
              "提到了" if "人设" in ans3 else "没提")

        # ---------- 记忆 + 知识库同时生效 ----------
        print("\n【4】记忆与知识库同时生效")
        print("-" * 66)
        session_d = f"s_d_{uuid.uuid4().hex[:6]}"
        d4 = await ask(
            client, user_id, session_d,
            "结合我自己的情况，我的记忆系统应该怎么设计才合理？",
        )
        check("这轮同时注入了记忆和资料",
              d4.get("memories_used", 0) >= 1 and d4.get("knowledge_used", 0) >= 1,
              f"记忆{d4.get('memories_used')} 资料{d4.get('knowledge_used')}")

        # ---------- 看看模型实际收到什么 ----------
        print("\n【5】调试接口：模型实际收到的 system 内容")
        print("-" * 66)
        r5 = await client.get(
            "/v1/debug/prompt",
            params={"user_id": user_id, "message": "记忆的衰减半衰期是多少？"},
        )
        dbg = r5.json()
        print(f"  估算 prompt：{dbg['est_prompt_tokens']} tokens")
        print(f"  分段：{dbg['sections']}")
        sysmsg = next(m for m in dbg["messages"] if m["role"] == "system")
        for line in sysmsg["content"].split("\n"):
            print(f"  | {line}")

        print("\n【清理】")
        await long_term.clear_user(user_id)
        for s in (session_a, session_b, session_c, session_d):
            await client.delete(f"/v1/sessions/{s}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Day 5 完成！长期记忆与知识库 RAG 同时生效。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
