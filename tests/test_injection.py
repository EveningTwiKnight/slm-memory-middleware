"""Day 4 验证：跨会话记忆（整个项目的招牌能力）。

分两部分：
  A. 离线测试（不需要 Key）：提示词组装、时间标记、预算分段统计
  B. 在线测试（需要 Key）：**真模型 + 全新会话**，看它记不记得你

Day 4 之前 vs 之后，模型对"你还记得我吗"的回答：
  之前：我无法记住用户的具体身份，但可以继续为你提供帮助。
  之后：你叫小明，是做前端的，之前在杭州，还提过想做宠物相关的副业。

用法：
    .venv\\Scripts\\python.exe -m tests.test_injection
    .venv\\Scripts\\python.exe -m tests.test_injection --offline
"""

import asyncio
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.core import tasks  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from app.memory.models import Memory  # noqa: E402
from app.orchestration import prompt_builder as pb  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# ============================================================ A. 离线测试
def test_relative_time() -> None:
    print("\n" + "=" * 66)
    print("A. 离线测试：时间标记翻译（模型不会算时间，我们要替它算好）")
    print("=" * 66)

    now = datetime(2026, 9, 21, 12, 0, 0)
    cases = [
        (now - timedelta(seconds=10), "刚刚"),
        (now - timedelta(minutes=5), "5 分钟前"),
        (now - timedelta(hours=3), "3 小时前"),
        (now - timedelta(days=1), "昨天"),
        (now - timedelta(days=5), "5 天前"),
        (now - timedelta(days=45), "1 个月前"),
        (now - timedelta(days=400), "1 年前"),
        (None, "时间未知"),
    ]
    for dt, expect in cases:
        got = pb.relative_time(dt, now)
        check(f"{expect}", got == expect, f"实际 {got}")


def test_formatting() -> None:
    print("\n" + "=" * 66)
    print("A2. 离线测试：档案与记忆的格式化")
    print("=" * 66)

    print("\n[1] 空档案/空记忆不产生多余段落")
    out = pb.format_profile([])
    check("空档案返回空串", out == "")
    check("空记忆返回空串", pb.format_memories([]) == "")

    print("\n[2] 档案格式化")
    out = pb.format_profile([{"key": "姓名", "value": "小明"}, {"key": "职业", "value": "前端工程师"}])
    print(f"      {out!r}")
    check("包含姓名行", "- 姓名：小明" in out)
    check("包含职业行", "- 职业：前端工程师" in out)

    print("\n[3] 记忆带类型和时间标记")
    now = datetime.now()
    m = Memory(
        user_id="u", content="用户想做宠物AI助手副业", type="goal",
        importance=0.9, valence=0.0, arousal=0.0,
    )
    m.id = 1
    m.created_at = now - timedelta(days=3)
    out = pb.format_memories([m], now)
    print(f"      {out!r}")
    check("带类型标记", "[目标" in out)
    check("带时间标记", "3 天前" in out)
    check("带内容", "宠物AI助手" in out)


def test_bundle() -> None:
    print("\n" + "=" * 66)
    print("A3. 离线测试：完整提示词组装")
    print("=" * 66)

    now = datetime.now()
    mem = Memory(user_id="u", content="用户叫小明", type="fact", importance=0.95, valence=0.0, arousal=0.0)
    mem.id = 42
    mem.created_at = now - timedelta(days=1)

    history = [
        pb.Message("user", "你好"),
        pb.Message("assistant", "你好！有什么可以帮你的？"),
    ]

    bundle = pb.build_messages(
        user_input="你还记得我是谁吗？",
        history=history,
        profile=[{"key": "姓名", "value": "小明"}],
        memories=[mem],
    )

    print(f"\n[4] 消息条数与顺序")
    check("共 4 条消息（system+2历史+user）", len(bundle.messages) == 4, f"实际 {len(bundle.messages)}")
    check("第 1 条是 system", bundle.messages[0].role == "system")
    check("最后 1 条是当前输入", bundle.messages[-1].content == "你还记得我是谁吗？")
    check("历史顺序保持", bundle.messages[1].content == "你好")

    print(f"\n[5] system 内容包含哪些段落")
    system = bundle.messages[0].content
    print("      " + system.replace("\n", "\n      ")[:500])
    check("含人设", "耐心、务实" in system)
    check("含用户档案段", "关于这位用户" in system)
    check("含记忆段", "你可能记得的相关信息" in system)
    check("含具体记忆内容", "用户叫小明" in system)
    check("含反幻觉提示", "不要编造" in system)

    print(f"\n[6] 统计信息")
    check("memory_ids 正确", bundle.memory_ids == [42], f"实际 {bundle.memory_ids}")
    check("memories_used=1", bundle.memories_used == 1)
    check("profile_used=1", bundle.profile_used == 1)
    check("history_used=2", bundle.history_used == 2)
    check("有分段统计", set(bundle.sections) >= {"persona", "profile", "memories", "history", "input"},
          f"实际 {list(bundle.sections)}")
    print(f"      分段 token: {bundle.sections}")
    check("估算 token > 0", bundle.est_prompt_tokens > 0, f"{bundle.est_prompt_tokens}")

    print("\n[7] 没有档案和记忆时（新用户）不该出现空段落")
    b2 = pb.build_messages(user_input="你好", history=[], profile=[], memories=[])
    s2 = b2.messages[0].content
    check("不含档案段", "关于这位用户" not in s2)
    check("不含记忆段", "你可能记得" not in s2)
    check("只有 2 条消息", len(b2.messages) == 2, f"实际 {len(b2.messages)}")


# ============================================================ B. 在线测试
async def test_cross_session() -> None:
    print("\n" + "=" * 66)
    print("B. 在线测试：跨会话记忆（需要 API Key）")
    print("=" * 66)

    if not settings.llm_api_key:
        print(f"  {FAIL} 没有 API Key，跳过")
        checks.append(("跨会话记忆", False, "缺 API Key"))
        return

    user_id = f"u_cross_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(user_id)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
        # ---------- 会话 A：自我介绍 ----------
        print("\n【会话 A】第一次见面，用户自我介绍")
        print("-" * 66)
        session_a = f"s_a_{uuid.uuid4().hex[:6]}"
        r = await client.post(
            "/v1/chat",
            json={
                "user_id": user_id,
                "session_id": session_a,
                "message": "你好，我叫小明，是个前端工程师，在杭州工作。我最近想做宠物相关的副业，但有点担心自己坚持不下来。",
            },
        )
        if r.status_code != 200:
            print(f"  {FAIL} 请求失败：{r.status_code} {r.text}")
            checks.append(("会话A对话", False, r.text[:100]))
            return
        d = r.json()
        print(f"  它：{d['answer'][:80]}...")

        # 等后台抽取完成
        await tasks.wait_all(timeout=60)
        n = await long_term.count(user_id)
        print(f"  （后台已抽取出 {n} 条记忆）")
        check("抽取到记忆", n >= 1, f"{n} 条")

        # ---------- 会话 B：全新会话 ----------
        print("\n【会话 B】全新会话（模拟关掉页面、第二天再来）")
        print("-" * 66)
        session_b = f"s_b_{uuid.uuid4().hex[:6]}"
        r2 = await client.post(
            "/v1/chat",
            json={
                "user_id": user_id,
                "session_id": session_b,
                "message": "你还记得我是谁吗？我之前跟你说过什么？",
            },
        )
        d2 = r2.json()
        print(f"  我：你还记得我是谁吗？我之前跟你说过什么？")
        print(f"  它：{d2['answer']}")
        print(f"      （注入 {d2['memories_used']} 条记忆 + {d2['profile_used']} 条档案）")

        answer = d2["answer"]
        check("新会话里注入了记忆", d2["memories_used"] >= 1, f"{d2['memories_used']} 条")
        check("模型说出了姓名", "小明" in answer, "说了" if "小明" in answer else "没说")
        check("模型说出了职业", "前端" in answer, "说了" if "前端" in answer else "没说")
        check("模型提到了副业/宠物", ("副业" in answer or "宠物" in answer),
              "提到了" if ("副业" in answer or "宠物" in answer) else "没提")

        # ---------- 情绪记忆检查 ----------
        print("\n【会话 C】再开一个新会话，问它我担心什么")
        print("-" * 66)
        session_c = f"s_c_{uuid.uuid4().hex[:6]}"
        r3 = await client.post(
            "/v1/chat",
            json={
                "user_id": user_id,
                "session_id": session_c,
                "message": "我之前有没有提过我担心什么？",
            },
        )
        d3 = r3.json()
        print(f"  它：{d3['answer']}")

        # ---------- 查看注入的原始内容 ----------
        print("\n【调试接口】看看模型实际收到了什么")
        print("-" * 66)
        r4 = await client.get(
            "/v1/debug/prompt",
            params={"user_id": user_id, "message": "我叫什么？"},
        )
        dbg = r4.json()
        print(f"  估算 prompt tokens：{dbg['est_prompt_tokens']}")
        print(f"  分段占用：{dbg['sections']}")
        print(f"  统计：{dbg['stats']}")
        system_msg = next(m for m in dbg["messages"] if m["role"] == "system")
        print("\n  ---- system 内容 ----")
        for line in system_msg["content"].split("\n"):
            print(f"  | {line}")

        check("debug 接口返回 system 段", "关于这位用户" in system_msg["content"] or "你可能记得" in system_msg["content"])

        print("\n【清理】")
        await long_term.clear_user(user_id)
        for s in (session_a, session_b, session_c):
            await client.delete(f"/v1/sessions/{s}")


async def main() -> int:
    offline_only = "--offline" in sys.argv
    await init_db()

    test_relative_time()
    test_formatting()
    test_bundle()

    if not offline_only:
        await test_cross_session()
    await tasks.wait_all()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Day 4 完成！记忆注入生效，跨会话记忆打通。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
