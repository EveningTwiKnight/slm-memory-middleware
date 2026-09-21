"""验证接入示例页面的记忆面板数据链路。

页面右上角的「看它记住了什么」按钮依赖两个接口：
  GET /v1/memory/{user_id}   用户档案 + 长期记忆
  GET /v1/emotion/{user_id}  情绪状态 + 轨迹

这个脚本验证它们返回的结构和字段，确保前端渲染不会拿到空值。
（前端是原生 JS，没法单元测试，所以在这里验证数据契约。）

用法：
    .venv\\Scripts\\python.exe -m web.verify_panel
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.memory import emotion_store, long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.memory.emotion import EmotionScore  # noqa: E402
from app.memory.extractor import MemoryItem  # noqa: E402
from web.server import app  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


async def main() -> int:
    await init_db()

    uid = f"panel_{uuid.uuid4().hex[:6]}"
    await long_term.clear_user(uid)
    await emotion_store.clear_user(uid)

    print("=" * 68)
    print("接入示例 · 记忆面板数据链路验证")
    print("=" * 68)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # ---------- 0. 空状态：新用户应该优雅返回空 ----------
        print("\n[1] 新用户（什么记忆都没有）")
        r = await client.get(f"/v1/memory/{uid}")
        check("记忆接口 200", r.status_code == 200, f"{r.status_code}")
        d = r.json()
        check("total 为 0", d["total"] == 0, f"total={d['total']}")
        check("profile 是数组", isinstance(d["profile"], list), str(type(d["profile"])))
        check("memories 是数组", isinstance(d["memories"], list))
        print(f"     返回：total={d['total']} profile={d['profile']}")

        r = await client.get(f"/v1/emotion/{uid}")
        check("情绪接口 200", r.status_code == 200, f"{r.status_code}")
        e = r.json()
        check("有 current 对象", isinstance(e["current"], dict))
        check("有 points 数组", isinstance(e["points"], list))
        print(f"     返回：current={e['current']}")

        # ---------- 1. 有数据时 ----------
        print("\n[2] 植入记忆与档案后")
        await long_term.save_memories(
            uid,
            [
                MemoryItem("林知远是算法工程师，专做推荐系统方向", "fact", 0.95),
                MemoryItem("用户正在开发一个面向中小商家的选品推荐工具", "goal", 0.90),
                MemoryItem("用户因项目进度慢而焦虑", "emotion", 0.75, valence=-0.6, arousal=0.7),
            ],
            semantic_dedupe=False,
        )
        await emotion_store.update_from_score(
            uid,
            EmotionScore(valence=-0.55, arousal=0.65, label="焦虑", confidence=1.0),
        )

        r = await client.get(f"/v1/memory/{uid}")
        d = r.json()
        check("记忆总数 > 0", d["total"] > 0, f"total={d['total']}")
        check("有档案项", len(d["profile"]) > 0, f"{len(d['profile'])} 条")
        check("记忆含要求字段",
              all(k in d["memories"][0] for k in
                  ("id", "content", "type", "importance", "valence", "arousal",
                   "emotion_label", "created_at")),
              str(list(d["memories"][0].keys())))
        print("     档案:")
        for p in d["profile"]:
            print(f"       {p['key']}：{p['value']}")
        print("     记忆:")
        for m in d["memories"][:5]:
            print(f"       [{m['importance']:.2f}] {m['content'][:32]} "
                  f"(情绪={m['emotion_label']})")

        r = await client.get(f"/v1/emotion/{uid}")
        e = r.json()
        c = e["current"]
        check("情绪标签非空", bool(c.get("label")), f"label={c.get('label')}")
        check("有效价数值", isinstance(c.get("valence"), (int, float)))
        check("有长期倾向", "stable_valence" in c)
        check("有趋势字段", "trend" in c)
        check("轨迹有点位", len(e["points"]) > 0, f"{len(e['points'])} 个")
        print(f"     当前情绪：{c['label']} 效价 {c['valence']:+.2f} 趋势 {c['trend']}")

        # ---------- 2. 页面本身 ----------
        print("\n[3] 页面与按钮")
        r = await client.get("/")
        html = r.text
        check("页面 200", r.status_code == 200)
        check("含面板按钮", 'id="showMemory"' in html)
        check("含面板容器", 'id="sidecol"' in html)
        check("含两个接口调用", "/v1/memory/" in html and "/v1/emotion/" in html)
        check("含 HTML 转义（防注入）", "&amp;" in html)

        # ---------- 3. 清空后回到空状态 ----------
        print("\n[4] 重置后应回到空状态")
        r = await client.delete(f"/v1/memory/{uid}")
        check("重置接口 200", r.status_code == 200, r.text)
        r = await client.get(f"/v1/memory/{uid}")
        check("记忆已清空", r.json()["total"] == 0)
        await emotion_store.clear_user(uid)

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 68)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} 记忆面板的数据链路正常。")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
