"""验证 Web 接入示例的 SSE 链路（模拟浏览器的读法）。

为什么要单独验证这个？
`web/server.py` 里的前端代码是"浏览器怎么读 SSE"的参考实现，
但浏览器里调试很麻烦。这个脚本用 httpx 模拟浏览器的读法：

  1. POST /v1/chat/stream，拿流式响应
  2. 按 SSE 帧格式（"data: {...}\\n\\n"）切分
  3. 解析出 meta / delta / done / error 四种事件
  4. 断言：meta 里有 session_id、delta 能拼出完整回答、done 有延迟数据

这验证的正是前端同学接手时要写的核心逻辑（约 30 行代码）。

用法：
    .venv\\Scripts\\python.exe -m web.verify
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.core import tasks  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from web.server import app  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


async def read_sse(client, payload: dict) -> dict:
    """完全按前端的方式读一遍 SSE 流。

    返回 {meta, text, done, errors, first_chunk_ms}
    """
    meta = None
    text = ""
    done = None
    errors: list[str] = []
    first_chunk_ms = 0
    t0 = time.perf_counter()
    frames = 0

    async with client.stream("POST", "/v1/chat/stream", json=payload) as resp:
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}

        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            # SSE 帧以空行分隔 —— 前端也是这么切的
            parts = buf.split("\n\n")
            buf = parts.pop()
            for part in parts:
                line = part.strip()
                if not line.startswith("data:"):
                    continue
                frames += 1
                try:
                    evt = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "meta":
                    meta = evt
                elif t == "delta":
                    if not first_chunk_ms:
                        first_chunk_ms = int((time.perf_counter() - t0) * 1000)
                    text += evt["text"]
                elif t == "done":
                    done = evt
                elif t == "error":
                    errors.append(evt.get("message", ""))

    return {
        "meta": meta,
        "text": text,
        "done": done,
        "errors": errors,
        "frames": frames,
        "first_chunk_ms": first_chunk_ms,
    }


async def main() -> int:
    await init_db()

    if not settings.llm_api_key:
        print(f"{FAIL} 没有 API Key")
        return 1

    user_id = f"web_{uuid.uuid4().hex[:6]}"
    session_id = f"web_s_{uuid.uuid4().hex[:6]}"

    print("=" * 70)
    print("Web 接入示例 · SSE 链路验证")
    print("=" * 70)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180) as client:
        # ---------- 0. 页面能打开 ----------
        print("\n[0] 页面与健康检查")
        r = await client.get("/")
        check("首页返回 200", r.status_code == 200, f"{r.status_code}")
        check("页面含 SSE 读取逻辑", "ReadableStream" in r.text)
        check("页面含打字机效果", "cursor" in r.text)

        h = await client.get("/health")
        check("健康检查正常", h.status_code == 200)
        check("接口已挂载到同一应用", h.json().get("status") == "ok")

        # ---------- 1. 第一轮：建立记忆 ----------
        print("\n[1] 第一轮对话（建立记忆）")
        res1 = await read_sse(
            client,
            {
                "user_id": user_id,
                "session_id": session_id,
                "message": "你好，我叫苏晚，在做数据分析，最近在学 Rust，有点吃力",
            },
        )
        check("没有错误事件", not res1["errors"], str(res1["errors"])[:60])
        check("收到 meta 事件", res1["meta"] is not None)
        if res1["meta"]:
            check("meta 带 session_id", bool(res1["meta"].get("session_id")))
            check("meta 带 trace_id", bool(res1["meta"].get("trace_id")))
            check(
                "meta 带 token 估算",
                res1["meta"].get("est_prompt_tokens", 0) > 0,
                f"{res1['meta'].get('est_prompt_tokens')}",
            )
        check("收到 done 事件", res1["done"] is not None)
        check("回答非空", len(res1["text"]) > 5, f"{len(res1['text'])} 字")
        if res1["done"]:
            check(
                "done 带首字延迟",
                res1["done"].get("first_token_ms", 0) > 0,
                f"{res1['done'].get('first_token_ms')}ms",
            )
            check("done 带总耗时", res1["done"].get("latency_ms", 0) > 0,
                  f"{res1['done'].get('latency_ms')}ms")
        print(f"     流式帧数：{res1['frames']}")
        print(f"     回答：{res1['text'][:70]}")

        # 等后台抽取
        await tasks.wait_all(timeout=90)
        n = await long_term.count(user_id)
        print(f"     后台抽取 {n} 条记忆")
        check("记忆已建立", n >= 1, f"{n} 条")

        # ---------- 2. 新会话：跨会话记忆 ----------
        print("\n[2] 换新会话（模拟前端点「开新会话」），问它记不记得")
        new_session = f"web_s_{uuid.uuid4().hex[:6]}"
        res2 = await read_sse(
            client,
            {
                "user_id": user_id,
                "session_id": new_session,
                "message": "你还记得我是谁吗？我在学什么？",
            },
        )
        check("新会话里注入了记忆",
              (res2["meta"] or {}).get("memories_used", 0) >= 1,
              f"{(res2['meta'] or {}).get('memories_used')} 条")
        check("说出了姓名", "苏晚" in res2["text"],
              "说了" if "苏晚" in res2["text"] else "没说")
        check("说出了在学的东西", "Rust" in res2["text"] or "rust" in res2["text"].lower(),
              "说了" if "Rust" in res2["text"] else "没说")
        print(f"     回答：{res2['text'][:100]}")

        # ---------- 3. 流式体验指标 ----------
        print("\n[3] 流式体验指标（前端会用这两个数显示在气泡下面）")
        if res2["done"]:
            first = res2["done"].get("first_token_ms", 0)
            total = res2["done"].get("latency_ms", 0)
            print(f"     首字延迟 {first}ms　总耗时 {total}ms　字符数 {res2['done'].get('chars')}")
            check("首字延迟早于总耗时（确实在流式）", 0 < first <= total,
                  f"{first} <= {total}")
            check("首字延迟小于 1 秒", first < 1000, f"{first}ms")
            ratio = first / total if total else 1
            print(f"     → 用户感知等待缩短约 {(1 - ratio) * 100:.0f}%")

        # ---------- 4. 重置按钮对应的接口 ----------
        print("\n[4] 「重置我的记忆」按钮对应的接口")
        d = await client.delete(f"/v1/memory/{user_id}")
        check("重置接口返回 200", d.status_code == 200)
        check("返回了删除数量", "memories_deleted" in d.json(), str(d.json()))
        after = await long_term.count(user_id)
        check("记忆确实清空", after == 0, f"剩余 {after}")

        # ---------- 5. 对比：非流式接口也能用 ----------
        print("\n[5] 非流式接口（前端如果想简单点可以只用这个）")
        t0 = time.perf_counter()
        r5 = await client.post(
            "/v1/chat",
            json={"user_id": user_id, "message": "简单介绍一下你自己"},
        )
        ms5 = int((time.perf_counter() - t0) * 1000)
        check("非流式返回 200", r5.status_code == 200)
        check("非流式也能用", bool(r5.json().get("answer")), f"{ms5}ms")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 70)
    print(f"结果：{passed}/{total} 项通过")
    if passed == total:
        print(f"{OK} Web 接入链路正常。")
        print("\n启动方式（在你自己的机器上）：")
        print("  .venv\\Scripts\\python.exe -m web.server")
        print("  打开 http://127.0.0.1:8080")
    else:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
