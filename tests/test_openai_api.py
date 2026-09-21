"""验证 OpenAI 兼容入口（不需要 API Key）。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m tests.test_openai_api

为什么要单独测这个？
这个入口是给"别人的客户端"用的（AstrBot / Cherry Studio / NextChat …），
它们只认 OpenAI 的协议格式。协议一旦拼错，客户端会报"未知错误"，
排查起来很痛苦。所以这里逐项验协议：

  1. GET  /v1/models                客户端拉模型列表
  2. POST /v1/chat/completions      非流式的响应结构
  3. 历史是否正确传入（客户端带的历史不能被吞掉）
  4. 客户端自带的人格是否能覆盖默认人设
  5. user 字段 → user_id，会话是否正确落库
  6. 流式：chunk 协议 + [DONE] 结尾
  7. 模型失败时是否返回 OpenAI 格式的 error（而不是 500 堆栈）
  8. Bearer 校验

它用"照镜子"的假模型替换真模型，所以没配 Key 也能跑，而且能断言内容。
"""

import asyncio
import json
import sys

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.api import openai_api  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.llm.base import LLMError  # noqa: E402
from app.llm.fake import EchoLLMClient  # noqa: E402
from app.main import app  # noqa: E402
from app.memory import long_term, short_term  # noqa: E402
from app.memory.database import init_db  # noqa: E402
from app.orchestration.prompt_builder import DEFAULT_PERSONA  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"  {OK if passed else FAIL} {name}" + (f" —— {detail}" if detail else ""))


class BoomLLM:
    """一个只会失败的假模型，用来验错误返回格式。"""

    async def chat(self, messages, **kwargs):
        raise LLMError("测试用的故意失败")

    async def stream(self, messages, **kwargs):
        raise LLMError("测试用的故意失败")
        yield ""  # pragma: no cover


async def complete(client, payload, headers=None) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions", json=payload, headers=headers or {}, timeout=120
    )


async def read_sse(client, payload, headers=None) -> tuple[int, list[dict], bool]:
    """按客户端的方式读一遍流式响应。"""
    status = 0
    events: list[dict] = []
    saw_done = False
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers or {}, timeout=120
    ) as resp:
        status = resp.status_code
        buf = ""
        async for piece in resp.aiter_text():
            buf += piece
            frames = buf.split("\n\n")
            buf = frames.pop()
            for frame in frames:
                line = frame.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                events.append(json.loads(data))
    return status, events, saw_done


async def main() -> int:
    await init_db()

    # 关掉抽取：这是后台任务，会让断言不稳定（抽取器还会去调模型）
    settings.enable_extraction = False

    stub = EchoLLMClient()
    openai_api.get_llm = lambda: stub  # type: ignore[assignment]

    uid = "u_openai_test"
    # 流式那段换个 user，避免命中 [2] 写进缓存的那条（否则"等于非流式结果"这个
    # 断言会因为缓存直接吐一模一样的内容而变得没意义）
    uid_stream = "u_openai_test_stream"
    await long_term.clear_user(uid)
    await long_term.clear_user(uid_stream)
    await short_term.clear_session(f"oai_{uid}")
    await short_term.clear_session(f"oai_{uid_stream}")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 68)
        print("OpenAI 兼容入口验证")
        print("=" * 68)

        # ---------- 1. 模型列表 ----------
        print("\n[1] GET /v1/models（客户端接之前会先拉这个）")
        r = await client.get("/v1/models")
        check("返回 200", r.status_code == 200, str(r.status_code))
        ids = [m["id"] for m in r.json().get("data", [])]
        check("是 OpenAI 的 list 结构", r.json().get("object") == "list")
        check(f"含配置的模型名 {settings.llm_model}", settings.llm_model in ids, str(ids))
        check("含中间件别名", "slm-memory-middleware" in ids, str(ids))

        # ---------- 2. 非流式 ----------
        print("\n[2] POST /v1/chat/completions（非流式）")
        r = await complete(client, {"model": settings.llm_model, "user": uid,
                                    "messages": [{"role": "user", "content": "我叫小明"}]})
        check("返回 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("object 正确", body.get("object") == "chat.completion", str(body.get("object")))
        check("choices 非空", bool(body.get("choices")), str(body)[:80])
        text2 = body["choices"][0]["message"]["content"]
        check("role 是 assistant", body["choices"][0]["message"]["role"] == "assistant")
        check("finish_reason 有值", body["choices"][0].get("finish_reason") == "stop")
        check("内容非空（走到了模型）", "[回声]" in text2, text2[:60])
        check("usage 是 OpenAI 结构",
              {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(body.get("usage", {})),
              str(body.get("usage")))

        # ---------- 3. 历史不能被吞 ----------
        print("\n[3] 客户端带来的历史要原样传给模型")
        r = await complete(client, {
            "user": uid,
            "messages": [
                {"role": "user", "content": "我叫小明"},
                {"role": "assistant", "content": "你好小明"},
                {"role": "user", "content": "我叫什么？"},
            ],
        })
        text3 = r.json()["choices"][0]["message"]["content"]
        check("历史 2 条被传入", "我看到 2 条历史消息" in text3, text3[:70])
        check("本轮输入取的是最后一条 user", "最新一句：我叫什么？" in text3, text3[:70])

        # ---------- 4. 客户端人格覆盖默认人设 ----------
        print("\n[4] 客户端自带的人格要覆盖中间件的默认人设")
        persona = "你是群里的猫娘助手，说话要短。"
        r = await complete(client, {
            "user": uid,
            "messages": [
                {"role": "system", "content": persona},
                {"role": "user", "content": "在吗"},
            ],
        })
        text4 = r.json()["choices"][0]["message"]["content"]
        check("用的是客户端人设（长度对得上）", f"人设长度={len(persona)}" in text4, text4[:70])
        check("不是内置默认人设", len(persona) != len(DEFAULT_PERSONA),
              f"客户端 {len(persona)} vs 默认 {len(DEFAULT_PERSONA)}")

        # ---------- 5. user 字段 → user_id，会话落库 ----------
        print("\n[5] user 字段要能决定记忆归属")
        sid = f"oai_{uid}"
        r = await client.get(f"/v1/sessions/{sid}/messages")
        msgs = r.json()
        check("该用户的会话已落库", msgs.get("total", 0) >= 2, f"total={msgs.get('total')}")
        roles = [m["role"] for m in msgs.get("messages", [])]
        check("落的是 user/assistant 成对消息",
              roles[:2] == ["user", "assistant"], str(roles[:4]))

        # ---------- 6. 流式 ----------
        print("\n[6] 流式：chunk 协议 + [DONE] 结尾")
        status, events, saw_done = await read_sse(client, {
            "user": uid_stream, "stream": True,
            "messages": [{"role": "user", "content": "我叫小明"}],
        })
        check("返回 200", status == 200, str(status))
        check("有 chunk 事件", len(events) > 0, f"{len(events)} 个")
        check("object 是 chat.completion.chunk",
              all(e.get("object") == "chat.completion.chunk" for e in events))
        check("第一个 chunk 带 role=assistant",
              events[0]["choices"][0]["delta"].get("role") == "assistant", str(events[0])[:90])
        streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        check("拼出的内容等于非流式结果", streamed == text2, streamed[:60])
        check("最后一个 chunk 的 finish_reason=stop",
              events[-1]["choices"][0].get("finish_reason") == "stop", str(events[-1])[:90])
        check("以 [DONE] 结尾", saw_done)

        # ---------- 7. 模型失败时的错误格式 ----------
        print("\n[7] 模型挂掉时要返回 OpenAI 格式的 error，而不是 500 堆栈")
        openai_api.get_llm = lambda: BoomLLM()  # type: ignore[assignment]
        r = await complete(client, {"user": uid,
                                    "messages": [{"role": "user", "content": "会失败"}]})
        check("状态码 502", r.status_code == 502, str(r.status_code))
        err = r.json().get("error", {})
        check("错误体是 OpenAI 结构", "message" in err and "code" in err, str(err)[:90])
        openai_api.get_llm = lambda: stub  # type: ignore[assignment]

        # ---------- 8. Bearer 校验 ----------
        print("\n[8] 配了 OPENAI_COMPAT_TOKEN 之后要校验 Bearer")
        settings.openai_compat_token = "tok_test_123"
        payload = {"user": uid, "messages": [{"role": "user", "content": "鉴权测试"}]}
        r = await complete(client, payload)
        check("没带 Key 时 401", r.status_code == 401, str(r.status_code))
        check("401 也是 OpenAI 错误结构",
              r.json().get("error", {}).get("code") == "invalid_api_key", str(r.json())[:90])
        r = await complete(client, payload, headers={"Authorization": "Bearer tok_test_123"})
        check("带对 Key 时 200", r.status_code == 200, str(r.status_code))
        r = await complete(client, payload, headers={"Authorization": "Bearer wrong"})
        check("带错 Key 时 401", r.status_code == 401, str(r.status_code))
        settings.openai_compat_token = ""

    # 清理测试数据
    await long_term.clear_user(uid)
    await long_term.clear_user(uid_stream)
    await short_term.clear_session(f"oai_{uid}")
    await short_term.clear_session(f"oai_{uid_stream}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("\n" + "=" * 68)
    print(f"结果：{passed}/{total} 项通过")
    if passed != total:
        print(f"{FAIL} 失败项：")
        for name, ok, detail in checks:
            if not ok:
                print(f"    - {name} ({detail})")
    else:
        print(f"{OK} OpenAI 兼容入口可用：AstrBot / Cherry Studio / NextChat 都能直接接。")
        print("   Base URL: http://127.0.0.1:8000/v1    API Key: 随便填")
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
