"""接口冒烟测试：确认服务骨架是好的。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m tests.test_api
    .venv\\Scripts\\python.exe -m tests.test_api --offline   # 不发起真实模型调用

它直接在进程内请求接口（不起服务器），检查：
  /health   是否正常
  /v1/chat  路由是否存在（没配 Key 时返回 502 而不是 404）

为什么要有 --offline？
默认情况下配了 Key 就会真的发一条消息给模型（验证端到端通不通）。
CI 里只有一个占位 Key，真发请求会先被服务商拒（401）、再重试两次，
白等约 9 秒还依赖外网。--offline 把 Key 清空后再请求，
这样模型客户端在构造阶段就抛错、直接被兜成 502 —— 不碰网络，
只验证路由与错误处理是好的。
"""

import asyncio
import os
import sys

sys.path.insert(0, ".")

OFFLINE = "--offline" in sys.argv

if OFFLINE:
    # ⚠️ 必须在 import app 之前改环境变量：配置是在 import 时读一次的，
    # 之后再改 settings 就晚了。环境变量优先级高于 .env 文件，
    # 所以这里能盖掉 .env 里的真实 Key（以及 CI 里的占位 Key）。
    os.environ["LLM_API_KEY"] = ""

import httpx  # noqa: E402

from app.main import app  # noqa: E402
from app.memory.database import init_db  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"


async def main() -> int:
    # ⚠️ 必须自己建表：httpx 的 ASGITransport 在进程内直接调 app，
    # 不会触发 FastAPI 的 lifespan，所以 app/main.py 里的 init_db() 不会跑。
    # 本地因为有历史遗留的 data/app.db（表早就建好了）看不出问题，
    # 但 CI 是干净检出，data/ 被 .gitignore 挡掉了，库是空的 ——
    # 于是第一次读表就报 "no such table: user_emotions"。
    print("=" * 60)
    print("准备数据库")
    print("=" * 60)
    await init_db()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("检查 /health")
        print("=" * 60)
        r = await client.get("/health")
        print(f"  状态码：{r.status_code}")
        print(f"  返回  ：{r.text}")
        if r.status_code != 200:
            print(f"{FAIL} /health 不正常")
            return 1
        has_key = r.json().get("has_api_key", False)
        print(f"{OK} 服务骨架正常")

        print("\n" + "=" * 60)
        print("检查 /v1/chat 路由是否注册")
        print("=" * 60)
        r = await client.post(
            "/v1/chat",
            json={"user_id": "u_test", "message": "你好"},
        )
        print(f"  状态码：{r.status_code}")

        if r.status_code == 404:
            print(f"{FAIL} 路由没注册上，检查 app/main.py 里有没有 include_router")
            return 1

        # 显式 offline 时（Key 已被清空），或本机本来就没配 Key：
        # 只验证"路由在、且失败被兜住"
        if OFFLINE or not has_key:
            print(f"  返回  ：{r.text[:160]}")
            if r.status_code == 502:
                print(f"{OK} 路由已注册；未配 Key 时返回 502 且带人话提示（不是 500 堆栈）")
                return 0
            print(f"{FAIL} 未配 Key 时应返回 502，实际 {r.status_code}")
            return 1

        if r.status_code == 200:
            print(f"{OK} 路由已注册且模型调用成功")
            return 0

        print(f"  返回  ：{r.text[:200]}")
        print(f"{FAIL} 意外状态码 {r.status_code}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
