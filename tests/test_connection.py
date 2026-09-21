r"""连通性自检脚本：确认 .env 配对了、能真的调通模型。

用法（在项目根目录）：
    .venv\Scripts\python.exe -m tests.test_connection

它会依次检查：配置是否读到 → 模型是否能回话 → 用量信息是否正常。
任何一步失败都会直接告诉你原因和怎么改。
"""

import asyncio
import sys
import time

# 让脚本能 import app 包（在项目根目录运行时需要）
sys.path.insert(0, ".")

from app.core.config import settings  # noqa: E402
from app.llm.base import LLMError, Message  # noqa: E402
from app.llm.openai_compat import get_llm  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"


async def main() -> int:
    print("=" * 64)
    print("第 1 步：检查配置")
    print("=" * 64)
    print(f"  base_url : {settings.llm_base_url}")
    print(f"  model    : {settings.llm_model}")
    key = settings.llm_api_key
    print(f"  api_key  : {key[:8]}...{key[-4:]} (len={len(key)})" if key else "  api_key  : (空)")

    if not key:
        print(f"\n{FAIL} 没有读到 API Key。")
        print("  解决：把 .env.example 复制成 .env，填入 LLM_API_KEY")
        return 1
    if key.startswith("sk-在这里"):
        print(f"\n{FAIL} .env 里还是模板里的占位值，没有换成你自己的 Key。")
        return 1

    print("\n" + "=" * 64)
    print("第 2 步：调用模型")
    print("=" * 64)

    try:
        llm = get_llm()
    except LLMError as e:
        print(f"{FAIL} 初始化失败：{e}")
        return 1

    messages = [
        Message("system", "你是一个简洁的助手，回答不超过 20 个字。"),
        Message("user", "用一句话说明什么是检索增强生成。"),
    ]

    started = time.perf_counter()
    try:
        reply = await llm.chat(messages)
    except LLMError as e:
        print(f"{FAIL} 调用失败：{e}")
        print("\n常见原因：")
        print("  1. 模型名写错了（阿里云百炼填 qwen-flash，智谱填 glm-4-flash）")
        print("  2. base_url 结尾少了 /v1 或者多写了")
        print("  3. Key 没开通对应模型的权限，或额度用完了")
        print("  4. 公司网络/代理拦截了 HTTPS 请求")
        return 1

    elapsed = int((time.perf_counter() - started) * 1000)

    print(f"{OK} 模型回话了（{elapsed} ms）")
    print(f"\n  回复内容：{reply.text}")
    print(f"  实际模型：{reply.model}")
    print(f"  token   ：prompt={reply.prompt_tokens} completion={reply.completion_tokens}")

    if reply.completion_tokens == 0:
        print("\n  ⚠️ 用量字段是 0。功能不受影响，但 Day 6 做成本统计前要确认一下。")

    print("\n" + "=" * 64)
    print(f"{OK} 全部通过！可以开始 Day 2 了。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
