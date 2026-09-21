"""OpenAI 兼容协议的模型客户端。

阿里云百炼、智谱、DeepSeek、硅基流动、本地 Ollama、vLLM 全都提供
"OpenAI 兼容"接口 —— 意思是它们的 HTTP 请求格式跟 OpenAI 一模一样。
所以这一个文件就能对接市面上绝大多数模型服务，只需要换 base_url。

内置能力：
- 超时与重试（网络抖动不该让用户看到报错）
- 用量统计（token 数是后面做成本优化的原料）
- 流式与非流式两种模式
"""

import asyncio
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import logger
from app.llm.base import LLMClient, LLMError, LLMReply, Message


class OpenAICompatClient(LLMClient):
    """对接任何 OpenAI 兼容接口。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries

        if not self.api_key:
            raise LLMError(
                "没有读到 API Key。请检查项目根目录下的 .env 文件里 LLM_API_KEY 是否填了值。\n"
                "  提示：可以复制 .env.example 为 .env，里面写了三家服务商的获取方法。"
            )
        if self.api_key.startswith("sk-在这里") or self.api_key in ("your-key", "sk-xxx"):
            raise LLMError(
                "LLM_API_KEY 还是 .env.example 里的占位符，没有换成你自己的 Key。\n"
                "  获取方法见项目 README 的「配置 API Key」一节，或看 .env.example 里的注释。"
            )

        # openai 库自带重试，但只对部分错误生效，所以外面再加一层自己的重试
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    # ---------- 内部工具 ----------

    def _payload(
        self,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
        }

    async def _with_retry(self, coro_factory, what: str):
        """失败重试。coro_factory 是个函数，每次调用生成一个新的协程。"""
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except Exception as e:  # noqa: BLE001 - 这里就是要兜住所有异常
                last_err = e
                if attempt < self.max_retries:
                    wait = 1.5 * (attempt + 1)
                    logger.warning(
                        f"{what} 第 {attempt + 1} 次失败：{type(e).__name__}: {e}；{wait}s 后重试"
                    )
                    await asyncio.sleep(wait)
        raise LLMError(f"{what}失败（已重试 {self.max_retries} 次）：{last_err}") from last_err

    # ---------- 对外接口 ----------

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        payload = self._payload(messages, temperature, max_tokens)

        async def _call():
            return await self._client.chat.completions.create(**payload)

        resp = await self._with_retry(_call, f"调用模型 {self.model}")

        choice = resp.choices[0]
        usage = resp.usage
        reply = LLMReply(
            text=(choice.message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=resp.model or self.model,
            finish_reason=choice.finish_reason or "",
        )
        logger.debug(
            f"模型返回 model={reply.model} "
            f"tokens={reply.prompt_tokens}+{reply.completion_tokens} "
            f"finish={reply.finish_reason}"
        )
        return reply

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, temperature, max_tokens)
        payload["stream"] = True

        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:  # noqa: BLE001
            logger.error(f"流式调用失败：{type(e).__name__}: {e}")
            raise LLMError(f"流式调用失败：{e}") from e


_client: OpenAICompatClient | None = None


def get_llm() -> LLMClient:
    """全局单例。整个进程复用一个客户端，避免反复建连接。"""
    global _client
    if _client is None:
        _client = OpenAICompatClient()
    return _client
