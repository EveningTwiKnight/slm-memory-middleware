"""一个"假模型"，用来在**没有 API Key** 的情况下验证管道是否通畅。

为什么需要它？
依赖真实模型做测试有两个问题：
  1. 没 Key 就完全没法测；
  2. 模型每次回答都不一样，没法做自动化断言。

解法：写一个"照镜子"的假模型 —— 你给它什么它就说什么。
于是"历史到底有没有被正确拼进来"这件事可以直接断言，而不用猜。

它的回答长这样：
    [回声] 我看到 5 条历史消息：user=我叫小明 | assistant=你好小明 | ...
       最新一句：我叫什么

一眼就能看出历史有没有进来、进来了几条。

这个类在正式代码里不会被用到，只在测试和本地无 Key 调试时使用。
"""

import asyncio
from typing import AsyncIterator

from app.llm.base import LLMClient, LLMReply, Message


class EchoLLMClient(LLMClient):
    """把收到的消息原样描述回去，用于验证提示词拼装是否正确。"""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.last_messages: list[Message] = []  # 测试里可以直接检查这个

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        self.last_messages = list(messages)
        if self.delay:
            await asyncio.sleep(self.delay)

        system = [m for m in messages if m.role == "system"]
        history = [m for m in messages if m.role != "system"]
        last = history[-1].content if history else ""
        earlier = history[:-1]

        parts = [f"[回声] 我看到 {len(earlier)} 条历史消息"]
        if earlier:
            preview = " | ".join(f"{m.role}={m.content[:20]}" for m in earlier[-6:])
            parts.append(f"（最近几条：{preview}）")
        if system:
            parts.append(f"人设长度={len(system[0].content)}")
        parts.append(f"最新一句：{last}")

        text = " ".join(parts)
        # 粗略估算 token：中文约 1 字 1 token
        return LLMReply(
            text=text,
            prompt_tokens=sum(len(m.content) for m in messages),
            completion_tokens=len(text),
            model="echo-stub",
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        for ch in reply.text:
            yield ch
            await asyncio.sleep(0.005)
