"""模型层抽象接口。

为什么要有这层抽象？
你的代码只跟 LLMClient 打交道，背后接谁（阿里云 qwen-flash / 智谱 glm-4-flash /
本地 Ollama）完全不影响上层。换服务商 = 改 .env 一行，不改业务代码。

面试价值：被问"你怎么做多模型适配/降级"，这层就是答案。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    """一条对话消息。role 只有三种：system / user / assistant。"""

    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMReply:
    """模型的一次回复。除了文本，还带上用量信息——后面算成本、做优化全靠它。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """模型调用失败。上层捕获它就能走降级逻辑，而不是抛 500 给用户。"""


class LLMClient(ABC):
    """所有模型后端的共同契约。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        """发一轮对话，等完整回复。"""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式发对话，一个字一个字往外吐（实现打字机效果）。"""
        raise NotImplementedError

    async def simple(self, prompt: str, *, system: str | None = None) -> str:
        """快捷方法：只问一句话，只取回复文本。脚本里测试用。"""
        msgs: list[Message] = []
        if system:
            msgs.append(Message("system", system))
        msgs.append(Message("user", prompt))
        reply = await self.chat(msgs)
        return reply.text
