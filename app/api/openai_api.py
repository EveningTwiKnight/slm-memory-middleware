"""OpenAI 兼容入口：把中间件伪装成一个「OpenAI 服务」，让现成客户端直接接。

为什么做这个？
任何支持"自定义 Base URL"的客户端（AstrBot / Cherry Studio / NextChat /
Open WebUI / LobeChat …）只要把 Base URL 指到这里，就自动获得：
    跨会话长期记忆 + 知识库检索 + 情绪姿态 + token 预算调度
**客户端一行代码都不用改**，也不用装插件。

怎么用（三个格）：
    Base URL :  http://127.0.0.1:8000/v1
    API Key  :  随便填（配了 OPENAI_COMPAT_TOKEN 就填那个值）
    模型名   :  qwen-flash 或 slm-memory-middleware（两个都认，见 /v1/models）

⚠️ 两个必须知道的边界（写在这里，免得后面自己都忘了）：

1. **OpenAI 协议里没有"谁在说话"**。身份只能靠请求里的 `user` 字段或
   `X-User-Id` 头带进来；拿不到就退化成同一个共享身份。
   后果：私聊大概等价于"按人记忆"，群聊就只有"按群记忆"，
   做不到"我记得张三说过…"。
   要真正的按人记忆，得走插件那条路（插件能拿到 sender_id / group_id）。

2. **客户端会把它那侧的完整历史一起发过来**，所以这里以请求里的 messages
   为准，**不再读中间件自己的短期会话**——两份历史都拼进去会重复。
   （中间件自己的会话表仍然落库，只用于调试查看，不参与组装。）

想看这一轮到底注入了什么：`GET /v1/debug/prompt?user_id=..&message=..`
"""

import json
import time
import uuid

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.chat import _extract_and_save
from app.core import tasks
from app.core.config import settings
from app.core.logging import logger, set_trace_id
from app.llm.base import LLMError, Message
from app.llm.openai_compat import get_llm  # 测试里会替换掉这个符号
from app.memory import emotion_store, long_term, short_term
from app.memory.emotion import analyze as analyze_emotion
from app.orchestration import cache as cache_mod
from app.orchestration import prompt_builder

router = APIRouter(prefix="/v1", tags=["OpenAI 兼容"])

# /v1/models 里除了真实模型名，再给一个"中间件"自己的名字，
# 这样客户端下拉框里出现的名字能提示"这一层是中间件"。
MODEL_ALIASES = ("slm-memory-middleware",)


# ------------------------------------------------------------ 小工具


def _to_text(content) -> str:
    """OpenAI 的 content 可能是字符串，也可能是分段数组（多模态客户端会这么发）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                out.append(str(part.get("text") or ""))
            else:
                out.append(str(part))
        return "".join(out)
    return "" if content is None else str(content)


def _split_messages(raw) -> tuple[str, list[Message], str]:
    """把请求里的 messages 拆成 (本轮输入, 历史, 客户端人设)。

    为什么要把 system 单独拿出来？
    因为客户端（比如 AstrBot）会带上它自己的"人格"。如果直接塞进历史，
    就会和中间件内置的人设打架；单独拿出来传给人设参数才是对的。
    """
    msgs = [m for m in (raw or []) if isinstance(m, dict)]

    # 最后一条 user 消息 = 本轮输入
    cut = len(msgs)
    user_input = ""
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            user_input = _to_text(msgs[i].get("content")).strip()
            cut = i
            break

    persona_parts: list[str] = []
    history: list[Message] = []
    for m in msgs[:cut]:
        role = m.get("role")
        text = _to_text(m.get("content")).strip()
        if not text:
            continue
        if role == "system":
            persona_parts.append(text)
        elif role in ("user", "assistant"):
            history.append(Message(role, text))

    return user_input, history, "\n".join(persona_parts)


def _identity(body: dict, x_user_id: str | None, x_session_id: str | None) -> tuple[str, str]:
    """确定 user_id / session_id。

    OpenAI 协议里只有可选的 `user` 字段能带业务身份，所以优先级是：
        user_id    = body.user  →  X-User-Id 头  →  兜底常量
        session_id = X-Session-Id 头  →  由 user_id 派生
    """
    user_id = str(body.get("user") or x_user_id or "").strip() or "openai_client"
    session_id = (x_session_id or "").strip() or f"oai_{user_id}"
    return user_id, session_id


def _error(message: str, status: int = 502, code: str = "upstream_error") -> JSONResponse:
    """按 OpenAI 的错误格式返回 —— 这样客户端才能把人话显示给用户，而不是"未知错误"。"""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "api_error",
                "code": code,
            }
        },
    )


def _check_token(authorization: str | None) -> JSONResponse | None:
    """校验 Bearer。

    没配 OPENAI_COMPAT_TOKEN（默认）时不校验：客户端强制要填 Key，
    接受任何非空 Bearer 即可，本机自用最省事。
    """
    if not settings.openai_compat_token:
        return None
    got = ""
    if authorization and authorization.lower().startswith("bearer "):
        got = authorization[7:].strip()
    if got != settings.openai_compat_token:
        return _error(
            "API Key 不正确：把 .env 里 OPENAI_COMPAT_TOKEN 的值填到客户端的 API Key 里",
            401,
            "invalid_api_key",
        )
    return None


# ------------------------------------------------------------ 接口


@router.get("/models", summary="模型列表（兼容客户端会先拉这个）")
async def list_models() -> dict:
    created = int(time.time())
    ids = list(dict.fromkeys([settings.llm_model, *MODEL_ALIASES]))
    return {
        "object": "list",
        "data": [
            {
                "id": i,
                "object": "model",
                "created": created,
                "owned_by": "slm-memory-middleware",
            }
            for i in ids
        ],
    }


@router.post("/chat/completions", summary="对话补全（OpenAI 兼容，支持流式）")
async def chat_completions(
    body: dict,
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    """把一次 OpenAI 格式的请求，走一遍中间件的完整管线。

    管线：情绪打分 → 组装（人设+档案+记忆+RAG+姿态+预算）→ 调模型 → 落库 → 后台抽取记忆
    """
    denied = _check_token(authorization)
    if denied is not None:
        return denied

    user_input, history, client_persona = _split_messages(body.get("messages"))
    if not user_input:
        return _error("messages 里没有找到 user 消息", 400, "invalid_request")

    user_id, session_id = _identity(body, x_user_id, x_session_id)
    trace_id = uuid.uuid4().hex[:12]
    set_trace_id(trace_id)

    is_stream = bool(body.get("stream"))
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    model_name = str(body.get("model") or settings.llm_model)
    logger.info(
        f"[openai] user={user_id} session={session_id} stream={is_stream} "
        f"历史{len(history)}条 人设{'有' if client_persona else '默认'} len={len(user_input)}"
    )

    # ---- 1. 情绪：和 /v1/chat 完全一致，放在检索之前（检索要用它做情感加权）
    emo_score = analyze_emotion(user_input)
    emo_state = (
        await emotion_store.update_from_score(user_id, emo_score)
        if settings.enable_emotion
        else None
    )

    # ---- 2. 组装提示词：历史用客户端给的，记忆/档案/知识/姿态由中间件补
    bundle = await prompt_builder.assemble(
        user_id=user_id,
        user_input=user_input,
        history=history,
        top_k=settings.memory_top_k,
        emotion_state=emo_state,
        use_emotion=settings.enable_emotion,
        persona=client_persona or None,
    )

    # ---- 3. 缓存：命中就跳过最慢的一步（流式也支持，直接吐一整块）
    rcache = cache_mod.get_cache()
    ckey = cache_mod.make_key(user_id, user_input, bundle.memory_ids)
    cached = rcache.get_exact(ckey)
    cache_type = "exact"
    if cached is None:
        cached = rcache.get_semantic(user_id, user_input)
        cache_type = "semantic"

    async def persist(answer: str) -> None:
        """落库 + 写缓存 + 记记忆使用 + 后台抽取新记忆。"""
        await short_term.append(session_id, user_id, "user", user_input)
        await short_term.append(session_id, user_id, "assistant", answer)
        if not cached:
            rcache.put_exact(ckey, answer)
            rcache.put_semantic(user_id, user_input, answer)
        if bundle.memory_ids:
            await long_term.touch(bundle.memory_ids)
        if settings.enable_extraction:
            tasks.spawn(
                _extract_and_save(user_id, session_id, user_input, answer),
                name="extract_memory",
            )

    # ---- 4a. 流式：转成 OpenAI 的 chunk 协议（客户端只认这个格式）
    if is_stream:
        cid = f"chatcmpl-{trace_id}"

        def chunk(delta: dict, finish: str | None = None) -> str:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        async def event_gen():
            collected: list[str] = []
            try:
                # 第一个 chunk 只带 role，这是 OpenAI 的惯例
                yield chunk({"role": "assistant", "content": ""})
                if cached is not None:
                    collected.append(cached)
                    yield chunk({"content": cached})
                else:
                    async for piece in get_llm().stream(
                        bundle.messages, temperature=temperature, max_tokens=max_tokens
                    ):
                        collected.append(piece)
                        yield chunk({"content": piece})
                answer = "".join(collected)
                await persist(answer)
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                logger.info(
                    f"[openai] 流式完成 user={user_id} chars={len(answer)} "
                    f"memory={bundle.memories_used} cache={cache_type if cached else 'miss'}"
                )
            except LLMError as e:
                logger.error(f"[openai] 流式失败：{e}")
                yield f"data: {json.dumps({'error': {'message': str(e), 'code': 'upstream_error'}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ---- 4b. 非流式
    if cached is not None:
        await persist(cached)
        logger.info(f"[openai] 缓存命中/{cache_type} user={user_id}")
        return _completion(cid=trace_id, model=model_name, text=cached, usage=None)

    try:
        reply = await get_llm().chat(
            bundle.messages, temperature=temperature, max_tokens=max_tokens
        )
    except LLMError as e:
        logger.error(f"[openai] 模型调用失败：{e}")
        return _error(f"模型服务暂时不可用：{e}")

    await persist(reply.text)
    logger.info(
        f"[openai] 完成 user={user_id} memory={bundle.memories_used} "
        f"tokens={reply.prompt_tokens}+{reply.completion_tokens}"
    )
    return _completion(
        cid=trace_id,
        model=reply.model or model_name,
        text=reply.text,
        usage={
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.total_tokens,
        },
        finish=reply.finish_reason,
    )


def _completion(
    *,
    cid: str,
    model: str,
    text: str,
    usage: dict | None,
    finish: str = "stop",
) -> dict:
    """拼一个 OpenAI 格式的回复。"""
    return {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish or "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
