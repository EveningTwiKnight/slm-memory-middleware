"""对话接口。

一次请求的完整流程（面试就按这个讲）：
  1. 打 trace_id
  2. 确定会话
  3. **情绪分析**：给用户这句话打分，更新他的情绪状态
  4. 读短期记忆（最近 N 轮原文）
  5. 读长期记忆（档案 + 语义检索到的记忆）+ 情绪姿态 → 组装提示词
  6. 调模型 → 返回给用户                     ← 用户感知的延迟到此结束
  7. 落库 + 后台异步抽取新记忆               ← 非关键路径，用户不等
  8. 响应里带上"这轮干了什么"的透明信息

演进过程：
  Day 3：把"流水账"升级成"结论"（抽取）
  Day 4：把"结论"喂给模型（注入）
  Day 5：语义检索 + 知识库 RAG
  Day 5后半：情绪分析 + 情感加权召回 + 情绪驱动的回复姿态
"""

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    EmotionTrajectoryResponse,
    MemoryItemOut,
    MemoryListResponse,
    MessageItem,
    ProfileResponse,
    SessionMessagesResponse,
)
from app.core import tasks
from app.core.config import settings
from app.core.logging import logger, set_trace_id
from app.llm.base import LLMError, Message
from app.llm.openai_compat import get_llm
from app.memory import emotion_store, long_term, short_term
from app.memory.database import SessionLocal
from app.memory.emotion import analyze as analyze_emotion
from app.memory.emotion import log_summary as emotion_log
from app.memory.extractor import MemoryExtractor
from app.memory.models import Conversation
from app.orchestration import cache as cache_mod
from app.orchestration import prompt_builder

router = APIRouter(prefix="/v1", tags=["对话"])


async def _extract_and_save(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """后台任务：从这轮对话里抽取长期记忆并入库。

    这个函数在后台跑，它出任何错都不该影响用户（tasks.spawn 会兜住异常）。
    """
    try:
        extractor = MemoryExtractor(get_llm())
        items = await extractor.extract(
            user_msg,
            assistant_msg,
            min_importance=settings.extract_min_importance,
        )
        if not items:
            logger.debug("本轮没有值得记住的内容")
            return
        await long_term.save_memories(user_id, items, source_session=session_id)
    except Exception as e:  # noqa: BLE001 - 后台任务要兜住一切
        logger.error(f"记忆抽取任务失败：{type(e).__name__}: {e}")


@router.post("/chat", response_model=ChatResponse, summary="发一条消息，拿回复")
async def chat(req: ChatRequest) -> ChatResponse:
    # 1. 打上 trace_id
    trace_id = uuid.uuid4().hex[:12]
    set_trace_id(trace_id)

    # 2. 确定会话
    session_id = req.session_id or f"s_{uuid.uuid4().hex[:12]}"
    logger.info(f"收到消息 user={req.user_id} session={session_id} len={len(req.message)}")
    started = time.perf_counter()

    # 3. 情绪分析：给用户这句话打分，并更新他的情绪状态
    #    为什么放在检索之前？因为检索要用到情绪状态做情感加权。
    emo_score = analyze_emotion(req.message)
    if settings.enable_emotion:
        emo_state = await emotion_store.update_from_score(req.user_id, emo_score)
        logger.debug(f"情绪分析：{emotion_log(emo_score)}")
    else:
        emo_state = None
    emotion_before = emo_score

    # 4. 读短期记忆（最近 N 轮原文）—— 保证对话连贯性
    history = await short_term.get_recent(session_id, req.history_limit)

    # 5. 读长期记忆 + 组装提示词
    bundle = await prompt_builder.assemble(
        user_id=req.user_id,
        user_input=req.message,
        history=history,
        top_k=settings.memory_top_k,
        emotion_state=emo_state,
        use_emotion=settings.enable_emotion,
    )
    logger.debug(
        f"提示词组装完成：历史{bundle.history_used}条 "
        f"档案{bundle.profile_used}条 记忆{bundle.memories_used}条 "
        f"资料{bundle.knowledge_used}条 姿态={'有' if getattr(bundle, 'emotion_used', False) else '无'} "
        f"估算 {bundle.est_prompt_tokens} tokens | 分段 {bundle.sections}"
    )

    # 6. 查缓存：命中的话直接返回，跳过最慢最贵的模型调用
    #    缓存键包含"注入的记忆 id"，避免记忆状态变了还返回旧答案
    rcache = cache_mod.get_cache()
    ckey = cache_mod.make_key(req.user_id, req.message, bundle.memory_ids)
    cached = rcache.get_exact(ckey)
    cache_type = "exact"
    if cached is None:
        cached = rcache.get_semantic(req.user_id, req.message)
        cache_type = "semantic"

    if cached is not None:
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(f"[缓存命中/{cache_type}] {elapsed}ms，跳过模型调用")
        # 缓存命中也要落库，保证对话历史完整
        await short_term.append(session_id, req.user_id, "user", req.message)
        await short_term.append(session_id, req.user_id, "assistant", cached)
        return ChatResponse(
            answer=cached,
            session_id=session_id,
            trace_id=trace_id,
            model="cache",
            latency_ms=elapsed,
            tokens={"prompt": 0, "completion": 0, "total": 0},
            history_used=bundle.history_used,
            profile_used=bundle.profile_used,
            memories_used=bundle.memories_used,
            knowledge_used=bundle.knowledge_used,
            est_prompt_tokens=bundle.est_prompt_tokens,
            sections={**bundle.sections, "cache_hit": cache_type},
            emotion={"label": emo_state.label if emo_state else "中性", "cache": cache_type},
        )

    rcache.note_miss()

    try:
        reply = await get_llm().chat(bundle.messages, temperature=req.temperature)
    except LLMError as e:
        logger.error(f"模型调用失败：{e}")
        raise HTTPException(status_code=502, detail=f"模型服务暂时不可用：{e}") from e

    # 7. 落库 + 写缓存 + 更新记忆使用时间 + 后台抽取新记忆
    await short_term.append(session_id, req.user_id, "user", req.message)
    await short_term.append(session_id, req.user_id, "assistant", reply.text)

    rcache.put_exact(ckey, reply.text)
    rcache.put_semantic(req.user_id, req.message, reply.text)

    # 记录这些记忆被用过。Day 5 算"记忆强度"时会用：
    # 经常被召回的记得更牢（衰减更慢），类似艾宾浩斯复习
    if bundle.memory_ids:
        await long_term.touch(bundle.memory_ids)

    if settings.enable_extraction:
        tasks.spawn(
            _extract_and_save(req.user_id, session_id, req.message, reply.text),
            name="extract_memory",
        )

    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info(
        f"回复完成 {elapsed}ms history={len(history)} memory={bundle.memories_used} "
        f"tokens={reply.prompt_tokens}+{reply.completion_tokens} finish={reply.finish_reason}"
    )

    return ChatResponse(
        answer=reply.text,
        session_id=session_id,
        trace_id=trace_id,
        model=reply.model,
        latency_ms=elapsed,
        tokens={
            "prompt": reply.prompt_tokens,
            "completion": reply.completion_tokens,
            "total": reply.total_tokens,
        },
        history_used=bundle.history_used,
        profile_used=bundle.profile_used,
        memories_used=bundle.memories_used,
        knowledge_used=bundle.knowledge_used,
        est_prompt_tokens=bundle.est_prompt_tokens,
        sections=bundle.sections,
        emotion={
            "label": emo_state.label if emo_state else "中性",
            "valence": emo_state.valence if emo_state else 0.0,
            "arousal": emo_state.arousal if emo_state else 0.0,
            "trend": emo_state.trend if emo_state else "平稳",
            "this_turn": {
                "valence": emotion_before.valence,
                "arousal": emotion_before.arousal,
                "label": emotion_before.label,
                "confidence": emotion_before.confidence,
                "matched": emotion_before.matched[:5],
            },
            "strategy_used": getattr(bundle, "emotion_used", False),
        },
    )


# ------------------------------------------------------------ 记忆相关接口


@router.get(
    "/memory/{user_id}",
    response_model=MemoryListResponse,
    summary="查看某个用户被记住了什么",
)
async def get_memory(user_id: str, limit: int = 50, min_importance: float = 0.0) -> MemoryListResponse:
    """这是 Day 3 之后最常用的接口 —— 抽取器干得好不好，看它就知道。"""
    memories = await long_term.get_memories(user_id, limit=limit, min_importance=min_importance)
    profile = await long_term.get_profile(user_id)
    return MemoryListResponse(
        user_id=user_id,
        total=len(memories),
        profile=profile,
        memories=[
            MemoryItemOut(
                id=m.id,
                content=m.content,
                type=m.type,
                importance=m.importance,
                valence=m.valence,
                arousal=m.arousal,
                emotion_label=m.emotion_label,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in memories
        ],
    )


@router.get("/profile/{user_id}", response_model=ProfileResponse, summary="查看用户档案")
async def get_profile(user_id: str) -> ProfileResponse:
    facts = await long_term.get_profile(user_id)
    return ProfileResponse(user_id=user_id, total=len(facts), facts=facts)


@router.delete("/memory/{user_id}", summary="删除某个用户的全部记忆（合规必需）")
async def delete_memory(user_id: str) -> dict:
    result = await long_term.clear_user(user_id)
    return {"user_id": user_id, **result}


@router.get(
    "/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
    summary="查看某个会话的历史消息",
)
async def session_messages(session_id: str) -> SessionMessagesResponse:
    async with SessionLocal() as db:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.id.asc())
        )
        rows = list(result.scalars().all())

    return SessionMessagesResponse(
        session_id=session_id,
        total=len(rows),
        messages=[
            MessageItem(
                role=r.role,
                content=r.content,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
            for r in rows
        ],
    )


@router.delete("/sessions/{session_id}", summary="清空一个会话的历史")
async def clear_session(session_id: str) -> dict:
    n = await short_term.clear_session(session_id)
    return {"session_id": session_id, "deleted": n}


@router.get("/users/{user_id}/sessions", summary="列出某个用户的所有会话")
async def user_sessions(user_id: str) -> dict:
    return {"user_id": user_id, "sessions": await short_term.list_sessions(user_id)}


@router.get("/tasks/pending", summary="查看还有几个后台任务在跑（调试用）")
async def pending_tasks() -> dict:
    return {"pending": tasks.pending_count()}


@router.post("/chat/stream", summary="流式对话（SSE，打字机效果）")
async def chat_stream(req: ChatRequest):
    """流式输出。

    为什么流式这么重要？
    实测非流式的一次请求要等 400ms~13000ms 才看到第一个字。
    流式之后，首字延迟降到几百毫秒 ——
    用户感知的"快"取决于第一个字什么时候出现，而不是整段话说完。

    实现要点：边吐字边攒完整回答，流结束后再落库和抽取记忆。
    注意抽取记忆仍然是**后台任务**，用户不等它。
    """
    trace_id = uuid.uuid4().hex[:12]
    set_trace_id(trace_id)
    session_id = req.session_id or f"s_{uuid.uuid4().hex[:12]}"
    logger.info(f"[stream] 收到消息 user={req.user_id} session={session_id}")

    # 情绪分析 + 预算裁剪 + 组装（和普通接口一致）
    emo_score = analyze_emotion(req.message)
    emo_state = (
        await emotion_store.update_from_score(req.user_id, emo_score)
        if settings.enable_emotion
        else None
    )
    history = await short_term.get_recent(session_id, req.history_limit)
    bundle = await prompt_builder.assemble(
        user_id=req.user_id,
        user_input=req.message,
        history=history,
        top_k=settings.memory_top_k,
        emotion_state=emo_state,
        use_emotion=settings.enable_emotion,
    )

    async def event_gen():
        collected: list[str] = []
        started = time.perf_counter()
        first_token_ms = 0
        try:
            # 先发一条 meta，让前端知道这次请求的上下文情况
            yield f"data: {json.dumps({'type': 'meta', 'trace_id': trace_id, 'session_id': session_id, 'memories_used': bundle.memories_used, 'memories': bundle.memories_used, 'knowledge_used': bundle.knowledge_used, 'est_prompt_tokens': bundle.est_prompt_tokens}, ensure_ascii=False)}\n\n"

            async for chunk in get_llm().stream(bundle.messages, temperature=req.temperature):
                if not first_token_ms:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                    logger.debug(f"[stream] 首字延迟 {first_token_ms}ms")
                collected.append(chunk)
                yield f"data: {json.dumps({'type': 'delta', 'text': chunk}, ensure_ascii=False)}\n\n"

            answer = "".join(collected)
            # 流结束：落库 + 后台抽取记忆
            await short_term.append(session_id, req.user_id, "user", req.message)
            await short_term.append(session_id, req.user_id, "assistant", answer)
            if bundle.memory_ids:
                await long_term.touch(bundle.memory_ids)
            if settings.enable_extraction:
                tasks.spawn(
                    _extract_and_save(req.user_id, session_id, req.message, answer),
                    name="extract_memory",
                )

            total_ms = int((time.perf_counter() - started) * 1000)
            yield f"data: {json.dumps({'type': 'done', 'latency_ms': total_ms, 'first_token_ms': first_token_ms, 'chars': len(answer)}, ensure_ascii=False)}\n\n"
            logger.info(
                f"[stream] 完成 首字{first_token_ms}ms/总{total_ms}ms "
                f"chars={len(answer)} memory={bundle.memories_used}"
            )
        except LLMError as e:
            logger.error(f"[stream] 失败：{e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/cache/stats", summary="查看缓存命中情况")
async def cache_stats() -> dict:
    return cache_mod.get_cache().info()


@router.delete("/cache", summary="清空缓存")
async def cache_clear() -> dict:
    cache_mod.get_cache().clear()
    return {"cleared": True}


@router.get(
    "/emotion/{user_id}",
    response_model=EmotionTrajectoryResponse,
    summary="查看用户情绪状态与轨迹",
)
async def get_emotion(user_id: str, limit: int = 20) -> EmotionTrajectoryResponse:
    """情绪轨迹是演示时最直观的一块 —— 能画成一条曲线，让人一眼看到状态变化。"""
    state = await emotion_store.get_state(user_id)
    points = await emotion_store.get_trajectory(user_id, limit=limit)
    return EmotionTrajectoryResponse(
        user_id=user_id,
        current={
            "label": state.label,
            "valence": state.valence,
            "arousal": state.arousal,
            "stable_valence": state.stable_valence,
            "stable_arousal": state.stable_arousal,
            "trend": state.trend,
            "recent_low": state.recent_low,
            "turns": state.turns,
        },
        points=points,
    )


@router.get("/debug/prompt", summary="预览将要发给模型的完整提示词（调试用）")
async def debug_prompt(user_id: str, message: str, session_id: str | None = None) -> dict:
    """看一眼"模型到底看到了什么"。

    这是最有用的调试接口：记忆注入没生效、模型答非所问时，
    先看这个，立刻就知道是"没捞到记忆"还是"模型没用上"。
    """
    history = await short_term.get_recent(session_id, settings.history_limit) if session_id else []
    state = await emotion_store.get_state(user_id)
    bundle = await prompt_builder.assemble(
        user_id=user_id,
        user_input=message,
        history=history,
        top_k=settings.memory_top_k,
        emotion_state=state,
        use_emotion=settings.enable_emotion,
    )
    return {
        "user_id": user_id,
        "message": message,
        "est_prompt_tokens": bundle.est_prompt_tokens,
        "sections": bundle.sections,
        "stats": {
            "history_used": bundle.history_used,
            "profile_used": bundle.profile_used,
            "memories_used": bundle.memories_used,
            "knowledge_used": bundle.knowledge_used,
            "emotion_strategy": getattr(bundle, "emotion_used", False),
        },
        "emotion_state": {
            "label": state.label,
            "valence": state.valence,
            "arousal": state.arousal,
            "trend": state.trend,
        },
        "messages": [{"role": m.role, "content": m.content} for m in bundle.messages],
    }
