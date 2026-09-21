"""中间件服务入口。

启动：uvicorn app.main:app --reload
文档：http://127.0.0.1:8000/docs
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat
from app.api.schemas import HealthResponse
from app.core.config import settings

# ⚠️ 必须在任何 transformers / sentence_transformers 导入之前设置，
# 否则首次下载嵌入模型时会直连 huggingface.co（国内不通）而卡死。
os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

from app.core.logging import logger, setup_logging  # noqa: E402
from app.memory.database import close_db, init_db  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭时执行。数据库在这里建表。"""
    setup_logging()
    logger.info("=" * 60)
    logger.info("中间件启动中……")
    logger.info(f"模型服务：{settings.llm_base_url}")
    logger.info(f"使用模型：{settings.llm_model}")
    if not settings.llm_api_key:
        logger.warning("⚠️  没有读到 LLM_API_KEY，请在项目根目录建一个 .env 文件（可复制 .env.example）")
    await init_db()

    # 预热嵌入模型：首次加载约 1.5 秒（要从磁盘读 torch 和模型权重）。
    # 不做这一步的话，第一个用户的请求会白等这 1.5 秒。
    # 放在后台线程里执行，避免拖慢服务启动（启动期间健康检查可能被调用）。
    async def _warmup() -> None:
        try:
            from app.memory import embeddings

            await asyncio.to_thread(embeddings.embed_query, "预热")
            logger.info("嵌入模型预热完成")
        except Exception as e:  # noqa: BLE001
            # 预热失败不该影响服务启动 —— 真正的请求会用降级方案
            logger.warning(f"嵌入模型预热失败（不影响启动，检索会走降级）：{e}")

    asyncio.create_task(_warmup())

    logger.info("=" * 60)
    yield
    await close_db()
    logger.info("中间件已关闭")

app = FastAPI(
    title="小模型增强的 RAG + 情感记忆中间件",
    description="为小模型补上长期记忆、外部知识与情感连续性",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许跨域，方便 Day 7 的 Gradio 页面直接调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health", response_model=HealthResponse, tags=["系统"], summary="健康检查")
async def health() -> HealthResponse:
    """用来确认服务活着、配置读对了。排查问题的第一站。"""
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        has_api_key=bool(settings.llm_api_key),
        database=str(settings.db_file),
    )
