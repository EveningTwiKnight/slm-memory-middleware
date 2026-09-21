"""嵌入模型：把文本变成向量，让"意思相近"可以被计算。

一句话理解：向量就是给每句话算一个"坐标"。
意思相近的句子，坐标就挨得近；我们可以用余弦相似度算它们有多近。
这是所有检索功能的地基。

两个容易踩的坑，这里都处理了：
  1. **联网卡死**：sentence-transformers 每次加载都会先联网检查更新。
     国内连 huggingface.co 不通，会卡好几分钟（实测 531 秒）。
     解法：提前设置 HF_HUB_OFFLINE，让它直接用本地缓存。
  2. **查询要加指令前缀**：BGE 系列模型训练时，查询侧带了固定指令前缀
     "为这个句子生成表示以用于检索相关文章："。
     文档侧不加。如果不加，检索效果会明显下降 —— 这是个很容易被忽略的细节。
"""

import os
from functools import lru_cache

from app.core.config import settings
from app.core.logging import logger

# ⚠️ 必须在 import sentence_transformers 之前设置。
#
# 用直接赋值而不是 setdefault，是因为 setdefault 在环境里已有该变量时不会覆盖，
# 而我们要的是"强制离线"。
#
# 踩坑记录：早期只设了 HF_HUB_OFFLINE，但 sentence_transformers 加载时
# 仍会去 HEAD 请求 huggingface.co 检查模型是否有更新（adapter_config.json、
# processor_config.json 等）。国内连不上，于是每次失败后重试 5 次、指数退避，
# 实测一次加载耗时约 3 分钟，日志里全是 WinError 10054。
#
# 现在三个开关一起设：HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 强制走本地缓存，
# HF_ENDPOINT 指向镜像（万一需要联网也有个快的落点）。
os.environ["HF_ENDPOINT"] = settings.hf_endpoint
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 关掉远程模型卡与遥测请求，避免额外的网络往返
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# BGE 中文模型的查询指令前缀（文档侧不加）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def _get_model():
    """加载嵌入模型。全局只加载一次（首次约 1~2 秒，之后走缓存）。"""
    from sentence_transformers import SentenceTransformer

    logger.info(f"加载嵌入模型：{settings.embed_model}")
    try:
        model = SentenceTransformer(settings.embed_model)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"加载嵌入模型失败：{e}\n"
            f"如果模型还没下载过，请先执行（联网一次即可）：\n"
            f'  $env:HF_ENDPOINT="{settings.hf_endpoint}"\n'
            f'  .venv\\Scripts\\python.exe -c "from sentence_transformers import SentenceTransformer; '
            f"SentenceTransformer('{settings.embed_model}')\""
        ) from e
    logger.info("嵌入模型加载完成")
    return model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """把文档/记忆编码成向量。文档侧不加指令前缀。"""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """把用户的查询编码成向量。查询侧要加指令前缀。"""
    model = _get_model()
    vector = model.encode(
        [QUERY_INSTRUCTION + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vector.tolist()


def similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """余弦相似度。因为已经归一化，点积就等于余弦相似度。

    取值范围 -1 ~ 1。中文短句之间的典型值：
      完全无关  0.1 ~ 0.3
      同一话题  0.4 ~ 0.6
      意思基本相同  0.75 以上
    """
    return float(sum(a * b for a, b in zip(vec_a, vec_b, strict=False)))
