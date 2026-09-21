# 中间件服务的容器镜像
#
# 为什么用多阶段构建？
# 因为 sentence-transformers 只在下拉模型时需要联网，
# 而运行时要能离线工作。分阶段可以让镜像更小、启动更快。
#
# 构建：
#   docker build -t slm-memory-middleware .
# 运行（需要 .env）：
#   docker run --env-file .env -p 8000:8000 -v %cd%/data:/app/data slm-memory-middleware

# ---------- 阶段 1：依赖安装 ----------
FROM python:3.12-slim AS deps

WORKDIR /app

# 系统依赖：
#   build-essential  部分包需要编译
#   curl             健康检查用
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先只拷 requirements —— 这样改了代码不会让依赖层缓存失效
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --timeout 120 --retries 3 -r requirements.txt

# ---------- 阶段 2：运行时 ----------
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 从上一阶段拷依赖
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY docs/kb_sample/ ./docs/kb_sample/

# 数据目录（挂卷覆盖）
RUN mkdir -p /app/data /app/chroma_data /app/logs

# ⚠️ 强制离线：镜像里不该在运行时去联网拉模型
# 模型文件通过挂卷或构建时预下载提供（见 docker-compose.yml 的注释）
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 生产环境不要用 --reload，也不要用单 worker（会阻塞）
# 注意：多个 worker 时，进程内缓存不共享，需要换 Redis（见文档"上生产要改什么"）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
