"""配置中心：全项目所有可调参数都从这里读，其他地方不许直接读环境变量。

为什么要单独一个文件？
- 换模型服务商（阿里云 ↔ 智谱 ↔ 本地）只改 .env，不改一行代码；
- 密钥只出现在一个地方，不会散落在代码里被误提交；
- 面试被问"你怎么管理配置"时，这里就是答案。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件在 app/core/ 下，往上两级就是根）
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """从 .env 文件读取配置。字段名大小写不敏感，LLM_BASE_URL 对应 llm_base_url。"""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里有额外的字段不报错
    )

    # --- 模型服务 ---
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen-flash"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # --- 服务 ---
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- 存储 ---
    database_path: str = "data/app.db"
    chroma_path: str = "chroma_data"

    # --- 嵌入模型（Day 5 用）---
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    hf_endpoint: str = "https://hf-mirror.com"

    # --- 上下文控制（Day 6 会用到，先定好默认值）---
    history_limit: int = 10  # 最近保留几轮对话

    # --- 记忆系统 ---
    enable_extraction: bool = True  # 是否在每轮对话后抽取记忆（关掉可省一次模型调用）
    extract_min_importance: float = 0.2  # 低于这个重要度的记忆不写入，过滤垃圾
    memory_top_k: int = 5  # 每轮注入几条记忆（Day 5 换成检索，条数仍受此限制）
    memory_min_importance: float = 0.3  # 注入时的重要度门槛，低于它的记忆不往提示词里放
    # 语义重复判定阈值。依据实测数据定的，不是拍脑袋：
    #   同一件事不同措辞（"担心自己坚持不下来" vs "担心自己无法坚持副业计划"）= 0.8168
    #   完全无关的句子（"是前端工程师" vs "在做牙医手术"）= 0.5401
    # 所以取 0.80：能拦住实测发现的重复对，又离无关句有足够安全距离。
    # 调这个值就是在"漏合并"和"误合并"之间做权衡，实测方法是跑
    # tests/test_retrieval.py 看两端的相似度分布。
    semantic_dedupe_threshold: float = 0.80
    memory_candidate_k: int = 20  # 检索时先召回多少条候选再重新排序
    enable_knowledge: bool = True  # 是否启用知识库检索（RAG）

    # 知识库相关度门槛。实测发现：不加门槛时，无论问什么都把整份文档塞进提示词，
    # 实测某轮 knowledge 段占了 851/1214 token（70%），纯属浪费。
    # 依据实测相似度分布：命中问题的资料 0.58~0.80，无关问题的最高约 0.35。
    knowledge_min_similarity: float = 0.45
    knowledge_top_k: int = 3  # 最多注入几条资料

    # 检索意图判断：问题是否"关于用户本人"。
    # 关掉它就会退化为"每轮都检索记忆"（问天气也会注入 5 条）。
    # 实测：相关与无关问题的相似度区间完全重叠（相关 0.353~0.483，
    # 无关 0.168~0.473），所以用固定门槛切不开，只能先判断意图。
    # 详见 orchestration/intent.py。
    enable_intent_gate: bool = True

    # --- 情感系统 ---
    # 关掉它就能做消融实验：对比"有情感加权"和"没情感加权"的检索差异
    enable_emotion: bool = True

    # --- Token 预算调度 ---
    # 提示词总预算上限。超出就按优先级裁剪（详见 orchestration/budget.py）
    max_prompt_tokens: int = 2400
    enable_budget: bool = True  # 关掉它可对比"有预算调度"和"没有"的长对话表现

    @property
    def db_file(self) -> Path:
        """数据库文件的绝对路径，自动创建父目录。"""
        p = ROOT_DIR / self.database_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_dir(self) -> Path:
        p = ROOT_DIR / self.chroma_path
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """带缓存的配置获取。lru_cache 保证整个进程只读一次 .env。"""
    return Settings()


settings = get_settings()
