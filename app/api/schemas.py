"""接口的请求/响应数据模型。

为什么不用裸 dict？
FastAPI + Pydantic 会自动做三件事：校验入参、生成 /docs 文档、把响应序列化成 JSON。
前端联调时你会非常庆幸有这层。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """发一条消息给中间件。"""

    user_id: str = Field(..., description="用户标识，记忆按这个维度隔离")
    message: str = Field(..., min_length=1, description="用户这句话")
    session_id: str | None = Field(
        None, description="会话标识。不传就自动新建一个会话"
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="采样温度，不传用默认值")
    history_limit: int | None = Field(
        None, ge=0, le=50, description="带几轮历史给模型，默认读配置（10）"
    )

    # --- 多模态预留字段（当前后端会忽略，未来支持表情包时启用）---
    # 为什么不干脆等服务端实现了再加？
    # 因为接口一旦对外，加字段就要改客户端。现在留着，客户端可以提前接。
    image_url: str | None = Field(
        None, description="【预留】图片地址。当前版本会忽略此字段"
    )
    image_base64: str | None = Field(
        None, description="【预留】图片的 base64 编码。当前版本会忽略此字段"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "u_1001",
                    "message": "你好，我叫小明，是个前端工程师",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """中间件的回复，附带"它背后干了什么"的透明信息。"""

    answer: str = Field(..., description="模型回复")
    session_id: str = Field(..., description="本次会话 ID，下次带上它就能续上上下文")
    trace_id: str = Field(..., description="链路 ID，拿它去日志里查这次请求的全过程")
    model: str = Field("", description="实际使用的模型")
    latency_ms: int = Field(0, description="总耗时（毫秒）")
    tokens: dict = Field(default_factory=dict, description="token 用量")
    history_used: int = Field(0, description="本轮带了几条历史消息给模型")
    profile_used: int = Field(0, description="本轮注入了几条用户档案")
    memories_used: int = Field(0, description="本轮注入了几条长期记忆")
    knowledge_used: int = Field(0, description="本轮注入了多少条知识库资料")
    est_prompt_tokens: int = Field(0, description="估算的 prompt token 数（用于后续做预算优化对比）")
    sections: dict = Field(
        default_factory=dict,
        description="提示词各段的 token 占用，做优化对比时看这个",
    )
    emotion: dict = Field(
        default_factory=dict,
        description="情绪状态：本轮打分 + 更新后的用户状态 + 是否启用了姿态指令",
    )


class EmotionTrajectoryResponse(BaseModel):
    """用户情绪轨迹，可以画成一条曲线。"""

    user_id: str
    current: dict = Field(default_factory=dict, description="当前状态")
    points: list[dict] = Field(default_factory=list, description="历史情绪点")


class MessageItem(BaseModel):
    """一条历史消息。"""

    role: str
    content: str
    created_at: str


class SessionMessagesResponse(BaseModel):
    """某个会话的全部消息，用来人工核对"记忆到底存进去没有"。"""

    session_id: str
    total: int = Field(..., description="该会话总消息数")
    messages: list[MessageItem]


class HistoryItem(BaseModel):
    """带条数的消息（用于会话列表）。"""

    message: str
    history_count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    base_url: str
    has_api_key: bool
    database: str


class MemoryItemOut(BaseModel):
    """一条长期记忆。"""

    id: int
    content: str
    type: str = Field(..., description="fact/preference/goal/event/emotion")
    importance: float = Field(..., description="重要度 0~1")
    valence: float = Field(0.0, description="效价 -1 极负 ~ +1 极正")
    arousal: float = Field(0.0, description="唤醒度 0 平静 ~ 1 激动")
    emotion_label: str | None = None
    created_at: str = ""


class MemoryListResponse(BaseModel):
    """某个用户的记忆与档案全貌。"""

    user_id: str
    total: int = Field(..., description="返回的记忆条数")
    profile: list[dict] = Field(default_factory=list, description="用户档案（确定的结论）")
    memories: list[MemoryItemOut] = Field(default_factory=list, description="长期记忆（素材）")


class ProfileResponse(BaseModel):
    user_id: str
    total: int
    facts: list[dict]
