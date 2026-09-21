"""数据库表结构。

三张表，各司其职：

  conversations  每轮对话原文          —— Day 2 用（"草稿纸"）
  memories       提炼出的长期记忆       —— Day 3 用（"便签墙"）
  profile_facts  用户档案，一事一条     —— Day 3 用（"档案卡"）

为什么现在就把三张表都建好？
因为反复改表结构很烦，而且 `memories` 里的 importance / valence / arousal
三个字段是整个项目的核心（Day 5 的情感加权检索全靠它们排序），
先把位置占好，后面只填数据不动结构。
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有表的基类。"""


class Conversation(Base):
    """一轮对话的一条消息。user 和 assistant 各占一行。"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, comment="会话 ID")
    user_id: Mapped[str] = mapped_column(String(64), index=True, comment="用户 ID")
    role: Mapped[str] = mapped_column(String(16), comment="user 或 assistant")
    content: Mapped[str] = mapped_column(Text, comment="消息正文")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="写入时间"
    )

    # 联合索引：按会话 + 时间查历史是最频繁的操作
    __table_args__ = (Index("ix_conv_session_time", "session_id", "created_at"),)


class Memory(Base):
    """从对话中提炼出的、值得长期记住的一件事。

    这是"记忆"与"聊天记录"的本质区别：
    聊天记录是流水账，记忆是经过判断后留下的结论。
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, comment="属于哪个用户")
    content: Mapped[str] = mapped_column(Text, comment="记忆内容的一句话表述")

    # fact=客观事实  preference=偏好  goal=目标  event=经历  emotion=情绪触发点
    type: Mapped[str] = mapped_column(String(16), default="fact", comment="记忆类型")

    importance: Mapped[float] = mapped_column(
        Float, default=0.5, comment="重要度 0~1，决定召回优先级和衰减速度"
    )

    # --- 情感字段（Day 5 的核心）---
    valence: Mapped[float] = mapped_column(
        Float, default=0.0, comment="效价 -1 极负 ~ +1 极正"
    )
    arousal: Mapped[float] = mapped_column(
        Float, default=0.0, comment="唤醒度 0 平静 ~ 1 激动"
    )
    emotion_label: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="离散情绪标签：焦虑/开心/愤怒..."
    )

    source_session: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="从哪个会话提炼出来的，便于回溯"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_access_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最近一次被召回的时间，用于算记忆强度"
    )

    # --- 多模态预留字段（当前恒为 text，未来支持表情包时启用）---
    # 为什么现在就建好？因为改表结构要写数据迁移，而现在表里还没什么数据，改起来零成本。
    modality: Mapped[str] = mapped_column(
        String(16), default="text", comment="text / image —— 这条记忆来自文字还是图片"
    )
    media_ref: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="关联的媒体资源 ID（见 media_assets 表）"
    )
    ocr_text: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="图片中识别出的文字（表情包的配文往往就是情绪的直接表达）",
    )


class MediaAsset(Base):
    """媒体资源（当前为空表，未来存表情包）。

    ⚠️ 当前未启用。建这张表是为了把"图片存哪、识别结果存哪"想清楚，
    避免未来加功能时又反过来动 memories 表。

    设计要点：
      - 图片本体不进向量库（二进制没法算语义相似度），
        所以检索靠的是 ocr_text + description 这两段文字。
        这是最容易忽略的一点：图片记忆的检索其实建立在"文字化描述"之上。
      - 识别结果与判断依据分开存，便于回溯"当时为什么判成这个情绪"。
      - 用 file_hash 做去重：同一张表情包被发多次，识别结果可以复用，省调用。
    """

    __tablename__ = "media_assets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)

    # 图片本体：存路径而不是二进制，避免数据库膨胀
    file_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True, comment="内容哈希，同一张图可复用识别结果"
    )

    # 识别结果
    is_sticker: Mapped[int] = mapped_column(default=0, comment="1=是表情包")
    has_text: Mapped[int] = mapped_column(default=0)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="画面描述，与 ocr_text 一起用于向量检索"
    )

    # 情绪判断
    valence: Mapped[float] = mapped_column(Float, default=0.0)
    arousal: Mapped[float] = mapped_column(Float, default=0.0)
    emotion_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # 判断依据（面部表情、肢体动作、用的是哪个识别器）
    signals: Mapped[str | None] = mapped_column(Text, nullable=True, comment="JSON")
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ProfileFact(Base):
    """用户档案：姓名、职业、目标、偏好……一件事一条。

    和 memories 的区别：这里是"确定的结论"，直接全量注入提示词；
    memories 是"可能有用的素材"，需要检索后才注入。
    """

    __tablename__ = "profile_facts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(64), comment="档案项名称，如 姓名 / 职业")
    value: Mapped[str] = mapped_column(Text, comment="档案项内容")

    # 新旧事实冲突时：旧记录 is_active 置 0 而不是删除，保留变更历史
    is_active: Mapped[int] = mapped_column(default=1, comment="1=当前有效 0=已被新事实覆盖")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("ix_fact_user_key", "user_id", "key"),)


class UserEmotion(Base):
    """用户当前的情绪状态（一人一条）。

    为什么要把它落库，而不是每次从对话历史重新算？
      1. 情绪状态是**跨会话**的：昨天聊得很丧，今天再聊时系统该记得；
      2. 重算要读全部历史，浪费且慢；
      3. 状态需要"随时间自然回落"（见 emotion.py 的衰退逻辑），
         这要求它有一个持久化的时间戳。

    字段分两组：即时状态（current_*）和长期倾向（stable_*）。
    即时状态回答"他现在什么心情"，长期倾向回答"他这个人整体偏什么状态"。
    """

    __tablename__ = "user_emotions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    current_valence: Mapped[float] = mapped_column(Float, default=0.0, comment="即时效价")
    current_arousal: Mapped[float] = mapped_column(Float, default=0.0, comment="即时唤醒度")
    stable_valence: Mapped[float] = mapped_column(Float, default=0.0, comment="长期倾向效价")
    stable_arousal: Mapped[float] = mapped_column(Float, default=0.0, comment="长期倾向唤醒")

    turns: Mapped[int] = mapped_column(default=0, comment="已经过多少轮更新")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
