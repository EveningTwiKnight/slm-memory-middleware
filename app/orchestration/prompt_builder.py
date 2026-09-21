"""提示词组装：把"人设 + 用户档案 + 相关记忆 + 最近对话 + 当前输入"拼成模型能读的一段话。

这是整个中间件的**心脏**。前面几天做的所有事（存历史、抽记忆）都是准备工作，
真正让它们产生价值的是这一步 —— 把对的信息在对的时候放到模型眼前。

为什么要有独立的一层，而不是在接口里拼字符串？
  1. Day 5 要把"取重要度最高的记忆"换成"检索最相关的记忆"，只改这里；
  2. Day 6 要加 token 预算裁剪，也只改这里；
  3. Day 5 的情感指引注入，还是只改这里。

一个关键细节：**记忆必须带时间标记**。
不带的话，模型会把三个月前的事当成刚发生的，说出"你昨天说的那个副业"，
用户一眼就看出它记错了时间 —— 这是最容易翻车、也最容易修的地方。
"""

from dataclasses import dataclass, field
from datetime import datetime

from app.core.config import settings
from app.core.logging import logger
from app.llm.base import Message
from app.memory.models import Memory

# 各段的默认 token 配额（中文约 1 字 1 token）。
# Day 6 的预算调度器会按这个表做超限裁剪，现在先把结构定好。
BUDGET = {
    "persona": 200,
    "profile": 150,
    "memories": 300,
    "history": 700,
    "input": 200,
}

DEFAULT_PERSONA = (
    "你是一个耐心、务实的助手。回答简洁，不说套话，不编造事实。\n"
    "如果用户之前说过相关信息，自然地用起来，不要生硬地复述。"
)

TYPE_LABEL = {
    "fact": "事实",
    "preference": "偏好",
    "goal": "目标",
    "event": "经历",
    "emotion": "情绪",
}


def est_tokens(text: str) -> int:
    """粗略估算 token 数。中文约 1 字 1 token，比 tiktoken 省一个依赖。

    面试可讲：为什么不精确算？因为预算调度的目的是"防止 prompt 爆炸"，
    需要的是量级判断而不是精确计费，多引入一个 tokenizer 依赖不值得。
    """
    return len(text)


def relative_time(dt: datetime | None, now: datetime | None = None) -> str:
    """把时间戳翻译成"3 天前"这种人话。

    为什么不让模型自己算？
    模型对时间计算很不靠谱，而且它根本不知道"现在"是什么时候。
    我们在中间件里算好，直接给它结论。
    """
    if dt is None:
        return "时间未知"
    now = now or datetime.now()
    delta = now - dt
    seconds = delta.total_seconds()

    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    days = int(seconds // 86400)
    if days == 1:
        return "昨天"
    if days < 30:
        return f"{days} 天前"
    months = days // 30
    if months < 12:
        return f"{months} 个月前"
    return f"{days // 365} 年前"


@dataclass
class PromptBundle:
    """组装结果。除了消息本身，还带上"用了什么"的透明信息。"""

    messages: list[Message] = field(default_factory=list)
    memory_ids: list[int] = field(default_factory=list)
    profile_used: int = 0
    memories_used: int = 0
    knowledge_used: int = 0
    history_used: int = 0
    est_prompt_tokens: int = 0
    sections: dict = field(default_factory=dict)


def format_profile(facts: list[dict]) -> str:
    """把用户档案格式化成紧凑的列表。"""
    if not facts:
        return ""
    lines = [f"- {f['key']}：{f['value']}" for f in facts]
    return "\n".join(lines)


def format_memories(memories: list[Memory], now: datetime | None = None) -> str:
    """把长期记忆格式化成带时间标记的列表。"""
    if not memories:
        return ""
    now = now or datetime.now()
    lines = []
    for i, m in enumerate(memories, 1):
        label = TYPE_LABEL.get(m.type, m.type)
        when = relative_time(m.created_at, now)
        lines.append(f"{i}. [{label}·{when}] {m.content}")
    return "\n".join(lines)


def format_knowledge(items: list[dict]) -> str:
    """把知识库检索结果格式化成带来源标记的列表。"""
    if not items:
        return ""
    lines = []
    for i, k in enumerate(items, 1):
        src = k.get("source") or "未知来源"
        lines.append(f"[资料{i}·{src}] {k.get('content', '')}")
    return "\n".join(lines)


def build_messages(
    *,
    user_input: str,
    history: list[Message],
    profile: list[dict] | None = None,
    memories: list[Memory] | None = None,
    knowledge: list[dict] | None = None,
    persona: str | None = None,
    budget: dict | None = None,
    emotion_block: str = "",
    apply_budget: bool = True,
) -> PromptBundle:
    """组装一次请求要发给模型的完整消息列表。

    结构（顺序很重要，模型对开头的注意力最强）：
        [system]  人设 → 用户档案 → 相关记忆 → 参考资料(RAG) → 情绪姿态
        [历史...] 最近 N 轮原文
        [user]    当前输入

    为什么要给人设、档案、记忆都放进 system，而不是拆成多条消息？
    因为 system 是"背景设定"，历史是"对话过程"，两者混在一起会让模型
    分不清哪句是设定、哪句是真人说过的话，容易把记忆当成对话内容照抄。

    情绪姿态段放在 system 的最末尾，这是有意的：
    它是"这一轮该怎么做"的即时指令，紧挨着用户输入，模型最容易遵守。

    apply_budget=True 时先按 token 预算裁剪各段（见 orchestration/budget.py）。
    裁剪发生在**拼装之前**，而不是拼完再截字符串 ——
    截字符串会把一个句子切成两半，模型读到残句反而更容易出错。
    """
    profile = profile or []
    memories = memories or []
    knowledge = knowledge or []
    persona = persona or DEFAULT_PERSONA
    budget_plan = None

    # ---- 预算裁剪：先决定每段留多少空间 ----
    if apply_budget and settings.enable_budget:
        from app.orchestration import budget as budget_mod

        memory_lines = [(m.content, budget_mod.est(m.content), m.id or 0) for m in memories]
        knowledge_lines = [k.get("content", "") for k in knowledge]

        kept, budget_plan = budget_mod.plan_and_trim(
            persona=persona,
            profile_text=format_profile(profile),
            memory_lines=memory_lines,
            knowledge_lines=knowledge_lines,
            history=history,
            user_input=user_input,
            total_budget=settings.max_prompt_tokens,
            budget=budget,
        )

        # 把裁剪结果映射回对象列表（按 id / 内容匹配）
        keep_mem_ids = {mid for _, _, mid in kept["memory_lines"]}
        memories = [m for m in memories if (m.id or 0) in keep_mem_ids]
        keep_k_texts = list(kept["knowledge_lines"])
        knowledge = [k for k in knowledge if k.get("content", "") in keep_k_texts]
        history = kept["history"]
        user_input = kept["user_input"]
        profile_text = kept["profile_text"]
    else:
        profile_text = format_profile(profile)

    # --- 组装 system 段 ---
    sections: dict = {}
    parts = [persona.strip()]
    sections["persona"] = est_tokens(persona)

    if profile_text:
        block = f"## 关于这位用户（已确认的档案）\n{profile_text}"
        parts.append(block)
        sections["profile"] = est_tokens(block)

    memory_text = format_memories(memories)
    if memory_text:
        block = (
            "## 你可能记得的相关信息\n"
            "（这些是过去对话中留下的记忆，括号里是距离现在的时间。"
            "自然地使用它们，不要生硬复述，也不要编造没提到的内容）\n"
            f"{memory_text}"
        )
        parts.append(block)
        sections["memories"] = est_tokens(block)

    knowledge_text = format_knowledge(knowledge)
    if knowledge_text:
        block = (
            "## 参考资料\n"
            "（以下是从知识库检索到的内容。回答时以这些资料为准；"
            "如果资料里没有答案，就直说不知道，不要编造）\n"
            f"{knowledge_text}"
        )
        parts.append(block)
        sections["knowledge"] = est_tokens(block)

    if emotion_block:
        parts.append(emotion_block)
        sections["emotion"] = est_tokens(emotion_block)

    system_content = "\n\n".join(parts)

    # --- 组装完整消息列表 ---
    messages: list[Message] = [Message("system", system_content)]
    messages.extend(history)
    messages.append(Message("user", user_input))

    total = sum(est_tokens(m.content) for m in messages)
    sections["history"] = sum(est_tokens(h.content) for h in history)
    sections["input"] = est_tokens(user_input)

    bundle = PromptBundle(
        messages=messages,
        memory_ids=[m.id for m in memories],
        profile_used=len(profile),
        memories_used=len(memories),
        knowledge_used=len(knowledge),
        history_used=len(history),
        est_prompt_tokens=total,
        sections=sections,
    )
    bundle.budget_plan = budget_plan  # type: ignore[attr-defined]
    return bundle


async def assemble(
    *,
    user_id: str,
    user_input: str,
    history: list[Message],
    top_k: int | None = None,
    session_id: str | None = None,
    emotion_state=None,
    use_emotion: bool = True,
    persona: str | None = None,
) -> PromptBundle:
    """取档案 + 语义检索记忆 + 情绪状态，然后组装提示词。

    演进过程（面试可以按这个讲）：
      Day 4：取重要度最高的 N 条        → 问"副业怎么开始"，端出"你叫小明"
      Day 5：语义检索 + 时间 + 重要度   → 端出跟问题相关的
      Day 5后半：再加情绪一致性         → 你很焦虑时，优先端出"上次你也焦虑时聊的事"

    use_emotion=False 用于消融实验：对比情感加权到底有没有用。
    """
    from app.memory import long_term  # 延迟导入，避免循环依赖
    from app.retrieval import memory_retriever

    top_k = top_k or settings.memory_top_k

    # 用户档案始终全量注入：它是"确定的结论"，条数少、价值高
    profile = await long_term.get_profile(user_id)

    # 记忆走语义检索（含情感加权）
    #
    # 但先判断"这句话是否关于用户本人"。
    # 问"今天天气怎么样"时不该去翻记忆库 —— 既浪费 token 也会干扰模型注意力。
    # 为什么不用相似度门槛代替？实测相关与无关问题的相似度区间完全重叠
    # （相关 0.353~0.483，无关 0.168~0.473），门槛切不开，切了会误杀。
    use_memory = True
    intent_reason = "已关闭意图判断"
    if settings.enable_intent_gate:
        from app.orchestration.intent import is_about_user

        use_memory, intent_reason = is_about_user(user_input)

    scored = []
    if use_memory:
        scored = await memory_retriever.retrieve_memories(
            user_id,
            user_input,
            top_k=top_k,
            candidate_k=settings.memory_candidate_k,
            emotion_state=emotion_state if use_emotion else None,
            use_emotion=use_emotion,
        )
    else:
        logger.debug(f"意图判断为与用户无关，跳过记忆检索（{intent_reason}）")

    memories = [
        s.memory
        for s in scored
        if s.importance >= settings.memory_min_importance
    ]

    # 知识库检索（RAG）
    # 关键：必须按相似度过滤。实测发现不过滤时，无关问题也会把整份文档塞进提示词，
    # 白白多烧几百 token，还会干扰模型对当前问题的注意力。
    knowledge: list[dict] = []
    if settings.enable_knowledge:
        raw = await memory_retriever.retrieve_knowledge(
            user_input, top_k=settings.knowledge_top_k
        )
        knowledge = [
            k for k in raw if k.get("similarity", 0.0) >= settings.knowledge_min_similarity
        ]
        dropped = len(raw) - len(knowledge)
        if dropped:
            logger.debug(
                f"知识库召回 {len(raw)} 条，因相似度低于 "
                f"{settings.knowledge_min_similarity} 丢弃 {dropped} 条"
            )

    # 情绪姿态
    emotion_block = build_emotion_block(emotion_state) if use_emotion else ""

    bundle = build_messages(
        user_input=user_input,
        history=history,
        profile=profile,
        memories=memories,
        knowledge=knowledge,
        emotion_block=emotion_block,
        # persona=None 时用内置的 DEFAULT_PERSONA。
        # 传了就用调用方的（OpenAI 兼容入口会把客户端自带的人格传进来）。
        persona=persona,
    )
    # 把打分细节挂上，便于调试和消融实验
    bundle.scored = scored  # type: ignore[attr-defined]
    bundle.emotion_used = bool(emotion_block)  # type: ignore[attr-defined]
    return bundle


def build_emotion_block(emotion_state) -> str:
    """根据情绪状态构造"回复姿态"段。

    关键设计：**低置信度时退回中性策略**。

    为什么需要这个兜底？
    情绪识别一定会有误差（实测词典方法的局限见测试脚本）。如果识别错了，
    用户明明很平静，系统却用"先接住他的情绪"的姿态回应，会显得莫名其妙甚至冒犯。

    所以：宁可退回中性、少做一点，也不要用错误的姿态去回应。
    这是"承认能力边界"的工程体现，也是面试可以讲的取舍。
    """
    if emotion_state is None:
        return ""

    # 中性状态不注入姿态段。
    # 为什么？因为"保持正常、务实的语气"这种指令模型本来就会执行，
    # 写进去纯粹浪费 token —— 实测发现中性时也占了 107 token。
    # 只在情绪状态确实偏离中性、姿态指令真的能改变行为时才注入。
    if abs(emotion_state.valence) < 0.15 and abs(emotion_state.stable_valence) < 0.15:
        return ""

    from app.memory.emotion import format_emotion_block, pick_strategy

    strategy = pick_strategy(emotion_state)
    return format_emotion_block(emotion_state, strategy)
