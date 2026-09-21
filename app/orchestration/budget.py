"""Token 预算调度：保证提示词长度可控。

问题是什么？
  聊到第 50 轮时，光历史就有几千 token，小模型的上下文直接爆掉。
  粗暴的解法是"只留最近 10 轮"，但这会丢掉更早的重要信息 ——
  用户三天前定的目标比刚才那句"嗯嗯"重要得多。

本模块的思路：**给每个段落分配配额，超预算时按优先级砍。**

  段落       默认配额   超预算时
  ─────────────────────────────────────────
  人设        200      永不裁剪（丢了它模型就没人格了）
  当前输入     200      永不裁剪（丢了它就没问题了）
  用户档案     150      最后才砍（这是确定的结论，价值最高）
  相关记忆     300      从分数最低的开始砍
  参考资料     300      先砍（知识库召回的，缺一条回答还能凑合）
  对话历史     700      从最早的消息开始砍

  合计       1850

为什么要"从最早的开始砍历史"，而不是平均砍？
  因为对话的连贯性靠的是**最近的上下文**。
  用户说"那这个怎么办"，指的是上一句说的事；
  十轮前那句"你好"砍掉完全不影响理解。
  这是局部性原理在对话上的体现。

一个容易忽略的细节：
  裁剪必须发生在**组装之前**，而不是组装之后再去截字符串。
  因为截字符串会把一个句子切成两半，模型读到残句反而更容易出错。
  所以这里的做法是"决定给每段多少空间"，而不是"组装完再切"。
"""

from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import logger
from app.llm.base import Message

# 各段默认配额（中文约 1 字 1 token）
DEFAULT_BUDGET: dict[str, int] = {
    "persona": 200,
    "profile": 150,
    "memories": 300,
    "knowledge": 300,
    "history": 700,
    "input": 200,
}

# 总预算上限。超过它就要开始裁剪。
# 为什么定 2400？qwen-flash 的上下文窗口远大于这个数，但：
#   1. 输入越长，小模型的注意力越分散（lost in the middle）
#   2. 输入 token 是花钱的
#   3. 我们实测过：注入内容稳定在 300~1300 token 时效果最好
DEFAULT_TOTAL_BUDGET = 2400

# 永不裁剪的段落
PROTECTED = {"persona", "input"}


@dataclass
class BudgetPlan:
    """一次裁剪的决策结果。留着是为了可解释性 —— 被问"你砍了什么"能答上来。"""

    total_budget: int
    used_before: int
    used_after: int
    trimmed: dict = field(default_factory=dict)   # 每段砍掉多少 token
    dropped_memories: int = 0
    dropped_knowledge: int = 0
    dropped_history: int = 0
    truncated_input: bool = False

    @property
    def saved(self) -> int:
        return max(0, self.used_before - self.used_after)

    def summary(self) -> str:
        parts = [f"预算 {self.total_budget}，{self.used_before}→{self.used_after}（省 {self.saved}）"]
        if self.dropped_memories:
            parts.append(f"丢记忆 {self.dropped_memories} 条")
        if self.dropped_knowledge:
            parts.append(f"丢资料 {self.dropped_knowledge} 条")
        if self.dropped_history:
            parts.append(f"丢历史 {self.dropped_history} 条")
        if self.truncated_input:
            parts.append("当前输入被截断")
        return " | ".join(parts)


def est(text: str) -> int:
    """估算 token 数。中文约 1 字 1 token。

    为什么不做精确计算？
    预算调度的目的是"防止提示词爆炸"，需要的是量级判断而不是精确计费。
    引入 tokenizer 依赖（还要为每个模型加载不同的词表）不值当。
    """
    return len(text)


def trim_text_to(text: str, limit: int, *, keep_tail: bool = True) -> str:
    """把一段文字裁到指定长度。

    keep_tail=True 保留尾部（历史消息保留后半句更自然，因为重点通常在后面）；
    keep_tail=False 保留头部。

    注意我们裁的是"整段文字"，而不是在每个 section 内乱切 ——
    调用方负责保证裁完的内容仍然是可读的。
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if keep_tail:
        return "…" + text[-limit:]
    return text[:limit] + "…"


def plan_and_trim(
    *,
    persona: str,
    profile_text: str,
    memory_lines: list[tuple[str, int, int]],  # (文本, 分数, 记忆id)
    knowledge_lines: list[str],
    history: list[Message],
    user_input: str,
    total_budget: int | None = None,
    budget: dict[str, int] | None = None,
) -> tuple[dict, BudgetPlan]:
    """按预算裁剪各段，返回 (裁剪后的内容, 裁剪报告)。

    输入是"各段的候选内容"，输出是"每段最终保留多少"。
    真正的拼装交给 prompt_builder —— 那个模块只管格式，这个模块只管空间。
    职责分开的好处：调预算策略不用碰提示词格式，反之亦然。
    """
    total_budget = total_budget or settings.max_prompt_tokens
    budget = {**DEFAULT_BUDGET, **(budget or {})}

    # ---- 先算现状 ----
    profile_tokens = est(profile_text)
    memory_tokens = [s for _, s, _ in memory_lines]  # 第二项是 token 数
    knowledge_tokens = [est(k) for k in knowledge_lines]
    history_tokens = [est(m.content) for m in history]
    input_tokens = est(user_input)
    persona_tokens = est(persona)

    used_before = (
        persona_tokens + profile_tokens + sum(memory_tokens)
        + sum(knowledge_tokens) + sum(history_tokens) + input_tokens
    )

    plan = BudgetPlan(total_budget=total_budget, used_before=used_before, used_after=0)

    # ---- 输入本身超预算：先截输入（这是唯一会动 protected 的情况）----
    # 用户发了一篇 5000 字的长文，这时候不截就没法玩了
    if input_tokens > budget["input"]:
        overflow = input_tokens - budget["input"]
        user_input = trim_text_to(user_input, budget["input"], keep_tail=True)
        input_tokens = budget["input"]
        plan.truncated_input = True
        logger.debug(f"当前输入超预算，截断 {overflow} token")

    # ---- 如果还没超，直接返回 ----
    if used_before <= total_budget:
        plan.used_after = used_before
        return {
            "persona": persona,
            "profile_text": profile_text,
            "memory_lines": memory_lines,
            "knowledge_lines": knowledge_lines,
            "history": history,
            "user_input": user_input,
        }, plan

    # ---- 开始按优先级裁剪 ----
    over = used_before - total_budget
    logger.debug(f"提示词超预算 {over} token，开始裁剪")

    # ① 先砍参考资料（知识库召回的，缺一条还能凑合）
    keep_knowledge = list(knowledge_lines)
    keep_k_tokens = list(knowledge_tokens)
    while over > 0 and keep_knowledge:
        dropped = keep_k_tokens.pop()   # 从最不像的（排在后面的）开始砍
        keep_knowledge.pop()
        over -= dropped
        plan.dropped_knowledge += 1
        plan.trimmed["knowledge"] = plan.trimmed.get("knowledge", 0) + dropped

    # ② 再砍最早的历史
    keep_history = list(history)
    keep_h_tokens = list(history_tokens)
    while over > 0 and keep_history:
        dropped = keep_h_tokens.pop(0)   # 从最早的消息开始砍
        keep_history.pop(0)
        over -= dropped
        plan.dropped_history += 1
        plan.trimmed["history"] = plan.trimmed.get("history", 0) + dropped

    # ③ 再砍分数最低的记忆
    #    注意 memory_lines 应该已按分数从高到低排序，所以从尾部砍
    keep_memories = list(memory_lines)
    while over > 0 and keep_memories:
        _, dropped, _ = keep_memories.pop()
        over -= dropped
        plan.dropped_memories += 1
        plan.trimmed["memories"] = plan.trimmed.get("memories", 0) + dropped

    # ④ 最后才动档案，而且是截断而不是丢弃
    #    因为档案是"确定的结论"，丢一条比截短更糟
    keep_profile = profile_text
    if over > 0 and profile_tokens > 60:
        allowed = max(60, profile_tokens - over)
        keep_profile = trim_text_to(profile_text, allowed, keep_tail=False)
        cut = profile_tokens - est(keep_profile)
        over -= cut
        plan.trimmed["profile"] = cut

    # ⑤ 到这里还超，说明 protected 的段落本身就超了 —— 记一下，不硬砍
    if over > 0:
        logger.warning(
            f"裁剪后仍超预算 {over} token（人设与当前输入受保护，不裁剪）。"
            f"建议调大 MAX_PROMPT_TOKENS 或缩短人设"
        )

    plan.used_after = (
        persona_tokens + est(keep_profile) + sum(t for _, t, _ in keep_memories)
        + sum(est(k) for k in keep_knowledge) + sum(est(m.content) for m in keep_history)
        + input_tokens
    )

    logger.debug(f"裁剪完成：{plan.summary()}")

    return {
        "persona": persona,
        "profile_text": keep_profile,
        "memory_lines": keep_memories,
        "knowledge_lines": keep_knowledge,
        "history": keep_history,
        "user_input": user_input,
    }, plan
