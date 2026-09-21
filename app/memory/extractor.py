"""记忆抽取器：让模型自己判断"这轮对话里有什么值得长期记住的"。

这是"记忆"和"聊天记录"的分界线：
  聊天记录 = 流水账，全存
  记忆     = 经过判断后留下的结论，只存值得的

一次抽取的完整过程：
  一轮对话 → 提示词 → 小模型 → JSON → 清洗解析 → 一批 MemoryItem

工程上最容易翻车的三个地方，这里都处理了：
  1. 小模型不老实：会把 JSON 包在 ```json 代码块里，或加一句"好的，以下是..." → 三层解析
  2. 抽取失败不能影响主流程：解析不出来就返回空列表，绝不抛异常给用户
  3. 抽取很慢（多一次模型调用）→ 由调用方决定是否放后台异步执行
"""

import json
import re
from dataclasses import dataclass, field

from app.core.logging import logger
from app.llm.base import LLMClient, LLMError, Message

# 允许的记忆类型。限制枚举值能显著降低模型乱造类型的概率
MEMORY_TYPES = {"fact", "preference", "goal", "event", "emotion"}

EXTRACT_PROMPT = """你是一个记忆抽取器。从下面这轮对话中，提取值得**长期记住**的用户信息。

判断标准（很重要）：
1. 只提取关于**用户本人**的稳定信息：客观事实、偏好、目标、重要经历、情绪触发点
2. 不要提取：寒暄、天气、常识、AI 说的话、一次性的临时状态
3. 如果这轮对话没有任何值得记住的内容，就返回空数组 —— 这是完全正常的，不要硬凑
4. 每条记忆必须能脱离本次对话独立看懂，所以要写清楚主语

importance 打分标准：
  0.9~1.0  身份核心：姓名、职业、长期重大目标
  0.6~0.8  重要：正在做的事、明确的偏好、重要的人
  0.3~0.5  一般：临时计划、轻微偏好
  0.1~0.2  琐事：可能很快就没用了

type 只能从这几个值里选：fact(客观事实) / preference(偏好) / goal(目标) / event(经历) / emotion(情绪触发点)

valence 是这轮对话里用户的情绪好坏（-1 极负面 ~ +1 极正面），arousal 是激烈程度（0 平静 ~ 1 激动）。

只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块。

对话内容：
用户：{user_msg}
助手：{assistant_msg}

输出格式：
{{"memories": [{{"content": "用户叫小明", "type": "fact", "importance": 0.95, "valence": 0.3, "arousal": 0.2}}]}}"""


@dataclass
class MemoryItem:
    """一条待写入的记忆。"""

    content: str
    type: str = "fact"
    importance: float = 0.5
    valence: float = 0.0
    arousal: float = 0.0
    emotion_label: str | None = None
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------- JSON 解析


def _strip_code_fence(text: str) -> str:
    """剥掉 markdown 代码块围栏。

    小模型最常见的"不听话"就是这种输出：
        ```json
        {"memories": [...]}
        ```
    """
    t = text.strip()
    # 匹配 ```json ... ``` 或 ``` ... ```
    m = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 只有开头有围栏、结尾没有的情况
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _find_json_object(text: str) -> str | None:
    """从一堆文字里抠出第一个完整的 JSON 对象。

    模型经常说"好的，以下是提取结果：{...} 希望有帮助" —— 前后全是废话。
    这里用括号配对的方式找到真正的那个对象，而不是靠找第一个 { 和最后一个 }。
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_memories(text: str) -> list[MemoryItem]:
    """把模型输出解析成 MemoryItem 列表。三层兜底，失败返回空列表。

    三层：直接解析 → 剥代码块 → 正则抠 JSON 对象
    """
    if not text or not text.strip():
        return []

    candidates: list[str] = [text.strip()]

    stripped = _strip_code_fence(text)
    if stripped != candidates[0]:
        candidates.append(stripped)

    found = _find_json_object(stripped)
    if found and found not in candidates:
        candidates.append(found)

    data = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue

    if data is None:
        logger.warning(f"记忆抽取：三层解析全部失败，原始输出前 200 字：{text[:200]}")
        return []

    # 容忍模型直接返回数组（而不是 {"memories": [...]}）
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        raw_items = data.get("memories") or data.get("items") or []
    else:
        return []

    if not isinstance(raw_items, list):
        return []

    items: list[MemoryItem] = []
    for r in raw_items:
        if not isinstance(r, dict):
            continue
        content = str(r.get("content", "")).strip()
        if len(content) < 2:  # 空内容或单字，直接丢
            continue

        mtype = str(r.get("type", "fact")).strip().lower()
        if mtype not in MEMORY_TYPES:
            mtype = "fact"  # 模型造了新类型，兜回默认值而不是丢弃整条

        items.append(
            MemoryItem(
                content=content,
                type=mtype,
                importance=_clamp(r.get("importance"), 0.5),
                valence=_clamp(r.get("valence"), 0.0, lo=-1.0),
                arousal=_clamp(r.get("arousal"), 0.0),
                emotion_label=infer_emotion_label(
                    _clamp(r.get("valence"), 0.0, lo=-1.0),
                    _clamp(r.get("arousal"), 0.0),
                ),
                raw=r,
            )
        )
    return items


def _clamp(value, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """把数值夹到合法区间。模型偶尔会返回 1.5 或 "0.8" 这种。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def infer_emotion_label(valence: float, arousal: float) -> str:
    """把效价-唤醒两个数字翻译成人能看懂的情绪标签。

    为什么要这一步？
    - 数值适合计算（Day 5 的加权检索要用），但不适合给人看，也不适合直接塞进提示词；
    - 模型给出的 emotion_label 经常不稳定（同样是焦虑，有时写"担心"有时写"紧张"），
      所以这里**用数值反推标签**，保证标签体系是固定的、可统计的。

    这就是情感计算里经典的二维模型（Russell 环形模型）：
        高唤醒 + 负效价 = 焦虑/愤怒
        低唤醒 + 负效价 = 低落/悲伤
        高唤醒 + 正效价 = 兴奋/开心
        低唤醒 + 正效价 = 平静/满足
    """
    if valence >= 0.3:
        return "兴奋" if arousal >= 0.5 else "平静"
    if valence <= -0.3:
        if arousal >= 0.5:
            return "焦虑" if arousal < 0.8 else "愤怒"
        return "低落"
    return "中性"


# ---------------------------------------------------------------- 抽取主流程


class MemoryExtractor:
    """用一个小模型调用，把一轮对话变成若干条记忆。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        user_msg: str,
        assistant_msg: str,
        *,
        min_importance: float = 0.0,
    ) -> list[MemoryItem]:
        """抽取一轮对话里的记忆。任何失败都返回空列表，不抛异常。"""
        # 太短的对话不值得花一次模型调用
        if len(user_msg.strip()) < 4:
            return []

        prompt = EXTRACT_PROMPT.format(
            user_msg=user_msg[:2000],        # 防止超长输入把 prompt 撑爆
            assistant_msg=assistant_msg[:1000],
        )

        try:
            reply = await self.llm.chat(
                [Message("user", prompt)],
                temperature=0.1,   # 抽取是"判断题"不是"创作题"，要稳定
            )
        except LLMError as e:
            logger.warning(f"记忆抽取失败（模型调用出错）：{e}")
            return []

        items = parse_memories(reply.text)
        if min_importance > 0:
            items = [i for i in items if i.importance >= min_importance]

        logger.debug(f"抽取到 {len(items)} 条记忆：{[i.content[:20] for i in items]}")
        return items
