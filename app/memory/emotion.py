"""情绪分析器：给每轮对话打情绪分。

用二维模型（效价-唤醒）而不是直接分类，原因：
  1. 数值可以计算（加权、平滑、算距离），"焦虑"这个标签算不了；
  2. 可以画情绪轨迹，用户可以直观看到"这几天心情怎么变的"；
  3. 连续值能表达强度差异，"有点烦"和"快崩溃了"是完全不同的状态。

一个关键设计决策：**默认走词典，不额外调模型**。

为什么？因为我们已经有抽取器在调模型了，再为情绪单独调一次意味着：
  - 每轮对话从 1 次模型调用变成 2 次（加上抽取就是 3 次）
  - 延迟再涨几百毫秒
  - 成本直接翻倍

而中文情绪表达的词汇集中度很高，"焦虑/烦/压力大/崩溃"这些词覆盖了绝大多数场景。
词典打分是**零延迟、零成本、结果稳定**的，正好适合放在关键路径上。
词典判不准的（比如反讽、隐晦表达）才需要模型——那个交给抽取器顺带处理，
反正它已经在读了。

这个取舍本身就是面试可讲的点：**不是所有地方都值得调模型。**
"""

import re
from dataclasses import dataclass

from app.core.logging import logger

# ============================================================ 情绪词典
#
# 每项是 (词, 强度)。强度范围 0~1，表示这个词的情绪浓度。
# 注意：词典打分不需要覆盖所有表达，只需要覆盖高频表达。
# 漏掉的会被判成中性，代价可接受；关键是别把中性误判成极端。

# 负向词
NEGATIVE_LEXICON: dict[str, float] = {
    # 高唤醒负向：焦虑、愤怒、恐慌
    "焦虑": 0.75, "着急": 0.6, "急死": 0.9, "烦躁": 0.7, "烦": 0.5, "生气": 0.8,
    "愤怒": 0.9, "气死": 1.0, "崩溃": 1.0, "受不了": 0.85, "慌了": 0.8, "慌": 0.6,
    "紧张": 0.6, "害怕": 0.75, "恐惧": 0.9, "担心": 0.55, "怕": 0.5, "压力大": 0.7,
    "压力": 0.5, "崩溃了": 1.0, "绝望": 1.0, "痛苦": 0.8, "难受": 0.6, "头疼": 0.5,
    "抓狂": 0.9, "恶心": 0.6, "气人": 0.7, "烦人": 0.6, "急": 0.5, "暴躁": 0.8,
    # 低唤醒负向：低落、沮丧、疲惫
    "低落": 0.6, "难过": 0.65, "伤心": 0.75, "沮丧": 0.7, "失落": 0.6, "郁闷": 0.55,
    "累": 0.45, "疲惫": 0.6, "没劲": 0.55, "提不起": 0.6, "无聊": 0.4, "孤独": 0.6,
    "委屈": 0.65, "无奈": 0.5, "没希望": 0.85, "放弃": 0.7, "迷茫": 0.55, "丧": 0.6,
    "不开心": 0.6, "没意思": 0.5, "心累": 0.7, "撑不住": 0.85, "坚持不下来": 0.8,
    "自责": 0.7, "后悔": 0.6, "羞愧": 0.7, "惭愧": 0.6,
}

# 正向词
POSITIVE_LEXICON: dict[str, float] = {
    # 高唤醒正向：兴奋、开心
    "开心": 0.7, "高兴": 0.7, "兴奋": 0.8, "太好了": 0.85, "棒": 0.7, "爽": 0.75,
    "激动": 0.8, "惊喜": 0.85, "爱了": 0.8, "牛逼": 0.8, "厉害": 0.6, "成功": 0.7,
    "通过了": 0.8, "搞定了": 0.75, "拿到了": 0.8, "涨薪": 0.8, "offer": 0.8,
    # 低唤醒正向：平静、满足
    "满意": 0.6, "舒服": 0.6, "踏实": 0.55, "安心": 0.55, "放松": 0.5, "不错": 0.5,
    "挺好": 0.55, "还行": 0.35, "顺利": 0.6, "喜欢": 0.5, "感谢": 0.5, "谢谢": 0.4,
    "有意思": 0.55, "值得": 0.5, "有希望": 0.6,
}

# 效价的固定参考强度。差值达到它就算满值（±1.0）。
# 取 1.5 的依据：大约相当于两个中等强度情绪词同时出现。
VALENCE_REFERENCE = 1.5

# 唤醒度的参考强度。取 2.0 而不是更小的值，是为了避免饱和 ——
# 实测发现用小分母时，"今天天气不错"这种轻描淡写也会算出 0.28 的唤醒度，
# 而它实际上应该接近 0.1。宁可让极端情绪多"顶格"，也不要让轻微表达虚高。
AROUSAL_REFERENCE = 2.0

# ---- 显式情绪词表：文本里直接说了情绪名，就该以此为准 ----
#
# 为什么需要这一层？
# 效价-唤醒二维模型有个固有的模糊区：焦虑和愤怒都是"高唤醒 + 负效价"，
# 光看两个数值分不开。实测就出现了"我最近特别焦虑"被判成"愤怒"。
#
# 但中文表达里，用户经常**直接说出情绪名**（"我很焦虑""我很难过"）。
# 这时候那个词就是最可靠的信号，比推算出来的数值更该被信任。
# 这是词典法相对模型法的一个优势：可以直接用上这个信号。
ANGER_WORDS = ("生气", "愤怒", "气死", "气人", "暴躁", "抓狂", "受不了", "恶心")
SAD_WORDS = ("难过", "伤心", "低落", "沮丧", "失落", "郁闷", "委屈", "孤独", "绝望", "没希望", "丧")
ANXIETY_WORDS = ("焦虑", "着急", "烦躁", "紧张", "害怕", "恐惧", "担心", "慌", "压力", "急")

# 高唤醒标志词（用于修正 arousal）
AROUSAL_BOOSTERS = (
    "特别", "非常", "超级", "极其", "太", "真的", "简直", "快", "！！！", "!!",
    "死", "爆", "疯了", "不行了",
)
# 低唤醒标志词
AROUSAL_DAMPENERS = ("有点", "稍微", "还好", "一般", "略", "勉强", "算是")

# 否定词：出现在情绪词前面会反转极性
NEGATIONS = ("不", "没", "别", "无", "非", "毫无")

# 程度副词修饰
INTENSIFIERS = {"很": 1.3, "非常": 1.5, "特别": 1.5, "超级": 1.6, "太": 1.4, "极其": 1.7, "真的": 1.2}
WEAKENERS = {"有点": 0.7, "稍微": 0.6, "略": 0.6, "还好": 0.5, "一点": 0.7}


@dataclass
class EmotionScore:
    """一轮对话的情绪打分结果。"""

    valence: float = 0.0      # 效价 -1 极负 ~ +1 极正
    arousal: float = 0.0      # 唤醒度 0 平静 ~ 1 激动
    label: str = "中性"
    confidence: float = 0.0   # 置信度 0~1，低了就退回中性策略
    matched: list[str] = None  # 命中了哪些词，便于调试

    # 这个分数是从哪来的：text / image / fused。
    # 未来图片情绪识别上线后，会把图片分和文字分融合（fused），
    # 保留来源便于排查"这个情绪到底是哪来的"。
    source: str = "text"

    def __post_init__(self) -> None:
        if self.matched is None:
            self.matched = []


def infer_label(valence: float, arousal: float) -> str:
    """把效价-唤醒两个数字翻译成情绪标签。

    这就是心理学里经典的二维情绪模型（罗素环形模型）：
        高唤醒 + 负效价 = 焦虑 / 愤怒
        低唤醒 + 负效价 = 低落 / 悲伤
        高唤醒 + 正效价 = 兴奋
        低唤醒 + 正效价 = 平静

    关于"焦虑"和"愤怒"的分界（实测调过）：
    两者都是高唤醒 + 负效价，区别在唤醒程度 —— 愤怒更激烈。
    最初把分界定在 0.85，结果"我最近特别焦虑"被判成了愤怒（它的唤醒算出来 0.9）。
    现在定在 0.75，让典型焦虑表达落在"焦虑"这一侧。
    这个边界本来就是连续的，工程上按实际表达习惯取一个值即可。
    """
    if valence >= 0.25:
        return "兴奋" if arousal >= 0.5 else "平静"
    if valence <= -0.25:
        if arousal >= 0.75:
            return "愤怒"
        if arousal >= 0.4:
            return "焦虑"
        return "低落"
    return "中性"


def _find_word_occurrence(text: str, word: str) -> int:
    """找一个情绪词的"有效出现位置"。

    为什么要专门处理？因为中文没有词边界，直接 find 会踩两个坑：

    坑 1：更长的高强度词会污染短词。
      "烦躁" 里含 "烦"，"不开心" 里含 "开心"，"不错" 里含 "错"。
      如果按词典顺序逐个 find，会同时命中长词和短词，重复计分。

    坑 2（更严重）：否定词会被误判。
      "怕做不成" 里的 "不" 位于 "怕" 之后 —— 它不是否定词，是 "做不成" 的一部分。
      早期版本在词的前 3 个字符里搜否定词，结果把这个 "不" 当成了否定 "怕"，
      把负向情绪翻成了正向，判定结果完全相反（实测："我最近特别焦虑，怕做不成"
      被判成"兴奋，效价 +0.39"）。

    现在的规则：只在**紧邻**情绪词之前的位置（最多 2 个字符）检查否定词，
    而且那 2 个字符必须构成一个独立的否定词（如"不""没""别"），
    不能是更长词的一部分。
    """
    start = 0
    lower = text.lower()  # 词典里有 "offer" 这种英文词，要大小写不敏感
    while True:
        idx = lower.find(word.lower(), start)
        if idx == -1:
            return -1
        # 检查这个词的左边是否已经有一段被"更长的高强度词"覆盖
        # 做法：如果紧邻左边 1~2 个字 + 本词能构成词典里的更长词，就跳过本次出现
        longer_exists = False
        for other in list(POSITIVE_LEXICON) + list(NEGATIVE_LEXICON):
            if other == word:
                continue
            if word in other:
                # word 是 other 的子串，检查这个位置是否正好是 other
                offset = other.find(word)
                begin = idx - offset
                if begin >= 0 and lower[begin : begin + len(other)] == other.lower():
                    longer_exists = True
                    break
        if not longer_exists:
            return idx
        start = idx + 1


def _has_negation(text: str, pos: int) -> bool:
    """检查情绪词前面紧邻位置有没有否定词。

    ⚠️ 中文的一个经典陷阱（实测踩到过）：
        "我最近特别焦虑"
        「特别」是程度副词，但它的第二个字是「别」—— 而「别」是否定词。
        如果只做子串匹配，「特别焦虑」会被读成「别+焦虑」= 不要焦虑 = 正向，
        判定结果完全反了（实测把焦虑判成了兴奋，效价 +0.39）。

    所以规则是：
      1. 先看紧邻位置有没有**程度副词**，有的话说明这里是"非常焦虑"这种结构，
         不该再往否定词上想；
      2. 否定词必须是紧贴词首的独立词，不能是更长词的一部分。

    "不开心"   → 前缀"不"，是否定 → 命中
    "特别焦虑" → 前缀"特别"是程度副词 → 不命中（正确）
    "怕做不成" → 前缀没有否定词（"不"在"怕"后面）→ 不命中（正确）
    """
    if pos == 0:
        return False

    # 规则 1：前缀是程度副词时，不判否定
    for word in list(INTENSIFIERS) + list(WEAKENERS):
        if text.endswith(word, 0, pos):
            return False

    # 规则 2：紧邻 1~2 个字符内找否定词，且它不能是更长词的一部分
    for width in (2, 1):
        start = max(0, pos - width)
        prefix = text[start:pos]
        for neg in NEGATIONS:
            if not prefix.endswith(neg):
                continue
            # 检查这个否定词是不是某个更长词的后缀（如"特别"里的"别"）
            begin = pos - len(neg)
            covered = False
            for word in list(INTENSIFIERS) + list(WEAKENERS) + list(POSITIVE_LEXICON) + list(NEGATIVE_LEXICON):
                if len(word) <= len(neg):
                    continue
                b = begin - (len(word) - len(neg))
                if b >= 0 and text[b : b + len(word)] == word:
                    covered = True
                    break
            if not covered:
                return True
    return False


def _intensity_multiplier(text: str, pos: int, window: int = 4) -> float:
    """检查情绪词前后的程度副词，返回强度倍数。"""
    start = max(0, pos - window)
    before = text[start:pos]
    for word, mult in INTENSIFIERS.items():
        if word in before:
            return mult
    for word, mult in WEAKENERS.items():
        if word in before:
            return mult
    return 1.0


def analyze(text: str) -> EmotionScore:
    """对一句话做情绪打分。纯词典 + 规则，零延迟。"""
    if not text or not text.strip():
        return EmotionScore()

    score = EmotionScore()
    pos_sum = 0.0
    neg_sum = 0.0
    matched: list[str] = []

    for lexicon, is_positive in ((POSITIVE_LEXICON, True), (NEGATIVE_LEXICON, False)):
        for word, strength in lexicon.items():
            idx = _find_word_occurrence(text, word)
            if idx == -1:
                continue
            matched.append(word)
            s = strength * _intensity_multiplier(text, idx)

            if _has_negation(text, idx):
                # "不开心" → 负向；"不焦虑" → 弱正向/中性
                is_positive_effective = not is_positive
            else:
                is_positive_effective = is_positive

            if is_positive_effective:
                pos_sum += s
            else:
                neg_sum += s

    total = pos_sum + neg_sum
    if total == 0:
        # 没命中任何情绪词
        return EmotionScore(valence=0.0, arousal=0.0, label="中性", confidence=0.0, matched=[])

    # 效价计算：用"正负之差 / 固定参考强度"，而不是"正负之差 / 总和"。
    #
    # 为什么不用归一化（(pos-neg)/(pos+neg)）？
    # 因为归一化会把**强度信息丢掉**：只要命中一个正向词，不管它多轻，
    # 结果都是 +1.0。于是"今天天气不错"（轻轻一句肯定）和"我太开心了"
    # （强烈情绪）会得到完全相同的效价 —— 实测发现的这个问题。
    #
    # 固定参考强度取 1.5：大约相当于两个中等强度词同时出现就是满值。
    diff = pos_sum - neg_sum
    score.valence = max(-1.0, min(1.0, diff / VALENCE_REFERENCE))
    score.matched = matched

    # ---- 唤醒度估算 ----
    # 同样基于绝对强度而非归一化，理由一样
    base_arousal = min(1.0, total / AROUSAL_REFERENCE)

    # 高唤醒标志词加成
    if any(b in text for b in AROUSAL_BOOSTERS):
        base_arousal = min(1.0, base_arousal + 0.25)
    # 低唤醒标志词削弱
    if any(d in text for d in AROUSAL_DAMPENERS):
        base_arousal = max(0.0, base_arousal - 0.2)

    # 感叹号密集 → 更激动
    exclaims = text.count("!") + text.count("！")
    if exclaims >= 2:
        base_arousal = min(1.0, base_arousal + 0.2)
    elif exclaims == 1:
        base_arousal = min(1.0, base_arousal + 0.1)

    # 问号多 → 疑惑/焦躁，轻微提升
    if text.count("?") + text.count("？") >= 2:
        base_arousal = min(1.0, base_arousal + 0.1)

    # 负向情绪通常唤醒更高（同样强度下，负面比正面更"闹心"）
    if score.valence < -0.2:
        base_arousal = min(1.0, base_arousal + 0.08)

    score.arousal = round(base_arousal, 3)
    score.valence = round(score.valence, 3)
    score.label = infer_label(score.valence, score.arousal)

    # ---- 显式情绪词优先 ----
    # 如果文本里直接说了情绪名，就以那个词为准，覆盖推算出来的标签。
    # 顺序很重要：愤怒 > 悲伤 > 焦虑，因为同时出现时更具体/更强烈的那个更该被采纳。
    if score.valence < -0.15:
        if any(w in text for w in ANGER_WORDS):
            score.label = "愤怒"
        elif any(w in text for w in SAD_WORDS):
            score.label = "低落"
        elif any(w in text for w in ANXIETY_WORDS):
            score.label = "焦虑"

    # 置信度：命中词越多越可信；句子越长越可能夹带无关内容，所以打个折
    hit_conf = min(1.0, len(matched) / 2.0)
    length_penalty = 1.0 if len(text) <= 40 else max(0.5, 40 / len(text))
    score.confidence = round(hit_conf * length_penalty, 3)

    return score


# ============================================================ 情绪状态维护


@dataclass
class EmotionState:
    """用户当前的情绪状态。"""

    valence: float = 0.0
    arousal: float = 0.0
    label: str = "中性"
    # 长期倾向（慢速 EMA）
    stable_valence: float = 0.0
    stable_arousal: float = 0.0
    trend: str = "平稳"          # 上升 / 下滑 / 平稳
    recent_low: bool = False     # 最近是否持续低落
    turns: int = 0


# EMA 平滑系数。
# 为什么必须平滑？单轮极端情绪不该让状态剧烈跳变 ——
# 用户随口说一句"烦死了"，系统不该立刻判定他处于情绪危机。
# η 越大反应越快但越抖，越小越稳但越迟钝。0.3 是体感上比较自然的折中。
EMA_FAST = 0.30   # 即时状态
EMA_SLOW = 0.10   # 长期倾向

# 判定"持续低落"的阈值
LOW_VALENCE_THRESHOLD = -0.25
LOW_TURNS_THRESHOLD = 2      # 连续几轮算持续
TREND_DELTA = 0.15           # 变化多少算有趋势


def label_for_state(valence: float, arousal: float) -> str:
    """给"状态"（而不是单句打分）定标签。

    和 infer_label 的区别：状态的量级天然比单句小。
    单句说"我特别焦虑"能到 -1.0，但 EMA 平滑后的状态可能只有 -0.3 ——
    因为系统有意不让单轮情绪主导状态。
    如果状态沿用单句那套阈值（±0.25），就会出现"效价 -0.60 却显示中性"
    这种自相矛盾（实测遇到过）。

    所以状态的标签阈值要更宽松。
    """
    if valence >= 0.25:
        return "积极" if arousal >= 0.4 else "平静"
    if valence <= -0.15:
        if arousal >= 0.6:
            return "焦虑"
        return "低落"
    if valence <= -0.05:
        return "略低"
    if valence >= 0.1:
        return "略好"
    return "中性"


def update_state(state: EmotionState, score: EmotionScore) -> EmotionState:
    """用新的一轮打分更新用户情绪状态。

    注意低置信度处理：词典没命中情绪词时 confidence=0，
    这时**不要把状态往中性拉**，而是保持原状。
    否则用户说了十句情绪化的话，中间夹一句"嗯"，状态就被拉回中性了。
    """
    if score.confidence <= 0.0:
        state.turns += 1
        return state

    prev_valence = state.valence

    state.valence = round((1 - EMA_FAST) * state.valence + EMA_FAST * score.valence, 4)
    state.arousal = round((1 - EMA_FAST) * state.arousal + EMA_FAST * score.arousal, 4)
    state.stable_valence = round(
        (1 - EMA_SLOW) * state.stable_valence + EMA_SLOW * score.valence, 4
    )
    state.stable_arousal = round(
        (1 - EMA_SLOW) * state.stable_arousal + EMA_SLOW * score.arousal, 4
    )

    delta = state.valence - prev_valence
    if delta <= -TREND_DELTA:
        state.trend = "下滑"
    elif delta >= TREND_DELTA:
        state.trend = "上升"
    else:
        state.trend = "平稳"

    state.label = label_for_state(state.valence, state.arousal)
    state.turns += 1

    # 持续低落判定：当前状态和长期倾向都偏负
    state.recent_low = (
        state.valence <= LOW_VALENCE_THRESHOLD
        and state.stable_valence <= LOW_VALENCE_THRESHOLD
        and state.turns >= LOW_TURNS_THRESHOLD
    )
    return state


def detect_shift(scores: list[EmotionScore], window: int = 3, threshold: float = 0.35) -> bool:
    """检测情绪是否在恶化。

    做法：比较最近 window 轮和更早 window 轮的平均效价，跌幅超过阈值就报警。
    为什么要单独做这个？因为 EMA 是平滑的，它会刻意"迟钝"，
    而有些情况需要敏感 —— 比如用户连续三轮越说越丧，这时候该主动关心，
    而不是等 EMA 慢慢滑下去。
    """
    vals = [s.valence for s in scores if s.confidence > 0]
    if len(vals) < window * 2:
        return False
    recent = sum(vals[-window:]) / window
    earlier = sum(vals[-window * 2 : -window]) / window
    return (earlier - recent) >= threshold


# ============================================================ 情感化回复策略


@dataclass
class ResponseStrategy:
    """给模型的"姿态指令"。"""

    name: str
    instruction: str


# 这张表是整个情感模块的落地点。
#
# 核心思想：**不要让模型自己猜该用什么语气，我们在中间件里算出来再告诉它。**
# 小模型猜不准语气，但执行明确的指令很在行 —— 这就是"用中间件增强小模型"。
STRATEGIES: dict[str, ResponseStrategy] = {
    "接情绪": ResponseStrategy(
        name="接情绪",
        instruction=(
            "用户此刻情绪比较激动且负面。请：\n"
            "- 先用一句话接住他的情绪，让他感觉被理解\n"
            "- 用短句，不要长篇分析，不要列条目\n"
            "- 这一轮不要给建议，不要用『你应该』『其实你可以』这类句式\n"
            "- 不要急着解决问题，先陪住"
        ),
    ),
    "陪伴": ResponseStrategy(
        name="陪伴",
        instruction=(
            "用户情绪比较低落、能量不足。请：\n"
            "- 语气温和，节奏放慢\n"
            "- 只给一个极小的、几分钟就能完成的下一步动作，不要给整套方案\n"
            "- 不要说教，不要用『加油』『想开点』这类空话\n"
            "- 可以提一句他之前说过的事，让他感到被记得"
        ),
    ),
    "同在": ResponseStrategy(
        name="同在",
        instruction=(
            "用户情绪偏正面且比较兴奋。请：\n"
            "- 和他一起把这份情绪延续下去，回应得热情一点\n"
            "- 可以追问细节，帮他展开想法\n"
            "- 不要泼冷水，不要立刻提醒风险"
        ),
    ),
    "平稳": ResponseStrategy(
        name="平稳",
        instruction=(
            "用户情绪平稳。请：\n"
            "- 保持正常、务实的语气\n"
            "- 可以聊长期规划，可以给结构化的建议"
        ),
    ),
}


def pick_strategy(state: EmotionState) -> ResponseStrategy:
    """根据情绪状态挑一个回复姿态。

    为什么用"低效价 + 高唤醒"和"低效价 + 低唤醒"分开？
    因为这两种状态的正确回应方式完全不同：
      焦虑/愤怒（高唤醒）→ 先让他把情绪说完，这时候给建议他听不进去
      低落（低唤醒）→ 需要一点微小的推动，但不能催，否则压力更大
    如果只按"负面"一个维度处理，两种情况会用同一套话术，效果都不好。
    """
    v, a = state.valence, state.arousal

    if v <= -0.25:
        return STRATEGIES["接情绪"] if a >= 0.5 else STRATEGIES["陪伴"]
    if v >= 0.25:
        return STRATEGIES["同在"] if a >= 0.5 else STRATEGIES["平稳"]
    return STRATEGIES["平稳"]


def format_emotion_block(state: EmotionState, strategy: ResponseStrategy) -> str:
    """把情绪状态和姿态指令格式化成提示词里的一段。"""
    trend_note = ""
    if state.trend == "下滑":
        trend_note = "（注意：他的情绪正在往下走）"
    elif state.recent_low:
        trend_note = "（注意：他已经连续几轮状态不佳）"

    # 标签现算，不依赖调用方有没有设置过 state.label ——
    # 避免出现"效价 -0.60 却标注中性"这种自相矛盾的展示（实测遇到过）
    label = label_for_state(state.valence, state.arousal)

    return (
        "## 用户此刻的状态\n"
        f"情绪：{label}（效价 {state.valence:+.2f}，唤醒 {state.arousal:.2f}）"
        f"{trend_note}\n"
        f"长期倾向：效价 {state.stable_valence:+.2f}\n\n"
        "### 这一轮的回复姿态要求\n"
        f"{strategy.instruction}"
    )


def log_summary(score: EmotionScore) -> str:
    return (
        f"情绪={score.label} 效价={score.valence:+.2f} 唤醒={score.arousal:.2f} "
        f"置信={score.confidence:.2f} 命中词={score.matched[:4]}"
    )
