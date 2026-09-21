"""评估集定义。

为什么要有这个文件？
前面每一步优化都单独测过（Recall@5、情感一致率、token 曲线），
但那些是"分项测试"。真正能写进简历的是一个**统一评估集上的对照结果**：

    配置                         Recall@5   忠实度   情感恰当性   延迟    token
    基线：小模型 + 全量历史
    + 语义检索
    + 情感加权
    + 预算调度
    完整方案

评估集由三部分组成：
  1. 一批**用户事实**（植入记忆库，模拟用户聊过这些）
  2. 一批**问题**，每题标注"应该召回哪条事实"
  3. 一批**情感场景**，用于评估回复姿态是否恰当

标注原则（很重要）：
  "应该召回哪条"是**人工标注**的，不是让模型判断的。
  否则评估就成了"模型给自己打分"，没有意义。
"""

from dataclasses import dataclass, field


@dataclass
class EvalUser:
    """一个评估用的虚拟用户，含他"聊过"的事实。"""

    user_id: str
    facts: list[tuple[str, str, float]]  # (内容, 类型, 重要度)
    profile_hint: str = ""


@dataclass
class EvalCase:
    """一道评估题。"""

    question: str
    expect_keywords: list[str]  # 回答里应出现的关键词（用于自动判定）
    should_recall: list[str]    # 应该被召回的事实的特征词（人工标注，算 Recall 用）
    category: str = "memory"    # memory / knowledge / emotion / refusal
    note: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)


# ============================================================ 评估用户
#
# 刻意设计成"事实分散、有正负情绪、有易混淆项"，
# 这样才测得出检索排序的能力。

EVAL_USER = EvalUser(
    user_id="eval_user_01",
    facts=[
        # 核心身份
        ("用户叫陈望舒，是一名做推荐系统的算法工程师", "fact", 0.95),
        ("用户在深圳工作，团队规模大约十个人", "fact", 0.70),
        # 目标与项目
        ("用户在做一个面向中小商家的选品推荐工具", "goal", 0.90),
        ("用户的目标是今年内让这个工具跑通第一批真实商家", "goal", 0.85),
        ("用户打算在十月份开始找第一批种子商家", "goal", 0.65),
        # 情绪与担忧
        ("用户担心自己的推荐效果比不过大厂成熟方案", "emotion", 0.80),
        ("用户因为项目进度慢而焦虑，睡眠质量变差了", "emotion", 0.75),
        ("用户在一次内部分享时被质疑方案可行性，当时很受打击", "event", 0.70),
        # 偏好与习惯
        ("用户习惯先做小范围验证再扩大，不喜欢一次性大改动", "preference", 0.60),
        ("用户喜欢用 Python 和 PyTorch，不太喜欢 Java 生态", "preference", 0.50),
        # 易混淆项（测试检索是否只挑相关的）
        ("用户养了一只叫元宝的橘猫", "fact", 0.45),
        ("用户上个月去成都出差了一周", "event", 0.40),
        ("用户的妹妹今年参加高考", "fact", 0.55),
        ("用户喜欢在周末爬山", "preference", 0.35),
        ("用户最近在读一本讲决策心理的书", "event", 0.30),
    ],
)


# ============================================================ 评估题
#
# should_recall 里的词是"应该被召回的事实"的特征词，人工标注。
# 判分方式：检索结果里是否包含至少一条匹配这些特征词的记忆。

EVAL_CASES: list[EvalCase] = [
    # ---- 记忆类：问已经存下来的信息 ----
    EvalCase(
        question="你还记得我是做什么的吗？",
        expect_keywords=["陈望舒", "算法"],
        should_recall=["推荐系统", "算法"],
        category="memory",
        note="问身份",
    ),
    EvalCase(
        question="我最近在推进的项目是什么？",
        expect_keywords=["选品"],
        should_recall=["选品", "商家"],
        category="memory",
        note="问项目",
    ),
    EvalCase(
        question="我今年想达成什么目标？",
        expect_keywords=["商家"],
        should_recall=["今年", "种子商家", "跑通"],
        category="memory",
        note="问目标",
    ),
    EvalCase(
        question="我平时更喜欢怎么推进工作？",
        expect_keywords=["小范围", "验证"],
        should_recall=["小范围", "验证"],
        category="memory",
        note="问偏好",
    ),
    EvalCase(
        question="我主要担心什么？",
        expect_keywords=["推荐效果", "大厂"],
        should_recall=["推荐效果", "大厂"],
        category="memory",
        note="问担忧",
    ),
    EvalCase(
        question="我最近状态怎么样？",
        expect_keywords=["焦虑", "睡眠"],
        should_recall=["焦虑", "睡眠", "进度"],
        category="memory",
        note="问状态",
    ),
    EvalCase(
        question="我用什么技术栈？",
        expect_keywords=["Python"],
        should_recall=["Python", "PyTorch"],
        category="memory",
        note="问技术栈",
    ),
    EvalCase(
        question="我之前是不是被质疑过？",
        expect_keywords=["质疑", "打击"],
        should_recall=["质疑", "分享", "打击"],
        category="memory",
        note="问经历",
    ),
    # ---- 干扰项：问的是没提过的事，应该说明不知道 ----
    EvalCase(
        question="我的车牌号是多少？",
        expect_keywords=["不知道", "没有", "没提", "无法", "不清楚"],
        should_recall=[],
        category="refusal",
        note="问没提过的信息（测反幻觉）",
    ),
    EvalCase(
        question="我上次体检的结果怎么样？",
        expect_keywords=["不知道", "没有", "没提", "无法", "不清楚"],
        should_recall=[],
        category="refusal",
        note="问没提过的信息（测反幻觉）",
    ),
]


# ============================================================ 情感场景
#
# 用于评估"情感姿态是否改变了回复的恰当性"。
# 判定方式：规则检查（是否出现说教句式、长度是否克制）+ 人工盲评。

EMOTION_SCENARIOS = [
    {
        "name": "焦虑求助",
        "history": [
            ("我那个选品工具又卡住了，这周什么都没推进", "卡在哪一步了？可以具体说说。"),
            ("我真的很焦虑，感觉再这样下去今年目标肯定完不成", "这种压力我理解，先别急着下结论。"),
        ],
        "question": "我该怎么办？",
        "expect_tone": "先接情绪，不急着给方案",
        # 姿态要求里明确禁止的句式，出现即扣分
        "forbidden": ["你应该", "其实你可以", "建议你", "问自己三个问题", "第一步：", "1."],
    },
    {
        "name": "兴奋分享",
        "history": [
            ("有个商家愿意试用我的工具了！", "这是个好的开始。"),
        ],
        "question": "我觉得这事有戏！",
        "expect_tone": "一起放大，追问细节",
        "forbidden": ["不过要小心", "但是风险", "别高兴太早"],
    },
    {
        "name": "低落倾诉",
        "history": [
            ("昨天内部分享被质疑了，感觉自己做的方向不对", "被质疑确实难受。"),
            ("最近都不太想打开代码了", "听起来你有点耗尽了。"),
        ],
        "question": "我是不是不适合做这个",
        "expect_tone": "温和陪伴，不给大方案",
        "forbidden": ["你应该", "我建议你", "你要相信", "加油"],
    },
]


# ============================================================ 知识库评估
#
# 针对 docs/kb_sample/ 里的《中间件设计笔记》，测回答是否有资料支撑。

KNOWLEDGE_CASES = [
    EvalCase(
        question="我的中间件在做 token 预算调度时，哪些段落是绝对不能裁剪的？",
        expect_keywords=["人设", "当前输入"],
        should_recall=[],
        category="knowledge",
        note="应从知识库答出人设与输入不可裁剪",
    ),
    EvalCase(
        question="中间件分了几层？分别负责什么？",
        expect_keywords=["六", "接入", "检索", "编排"],
        should_recall=[],
        category="knowledge",
        note="应从知识库答出分层设计",
    ),
    EvalCase(
        question="记忆的语义去重阈值通常设多少？",
        expect_keywords=["0.8"],
        should_recall=[],
        category="knowledge",
        note="应从知识库答出阈值",
    ),
    EvalCase(
        question="记忆衰减的半衰期一般是多少天？",
        expect_keywords=["30"],
        should_recall=[],
        category="knowledge",
        note="应从知识库答出半衰期",
    ),
]


def all_cases() -> list[EvalCase]:
    return EVAL_CASES + KNOWLEDGE_CASES


def summary() -> dict:
    cases = all_cases()
    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    return {
        "用户事实条数": len(EVAL_USER.facts),
        "评估题总数": len(cases),
        "按类别": by_cat,
        "情感场景数": len(EMOTION_SCENARIOS),
    }
