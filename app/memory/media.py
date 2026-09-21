"""多模态拓展点（占位模块）。

**当前状态：仅定义接口，未实现具体识别逻辑。**

这个模块的存在意义是"先把口子开好"，让未来加图片支持时不用动核心链路。
现在只有文字，但接口按"未来会有图片"来设计。

---

未来要实现的能力（按用户需求）：
  1. **判断一张图是不是表情包**
     表情包有很强的视觉特征：带文字、分辨率低、比例偏方、有明显的表情夸张。
     可以用轻量分类模型，也可以用多模态大模型判断。
  2. **判断表情包里有没有文字**
     这一步是 OCR（文字识别）。
  3. **从文字 + 肢体动作 + 面部表情推断情绪**
     - 文字：表情包的配文本身往往就是情绪的直接表达（"我裂开了"）
     - 面部表情：大笑 / 哭泣 / 无语 / 生气
     - 肢体动作：摊手 / 捂脸 / 竖大拇指

---

为什么要把口子开在"四个位置"而不是只加个识别函数？

因为一张图片进来之后，它要走完整条链路：
  ① 接入层：请求要能带上图片
  ② 情绪层：图片要参与情绪打分（和文字一起决定效价/唤醒）
  ③ 存储层：图片本身和它的识别结果要能存下来
  ④ 检索层：历史图片记忆要能被检索出来，并且能注入提示词

只做 ② 的话，用户发过的表情包第二次就找不到了 —— 那样"表情包记忆"就没意义。

---

设计约定（未来实现时必须遵守）：

1. **识别失败不能影响主流程**
   和现有的记忆抽取一样：识别不出来就当没有图片，对话继续。
   绝不能让一张破图把整个请求搞崩。

2. **结果要结构化，不要散文**
   识别结果要能落到 emotion 模块能直接用的格式：
   `{valence, arousal, label, confidence, source}`

3. **置信度必须带上**
   表情包的情绪判断比文字更不确定（同一张图在不同语境下情绪可能相反），
   所以要带 confidence，让上层决定用不用。

4. **图片不进提示词原文**
   图片本身是二进制，不能塞进 prompt。
   进 prompt 的应该是"这张图的描述"和"它表达的情绪"。
"""

from dataclasses import dataclass, field
from typing import Protocol

# ============================================================ 输入抽象


@dataclass
class MediaInput:
    """一段多模态输入。

    现在只有 text 会被真实填充；image_bytes / image_url 为未来预留。
    """

    text: str = ""
    image_bytes: bytes | None = None
    image_url: str | None = None
    mime_type: str | None = None

    @property
    def has_image(self) -> bool:
        return bool(self.image_bytes or self.image_url)

    @property
    def modality(self) -> str:
        """text / image / text+image"""
        if self.has_image and self.text.strip():
            return "text+image"
        if self.has_image:
            return "image"
        return "text"


@dataclass
class MediaAnalysis:
    """媒体分析的统一结果。所有识别器都返回这个结构。"""

    # 这是什么
    is_sticker: bool = False          # 是不是表情包
    has_text: bool = False            # 图里有没有文字
    ocr_text: str = ""                # 识别出的图内文字
    description: str = ""             # 对画面的文字描述（供提示词使用）

    # 情绪判断（与 emotion.EmotionScore 对齐，便于直接合并）
    valence: float = 0.0
    arousal: float = 0.0
    label: str = "中性"
    confidence: float = 0.0

    # 判断依据，便于调试和解释
    signals: dict = field(default_factory=dict)
    # 面部表情 / 肢体动作等细节（未来填）
    features: dict = field(default_factory=dict)

    # 用哪个分析器得到的（未来可能并行跑多个）
    provider: str = "not_implemented"

    def to_emotion_dict(self) -> dict:
        """转成 emotion 模块能直接消费的格式。"""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "label": self.label,
            "confidence": self.confidence,
            "source": self.provider,
        }

    def to_prompt_line(self) -> str:
        """转成一行可以放进提示词的描述。

        注意：图片本身进不了提示词，只有这段文字描述能进。
        """
        if not self.is_sticker:
            return ""
        parts = ["[用户发了一张表情包]"]
        if self.has_text and self.ocr_text:
            parts.append(f"图中文字：「{self.ocr_text}」")
        if self.description:
            parts.append(f"画面：{self.description}")
        parts.append(f"情绪倾向：{self.label}")
        return " ".join(parts)


# ============================================================ 识别器接口


class StickerRecognizer(Protocol):
    """表情包识别器接口。未来的实现要满足这个协议。

    为什么用 Protocol 而不是抽象基类？
    因为未来可能有多个实现（轻量本地模型 / 多模态大模型 API / 混合），
    只要"长得像"就能插进来，不需要继承什么。
    """

    name: str

    async def is_sticker(self, media: MediaInput) -> tuple[bool, float]:
        """判断是不是表情包。返回 (是否表情包, 置信度)。"""
        ...

    async def extract_text(self, media: MediaInput) -> tuple[str, float]:
        """OCR：提取图内文字。返回 (文字, 置信度)。"""
        ...

    async def analyze_emotion(self, media: MediaInput) -> MediaAnalysis:
        """综合文字 + 面部表情 + 肢体动作，给出情绪判断。"""
        ...


# ============================================================ 占位实现


class NullMediaAnalyzer:
    """空实现：什么都不分析，永远返回"没有图片信息"。

    为什么要有个空实现，而不是干脆不写？
      1. 调用方可以无条件调用它，不用到处判断 None；
      2. 未来替换成真实现时，调用方代码一行都不用改；
      3. 测试可以拿它做基线（"不分析图片时的行为"）。

    这是"预留接口"的标准做法：**先让链路通，再让链路准。**
    """

    name = "null"

    async def analyze(self, media: MediaInput) -> MediaAnalysis:
        """返回一个空结果。注意 is_sticker 有意设为 False —— 
        不确定是不是表情包时，不要假装它是。
        """
        return MediaAnalysis(
            is_sticker=False,
            provider=self.name,
            signals={"note": "多模态分析尚未实现，当前仅支持文字"},
        )

    async def to_emotion_dict(self, media: MediaInput) -> dict | None:
        """给 emotion 模块用的便捷方法。没有图片时返回 None。"""
        if not media.has_image:
            return None
        if not media.image_bytes and not media.image_url:
            return None
        result = await self.analyze(media)
        return result.to_emotion_dict() if result.confidence > 0 else None


def get_media_analyzer() -> NullMediaAnalyzer:
    """获取当前生效的媒体分析器。

    未来这里会读配置决定用哪个实现：
        if settings.media_provider == "qwen-vl":
            return QwenVLMediaAnalyzer()
        return NullMediaAnalyzer()

    现在固定返回空实现，保证链路可用。
    """
    return NullMediaAnalyzer()


# ============================================================ 未来实现路线


IMPLEMENTATION_ROADMAP = """
未来实现路线（按投入从小到大，可任选其一或组合）：

方案 A：多模态大模型 API（推荐先做，最快）
  - 用 qwen-vl-max / GLM-4V 这类模型，一次调用同时完成三件事：
    判断是否表情包、OCR、情绪判断
  - 优点：开发量最小（一个 API 调用），准确率高，无需本地算力
  - 缺点：有成本，且图片要出本地（隐私敏感场景不适用）
  - 提示词要点：要求返回结构化 JSON，含 is_sticker / ocr_text /
    expression(面部) / gesture(肢体) / emotion / confidence

方案 B：本地轻量组合（隐私友好，成本为零）
  - 是否表情包：小分类模型，或用启发式规则
    （带文字 + 分辨率低 + 比例接近 1:1 + 尺寸小 → 大概率是表情包）
  - OCR：PaddleOCR（中文效果好，CPU 可跑）
  - 情绪：面部表情分类模型 + 肢体关键点，或直接用 CLIP 做零样本分类
    （把"开心/难过/生气/无语"当候选标签，看哪个相似度最高）
  - 优点：完全本地、零成本、隐私安全
  - 缺点：开发量大，准确率不如大模型

方案 C：混合（工程上最实际）
  - 先用启发式规则快速筛掉"明显不是表情包"的图（省调用）
  - 剩下的走方案 A 或 B
  - 结果带置信度，低的丢弃

接入时要改的地方（已经在代码里留好位置）：
  1. app/api/schemas.py     ChatRequest 已有 image_url / image_base64 字段
  2. app/memory/models.py   memories 表已有 modality / media_ref / ocr_text 字段
                            + media_assets 表（存图片与识别结果）
  3. app/memory/emotion.py  EmotionScore 已有 source 字段
  4. 本模块                把 NullMediaAnalyzer 换成真实现
  5. app/api/chat.py        组装时把 MediaAnalysis.to_prompt_line() 加进提示词
  6. 检索层                 图片记忆的向量用"OCR 文字 + 画面描述"来编码
                            （图片本身没法直接算语义相似度）

一个容易忽略的点：
  图片记忆的检索必须建立在**文字化描述**之上。
  因为向量检索比的是文本语义，图片要先转成文字描述才能进向量库。
  所以 OCR 和"画面描述"不只是给模型看的，也是给检索用的。
"""
