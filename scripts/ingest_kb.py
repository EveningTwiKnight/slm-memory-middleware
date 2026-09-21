"""知识库入库：把文档切块、向量化、存进向量库。

为什么必须切块，不能整篇存进去？
  1. 模型的上下文有限，整篇文章塞不进去；
  2. 检索精度会暴跌 —— 一整篇 5000 字的文章只有一小段是相关的，
     但整篇都会作为一个检索结果返回，等于往提示词里灌水；
  3. 切块后每块只讲一件事，检索时能精准命中。

切块参数怎么定？
  块大小 500 字：太小会割裂上下文（一个观点被切成两半），太大则精度下降。
  重叠 50 字：防止关键句刚好被切在边界上，两边都读不全。
  这是工程上的经验值，不是理论最优 —— 面试可以这么说。

用法：
    # 入库单个文件
    .venv\\Scripts\\python.exe -m scripts.ingest_kb docs/我的笔记.md

    # 入库整个目录
    .venv\\Scripts\\python.exe -m scripts.ingest_kb --dir my_docs

    # 先清空再入库（重建索引）
    .venv\\Scripts\\python.exe -m scripts.ingest_kb --dir my_docs --reset

    # 看当前知识库状态
    .venv\\Scripts\\python.exe -m scripts.ingest_kb --stats
"""

import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, ".")

from app.memory import embeddings, vector_store  # noqa: E402

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TEXT_SUFFIXES = {".md", ".txt", ".markdown"}

# 优先在这些位置断开，让每块尽量是完整句子
BREAK_CHARS = "\n。！？；.!?;"


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把长文本切成有重叠的块。

    切块的目标是：**每块是一个能独立读懂的完整意思**。

    策略（按优先级）：
      1. 优先在标题（## / #）处切 —— markdown 文档的标题本身就是天然的语义边界；
      2. 其次在空行（段落边界）处切 —— 一段话讲一件事；
      3. 再其次在句末标点处切；
      4. 都没有才硬切。

    为什么必须这么做？
    早期版本简单地在"距离目标长度最近的句末标点"处切，结果切出了以
    "额。" 开头的块 —— 因为它落在一句话的中间。这种块被检索出来注入提示词，
    模型读到的是半句话，效果还不如不注入。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    # 重叠部分最多向前对齐这么多个字符，用来找句子开头
    snap_window = max(size // 4, 30)

    def find_break(lo: int, hi: int) -> int:
        """在 [lo, hi) 区间里找一个最佳切点，找不到返回 -1。

        从后往前找，返回的是"切点位置"（该位置之前的属于当前块）。
        """
        segment = text[lo:hi]

        # 1. 找 markdown 标题（行首的 #）
        for i in range(len(segment) - 1, -1, -1):
            if segment[i] == "#" and (i == 0 or segment[i - 1] == "\n"):
                return lo + i

        # 2. 找段落边界（空行）
        idx = segment.rfind("\n\n")
        if idx >= 0:
            return lo + idx + 1

        # 3. 找句末标点
        for i in range(len(segment) - 1, -1, -1):
            if segment[i] in BREAK_CHARS:
                return lo + i + 1

        return -1

    def snap_to_sentence(pos: int) -> int:
        """把块起点向后对齐到句子开头。

        为什么需要这一步？
        重叠部分是靠"往回退 50 个字"硬算出来的，会落在句子中间。
        结果就是块以"；换一个模型服务商就要…"这种半句话开头，
        注入提示词后模型读到的是残句，效果还不如不注入。

        做法：在 pos 之后找第一个句末标点，从它后面开始。
        找不到就保持原位（宁可重叠少一点，也要块是完整的）。
        """
        if pos <= 0 or pos >= n:
            return pos
        # 已经在句首就不用动
        if text[pos - 1] in BREAK_CHARS or text[pos - 1] == "\n":
            return pos
        limit = min(pos + snap_window, n)
        for i in range(pos, limit):
            if text[i] in BREAK_CHARS:
                return i + 1
            if text[i] == "\n":
                return i + 1
        return pos

    while start < n:
        end = min(start + size, n)

        if end < n:
            # 只在块的后半段找切点，避免把块切得太短
            lo = max(start + size // 2, start)
            pos = find_break(lo, end)
            if pos > start:
                end = pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = snap_to_sentence(max(end - overlap, start + 1))

    return chunks


def doc_id_for(source: str, index: int, content: str) -> str:
    """给每个块生成稳定 ID。

    为什么用内容哈希而不是序号？
    重新入库同一个文件时，内容没变的块 ID 不变，Chroma 的 upsert 会覆盖而不是新增，
    这样就不会产生重复。内容变了才产生新 ID。
    """
    h = hashlib.md5(content.encode("utf-8")).hexdigest()[:10]
    return f"kb_{Path(source).stem}_{index}_{h}"


async def ingest_file(path: Path, *, verbose: bool = True) -> int:
    """入库一个文件，返回切出的块数。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = split_text(text)
    if not chunks:
        print(f"  跳过（无内容）：{path}")
        return 0

    # 批量向量化比逐条快很多
    vectors = embeddings.embed_documents(chunks)

    for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=False)):
        vector_store.add_knowledge(
            doc_id_for(str(path), i, chunk),
            chunk,
            vec,
            source=path.name,
            chunk_index=i,
        )

    if verbose:
        print(f"  {path.name}：{len(text)} 字 → {len(chunks)} 块")
        if chunks:
            print(f"    首块预览：{chunks[0][:60]}…")
    return len(chunks)


def main_sync() -> int:
    """本地自检：只测切块逻辑，不加载模型、不联网。

    用法：.venv\\Scripts\\python.exe -m scripts.ingest_kb --selftest
    """
    sample = (
        "# 中间件设计笔记\n\n"
        "## 为什么需要中间件\n\n"
        "大模型应用开发中，业务代码直接调用模型会很快失控。原因是：\n"
        "每个业务模块都要自己拼提示词，格式不统一，改一次要改十处；"
        "用户记忆散落在各处，无法跨会话复用；换一个模型服务商就要改遍所有调用点。\n\n"
        "## 分层设计\n\n"
        "中间件分为六层。接入层负责协议与鉴权；状态层管理短期记忆与用户档案；"
        "检索层做混合召回与重排序；编排层负责提示词拼装与预算调度。\n\n"
        "## Token 预算调度\n\n"
        "小模型上下文窗口有限，必须给每个段落分配配额。"
        "超出预算时的裁剪优先级为：先砍最早的对话历史，再砍低分记忆，人设与输入永不裁剪。\n"
    )
    chunks = split_text(sample, size=120, overlap=20)
    print(f"原文 {len(sample)} 字 → {len(chunks)} 块（size=120, overlap=20）\n")
    ok = True
    for i, c in enumerate(chunks, 1):
        head = c.split("\n")[0][:40]
        print(f"  [{i}] {len(c)} 字 | 开头：{head!r}")
        # 关键断言：不能以标点或半句话开头
        first = c.lstrip()[0]
        if first in BREAK_CHARS or first in "，。、）】":
            print("      ↑ 问题：块以标点/断句开头")
            ok = False
        if len(c) > 120 + 60:
            print("      ↑ 问题：块明显超长")
            ok = False
    print()
    if ok:
        print("[OK] 切块自检通过：每块都以完整内容开头，没有切在半句话中间")
        return 0
    print("[FAIL] 切块自检失败")
    return 1


async def main() -> int:
    args = sys.argv[1:]

    if "--selftest" in args:
        return main_sync()

    if "--stats" in args or not args:
        s = vector_store.stats()
        print("=" * 60)
        print("向量库状态")
        print("=" * 60)
        print(f"  记忆索引：{s['memories']} 条")
        print(f"  知识库：  {s['knowledge']} 块")
        print("\n用法：")
        print("  .venv\\Scripts\\python.exe -m scripts.ingest_kb <文件路径>")
        print("  .venv\\Scripts\\python.exe -m scripts.ingest_kb --dir <目录> [--reset]")
        return 0

    if "--reset" in args:
        print("清空知识库…")
        vector_store.clear_knowledge()

    targets: list[Path] = []
    if "--dir" in args:
        d = Path(args[args.index("--dir") + 1])
        if not d.is_dir():
            print(f"[FAIL] 目录不存在：{d}")
            return 1
        targets = [
            p for p in sorted(d.rglob("*")) if p.suffix.lower() in TEXT_SUFFIXES and p.is_file()
        ]
    else:
        for a in args:
            if a.startswith("--"):
                continue
            p = Path(a)
            if p.is_file():
                targets.append(p)
            else:
                print(f"[FAIL] 文件不存在：{p}")
                return 1

    if not targets:
        print("[FAIL] 没有找到可入库的文件（支持 .md / .txt）")
        return 1

    print("=" * 60)
    print(f"开始入库 {len(targets)} 个文件（块大小 {CHUNK_SIZE} 字，重叠 {CHUNK_OVERLAP} 字）")
    print("=" * 60)

    total = 0
    for p in targets:
        total += await ingest_file(p)

    print("=" * 60)
    print(f"完成：共 {total} 块")
    print(f"知识库现有：{vector_store.stats()['knowledge']} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
