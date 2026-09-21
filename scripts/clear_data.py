"""清空数据：把中间件"记住的东西"全部删掉。

两个使用场景：
  1. **准备把仓库给别人用**：你自己跑出来的记忆、档案、情绪、向量索引都还在，
     直接给出去会让别人看到你的数据，而且演示时状态不干净。
  2. **重新演示**：想从头展示"它怎么一点点记住我"，需要先清空。

清空范围（五张表 + 向量库 + 缓存）：
  conversations   会话原文
  memories        结构化记忆
  profile_facts   用户档案
  user_emotions   情绪状态
  media_assets    媒体资源（预留表）
  chroma_data/    向量索引
  进程内缓存      （脚本跑不到，需要重启服务）

用法：
    # 看现在有什么（不删）
    .venv\\Scripts\\python.exe -m scripts.clear_data --stats

    # 清空所有对话数据，但保留知识库
    .venv\\Scripts\\python.exe -m scripts.clear_data --all

    # 只清某个用户
    .venv\\Scripts\\python.exe -m scripts.clear_data --user demo_user

    # 连知识库一起清（如果你往里入库过自己的文档）
    .venv\\Scripts\\python.exe -m scripts.clear_data --all --with-kb

    # 加 --yes 跳过确认
    .venv\\Scripts\\python.exe -m scripts.clear_data --all --yes
"""

import asyncio
import sys

sys.path.insert(0, ".")

from sqlalchemy import delete, func, select  # noqa: E402

from app.memory import long_term, vector_store  # noqa: E402
from app.memory.database import SessionLocal, init_db  # noqa: E402
from app.memory.models import (  # noqa: E402
    Conversation,
    MediaAsset,
    Memory,
    ProfileFact,
    UserEmotion,
)


async def stats() -> dict:
    """统计各类数据的条数与涉及的用户。"""
    out: dict = {}
    async with SessionLocal() as db:
        for model, name in (
            (Conversation, "会话消息"),
            (Memory, "结构化记忆"),
            (ProfileFact, "用户档案"),
            (UserEmotion, "情绪状态"),
            (MediaAsset, "媒体资源"),
        ):
            n = (await db.execute(select(func.count()).select_from(model))).scalar_one()
            out[name] = int(n)

        # 涉及哪些用户
        rows = (await db.execute(select(Conversation.user_id).distinct())).all()
        users = sorted({r[0] for r in rows if r[0]})
        rows2 = (await db.execute(select(Memory.user_id).distinct())).all()
        users = sorted(set(users) | {r[0] for r in rows2 if r[0]})
        out["涉及用户"] = users

    out["向量库"] = vector_store.stats()
    return out


def print_stats(s: dict) -> None:
    print("=" * 62)
    print("当前数据")
    print("=" * 62)
    for k, v in s.items():
        if k == "涉及用户":
            print(f"  {k}：{len(v)} 个" + (f"  {v[:8]}" + ("…" if len(v) > 8 else "") if v else ""))
        elif k == "向量库":
            print(f"  {k}：记忆索引 {v['memories']} 条，知识库 {v['knowledge']} 块")
        else:
            print(f"  {k}：{v} 条")


async def clear_all(*, with_kb: bool = False) -> dict:
    """清空全部对话数据。"""
    removed: dict = {}
    async with SessionLocal() as db:
        for model, name in (
            (Conversation, "会话消息"),
            (Memory, "结构化记忆"),
            (ProfileFact, "用户档案"),
            (UserEmotion, "情绪状态"),
            (MediaAsset, "媒体资源"),
        ):
            r = await db.execute(delete(model))
            removed[name] = r.rowcount or 0
        await db.commit()

    # 向量库要单独清（它和 SQLite 是两套存储，删了 SQLite 不等于删了向量）
    try:
        from app.memory.vector_store import COLLECTION_MEMORIES, get_client

        get_client().delete_collection(COLLECTION_MEMORIES)
        removed["向量索引"] = "已清空"
    except Exception as e:  # noqa: BLE001
        removed["向量索引"] = f"跳过（{e}）"

    if with_kb:
        vector_store.clear_knowledge()
        removed["知识库"] = "已清空"
    else:
        removed["知识库"] = "保留"

    return removed


async def clear_user(user_id: str) -> dict:
    """清空单个用户（含向量与情绪）。"""
    r = await long_term.clear_user(user_id)
    from app.memory.emotion_store import clear_user as clear_emo

    emo = await clear_emo(user_id)

    # 会话原文要单独删（long_term.clear_user 不管这部分）
    async with SessionLocal() as db:
        c = await db.execute(delete(Conversation).where(Conversation.user_id == user_id))
        await db.commit()

    return {
        "结构化记忆": r["memories_deleted"],
        "用户档案": r["facts_deleted"],
        "情绪状态": emo,
        "会话消息": c.rowcount or 0,
    }


async def main() -> int:
    args = sys.argv[1:]
    await init_db()

    s = await stats()

    if "--stats" in args or not args:
        print_stats(s)
        print()
        print("用法：")
        print("  --stats              只看统计（默认）")
        print("  --all                清空所有对话数据（保留知识库）")
        print("  --all --with-kb      连知识库一起清")
        print("  --user <user_id>     只清某个用户")
        print("  --yes                跳过确认")
        return 0

    # ---------- 只清某个用户 ----------
    if "--user" in args:
        idx = args.index("--user")
        if idx + 1 >= len(args):
            print("[FAIL] --user 后面要跟 user_id")
            return 1
        uid = args[idx + 1]
        print(f"即将清空用户 {uid} 的所有数据。")
        if "--yes" not in args:
            ans = input("确认？(y/N) ").strip().lower()
            if ans not in ("y", "yes"):
                print("已取消。")
                return 0
        removed = await clear_user(uid)
        print("\n完成：")
        for k, v in removed.items():
            print(f"  {k}：{v}")
        return 0

    # ---------- 清空全部 ----------
    if "--all" in args:
        with_kb = "--with-kb" in args
        print_stats(s)
        print()
        print("即将清空上面所有对话数据。")
        if with_kb:
            print("⚠️ 同时会清空知识库（你入库的文档索引）")
        else:
            print("知识库会保留。")
        print()
        print("注意：服务若正在运行，它内存里的缓存还留着。")
        print("      清完后建议重启一下服务。")
        print()
        if "--yes" not in args:
            ans = input("确认清空？(y/N) ").strip().lower()
            if ans not in ("y", "yes"):
                print("已取消。")
                return 0

        removed = await clear_all(with_kb=with_kb)
        print("\n完成：")
        for k, v in removed.items():
            print(f"  {k}：{v}")

        after = await stats()
        print()
        print_stats(after)
        print()
        print("[OK] 数据已清空。现在别人 clone 下来看到的是干净状态。")
        return 0

    print("[FAIL] 没听懂参数，用 --help 看看用法")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
