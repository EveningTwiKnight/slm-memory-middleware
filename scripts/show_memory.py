"""查看某个用户被记住了什么。

这是 Day 3 之后最常用的调试命令 —— 判断"抽取器到底干得怎么样"就看它。

用法：
    .venv\\Scripts\\python.exe -m scripts.show_memory u_1001
    .venv\\Scripts\\python.exe -m scripts.show_memory --all
"""

import asyncio
import sys

sys.path.insert(0, ".")

from sqlalchemy import desc, select  # noqa: E402

from app.memory.database import SessionLocal, init_db  # noqa: E402
from app.memory.models import Memory, ProfileFact  # noqa: E402

TYPE_LABEL = {
    "fact": "事实",
    "preference": "偏好",
    "goal": "目标",
    "event": "经历",
    "emotion": "情绪",
    "other": "其他",
}


def bar(score: float, width: int = 10) -> str:
    """把 0~1 的重要度画成一根小横条，一眼看出高低。"""
    n = max(0, min(width, round(score * width)))
    return "#" * n + "." * (width - n)


async def main() -> int:
    args = sys.argv[1:]
    await init_db()

    async with SessionLocal() as db:
        if not args or args[0] == "--all":
            rows = (
                await db.execute(select(Memory).order_by(desc(Memory.created_at)).limit(50))
            ).scalars().all()
            if not rows:
                print("数据库里还没有任何记忆。")
                print("先跑 tests/test_e2e.py 或 tests.test_extractor 产生一些数据。")
                return 0
            print(f"最近 {len(rows)} 条记忆（所有用户）：\n")
            for m in rows:
                print(
                    f"  [{m.user_id}] {TYPE_LABEL.get(m.type, m.type):<4} "
                    f"重要度 {bar(m.importance)} {m.importance:.2f}  {m.content}"
                )
            return 0

        user_id = args[0]
        memories = (
            await db.execute(
                select(Memory)
                .where(Memory.user_id == user_id)
                .order_by(desc(Memory.importance), desc(Memory.created_at))
            )
        ).scalars().all()
        facts = (
            await db.execute(
                select(ProfileFact).where(
                    ProfileFact.user_id == user_id, ProfileFact.is_active == 1
                )
            )
        ).scalars().all()

        print("=" * 66)
        print(f"用户 {user_id}")
        print("=" * 66)

        print(f"\n【用户档案】{len(facts)} 条")
        if facts:
            for f in facts:
                print(f"  {f.key}：{f.value}")
        else:
            print("  （空）")

        print(f"\n【长期记忆】{len(memories)} 条（按重要度排序）")
        if memories:
            for m in memories:
                emo = ""
                if m.emotion_label:
                    emo = f" 情绪={m.emotion_label}({m.valence:+.2f})"
                print(
                    f"  {bar(m.importance)} {m.importance:.2f} "
                    f"[{TYPE_LABEL.get(m.type, m.type)}]{emo}\n"
                    f"      {m.content}"
                )
        else:
            print("  （空）")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
