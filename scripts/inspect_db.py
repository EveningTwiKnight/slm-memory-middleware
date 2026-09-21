"""打印数据库里现有的表结构。排查"表建歪了没有"用。

用法：.venv\\Scripts\\python.exe -m scripts.inspect_db
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, ".")

from app.core.config import settings  # noqa: E402


def main() -> int:
    db = settings.db_file
    if not Path(db).exists():
        print(f"数据库文件还不存在：{db}")
        print("先启动一次服务（uvicorn app.main:app），或跑一次 tests/test_memory.py")
        return 1

    conn = sqlite3.connect(str(db))
    print(f"数据库：{db}\n")

    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    tables = [r[0] for r in rows if not r[0].startswith("sqlite_")]

    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"表 {t}  （{n} 行）")
        for c in conn.execute(f"PRAGMA table_info({t})").fetchall():
            # c = (cid, name, type, notnull, dflt_value, pk)
            flag = " PK" if c[5] else ""
            print(f"    {c[1]:<18} {c[2]:<12}{flag}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
