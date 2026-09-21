"""轻量数据库迁移。

解决什么问题？
  SQLAlchemy 的 `create_all()` 只会**建缺失的表**，不会**改已存在的表**。
  所以当你给 models.py 加了一个字段（比如多模态预留的 modality），
  老数据库不会自动获得这一列，一插入就报：

      sqlite3.OperationalError: table memories has no column named modality

  这就是"数据迁移"问题，每个持续迭代的项目都会遇到。

为什么不用 Alembic？
  Alembic 是标准方案，功能完整（版本化、可回滚、支持复杂变更）。
  但它需要额外维护 migrations 目录、生成脚本、按顺序执行。
  在项目早期、表结构还在快速变化时，它的开销大于收益。

本模块的做法：**启动时对比"模型定义"和"实际表结构"，自动补齐缺失的列。**
  - 只做"加列"这一种最简单的变更（绝大多数情况下够用）
  - 加列时带上默认值，并对已有行填充该默认值
  - 不做删列、改类型、改约束（这些有风险，应该人工确认）

局限与边界（必须知道）：
  ⚠️ 这个方案只适合"加法式"变更。遇到删列/改类型/加约束，
     它不会处理，也不该处理 —— 那需要人工决策。
  ⚠️ 生产环境应该换成 Alembic。本模块的定位是"开发期的便利工具"，
     不是"生产级迁移方案"。

面试可以这么讲：
  "我早期用了一个自动补列的轻量方案，因为表结构在快速迭代；
   但它只能处理加列。如果要上生产，我会换成 Alembic 做版本化迁移，
   因为自动迁移对删列和改类型这种破坏性操作无法保证安全。"
"""

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.logging import logger


def _default_clause(column) -> tuple[str, object | None]:
    """根据列定义推出 ALTER TABLE 需要的默认值片段。

    返回 (SQL 片段, 用于回填已有行的 Python 值)
    """
    # 1. 模型上显式声明了默认值
    if column.default is not None and getattr(column.default, "arg", None) is not None:
        arg = column.default.arg
        if callable(arg):  # 比如 default=func.now()
            return "", None
        if isinstance(arg, bool):
            return f" DEFAULT {1 if arg else 0}", None
        if isinstance(arg, (int, float)):
            return f" DEFAULT {arg}", None
        return f" DEFAULT '{arg}'", None

    # 2. 可空列：不需要默认值
    if column.nullable:
        return "", None

    # 3. 非空但没有默认值：给一个类型安全的兜底值
    type_name = column.type.__class__.__name__.upper()
    if "INT" in type_name or "FLOAT" in type_name or "NUMERIC" in type_name:
        return " DEFAULT 0", 0
    if "BOOL" in type_name:
        return " DEFAULT 0", 0
    if "DATETIME" in type_name or "DATE" in type_name:
        return "", None
    return " DEFAULT ''", ""


async def auto_migrate(engine: AsyncEngine) -> list[str]:
    """对比模型与实际表结构，自动补齐缺失的列。返回执行过的变更列表。"""
    changes: list[str] = []

    async with engine.begin() as conn:
        # run_sync 里用同步的 inspector（SQLAlchemy 的异步接口不直接提供）
        def _inspect(sync_conn):
            insp = inspect(sync_conn)
            existing_tables = set(insp.get_table_names())
            result = {}
            for table_name in existing_tables:
                cols = {c["name"]: c for c in insp.get_columns(table_name)}
                result[table_name] = cols
            return result

        actual = await conn.run_sync(_inspect)

        # 从模型元数据里找出所有表
        from app.memory.models import Base

        for table in Base.metadata.sorted_tables:
            if table.name not in actual:
                continue  # 表不存在，create_all 会负责创建

            actual_cols = actual[table.name]
            for column in table.columns:
                if column.name in actual_cols:
                    continue

                default_sql, backfill = _default_clause(column)
                ddl = (
                    f"ALTER TABLE {table.name} "
                    f"ADD COLUMN {column.name} {column.type.compile(engine.dialect)}"
                    f"{default_sql}"
                )
                try:
                    await conn.execute(text(ddl))
                    changes.append(f"{table.name}.{column.name} 已补齐")
                    logger.info(f"[迁移] 已添加列：{table.name}.{column.name}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[迁移] 添加列失败 {table.name}.{column.name}：{e}")
                    continue

                # 如果列是非空且需要回填，把已有行的 NULL 填上
                if backfill is not None:
                    try:
                        await conn.execute(
                            text(
                                f"UPDATE {table.name} SET {column.name} = :v "
                                f"WHERE {column.name} IS NULL"
                            ),
                            {"v": backfill},
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"[迁移] 回填 {table.name}.{column.name} 跳过：{e}")

        # 顺便检查索引（只建缺失的）
        def _indexes(sync_conn):
            insp = inspect(sync_conn)
            out = {}
            for t in Base.metadata.sorted_tables:
                if t.name in actual:
                    out[t.name] = {i["name"] for i in insp.get_indexes(t.name)}
            return out

        existing_idx = await conn.run_sync(_indexes)
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_idx:
                continue
            for index in table.indexes:
                if index.name in existing_idx[table.name]:
                    continue
                try:
                    await conn.run_sync(
                        lambda c, idx=index: idx.create(c)
                    )
                    changes.append(f"索引 {index.name} 已补齐")
                    logger.info(f"[迁移] 已创建索引：{index.name}")
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[迁移] 创建索引失败 {index.name}：{e}")

    if changes:
        logger.info(f"[迁移] 共完成 {len(changes)} 项变更：{changes}")
    else:
        logger.debug("[迁移] 表结构与模型一致，无需变更")
    return changes
