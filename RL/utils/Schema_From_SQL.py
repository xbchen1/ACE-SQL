"""
从 Gold SQL 提取涉及的表和列（traverse_scope 修复版）

使用 sqlglot 的 qualify + traverse_scope 按作用域提取列，
解决 CTE 别名冲突问题。支持两种输入：
  - extract_sql_columns_from_db(db_path, sql)      从 sqlite 文件
  - extract_sql_columns_from_db_info(db_info, sql)  从 tables.json dict
"""

import re
import sqlite3

from sqlglot import parse_one, expressions
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope


# ============== 辅助函数 ==============

def make_safe_name(name: str) -> str:
    """将带特殊字符的名称转换为安全名称（只保留字母数字下划线）"""
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if safe and safe[0].isdigit():
        safe = '_' + safe
    return safe


def preprocess_sql(sql: str, original_to_safe: dict) -> str:
    """将 SQL 中带特殊字符的列名替换为安全名"""
    processed = sql
    # 按长度降序排序，避免短名称先匹配
    for orig, safe in sorted(original_to_safe.items(), key=lambda x: -len(x[0])):
        processed = re.sub(rf'`{re.escape(orig)}`', safe, processed)
        processed = re.sub(rf'"{re.escape(orig)}"', safe, processed)
    return processed


# ============== 映射构建 ==============

def build_column_mappings(db_path: str):
    """
    从 sqlite 数据库构建列名映射

    Returns:
        original_to_safe: dict, 原始列名 -> 安全列名
        safe_to_original_lower: dict, 安全列名(小写) -> 原始列名
        db_schema_safe: dict, 用安全名构建的 schema dict（供 sqlglot 使用）
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall() if t[0] != 'sqlite_sequence']

    original_to_safe = {}
    safe_to_original_lower = {}
    db_schema_safe = {}

    for table in tables:
        cursor.execute(f'PRAGMA table_info(`{table}`);')
        table_info = cursor.fetchall()
        db_schema_safe[table] = {}
        for row in table_info:
            col_name = row[1]
            col_type = row[2] or 'TEXT'
            safe_col = make_safe_name(col_name)
            db_schema_safe[table][safe_col.lower()] = col_type
            original_to_safe[col_name] = safe_col
            safe_to_original_lower[safe_col.lower()] = col_name

    conn.close()
    return original_to_safe, safe_to_original_lower, db_schema_safe


def build_column_mappings_from_db_info(db_info: dict):
    """
    从 tables.json 的 db_info 构建列名映射（当数据库文件不可用时使用）

    Args:
        db_info: tables.json 中的单个数据库信息

    Returns:
        original_to_safe, safe_to_original_lower, db_schema_safe
    """
    original_to_safe = {}
    safe_to_original_lower = {}
    db_schema_safe = {}

    table_names = db_info.get("table_names_original", [])
    column_names = db_info.get("column_names_original", [])
    column_types = db_info.get("column_types", [])

    for table_name in table_names:
        db_schema_safe[table_name] = {}

    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue
        table_name = table_names[table_idx]
        col_type = column_types[col_idx] if col_idx < len(column_types) else 'TEXT'
        safe_col = make_safe_name(col_name)
        db_schema_safe[table_name][safe_col.lower()] = col_type
        original_to_safe[col_name] = safe_col
        safe_to_original_lower[safe_col.lower()] = col_name

    return original_to_safe, safe_to_original_lower, db_schema_safe


# ============== 核心提取逻辑 ==============

def _extract_columns_core(sql, original_to_safe, safe_to_original_lower,
                          db_schema_safe):
    """
    核心列提取逻辑（traverse_scope 版本，修复 CTE 别名冲突）

    Returns:
        dict: {table_name: [column_names]}
    """
    if not sql or not sql.strip() or not db_schema_safe:
        return {}

    try:
        processed_sql = preprocess_sql(sql, original_to_safe)

        try:
            parsed = parse_one(processed_sql, read='sqlite')
            qualified = qualify(
                parsed,
                schema=db_schema_safe,
                qualify_columns=True,
                validate_qualify_columns=False
            )
        except Exception:
            qualified = parse_one(processed_sql, read='sqlite')

        # 按作用域提取别名映射和列（修复 CTE 别名冲突）
        db_tables_lower = {t.lower(): t for t in db_schema_safe.keys()}
        columns_dict = {}

        for scope in traverse_scope(qualified):
            scope_alias_map = {}
            for alias, source in scope.sources.items():
                if isinstance(source, expressions.Table):
                    real_table = db_tables_lower.get(source.name.lower())
                    if real_table:
                        scope_alias_map[alias.lower()] = real_table

            for col in scope.columns:
                table_ref = col.table
                if not table_ref:
                    continue
                original_table = scope_alias_map.get(table_ref.lower())
                if not original_table:
                    original_table = db_tables_lower.get(table_ref.lower())
                if not original_table:
                    continue
                original_col = safe_to_original_lower.get(
                    col.name.lower(), col.name)
                if original_table not in columns_dict:
                    columns_dict[original_table] = []
                if original_col not in columns_dict[original_table]:
                    columns_dict[original_table].append(original_col)

        return columns_dict

    except Exception:
        return {}


# ============== 公开接口 ==============

def extract_sql_columns_from_db(db_path: str, sql: str) -> dict:
    """
    从 sqlite 数据库文件提取 SQL 涉及的表和列

    Args:
        db_path: sqlite 数据库文件路径
        sql: SQL 查询语句

    Returns:
        dict: {table_name: [column_names]}
    """
    original_to_safe, safe_to_original_lower, db_schema_safe = \
        build_column_mappings(db_path)
    return _extract_columns_core(
        sql, original_to_safe, safe_to_original_lower, db_schema_safe
    )


def extract_sql_columns_from_db_info(db_info: dict, sql: str) -> dict:
    """
    从 tables.json 的 db_info 提取 SQL 涉及的表和列

    Args:
        db_info: tables.json 中的单个数据库信息
        sql: SQL 查询语句

    Returns:
        dict: {table_name: [column_names]}
    """
    original_to_safe, safe_to_original_lower, db_schema_safe = \
        build_column_mappings_from_db_info(db_info)
    return _extract_columns_core(
        sql, original_to_safe, safe_to_original_lower, db_schema_safe
    )
