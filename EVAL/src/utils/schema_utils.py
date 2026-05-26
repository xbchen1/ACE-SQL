"""
Schema格式化工具
处理数据库Schema的各种格式转换
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Union


# SQL保留字集合
SQL_RESERVED_WORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "ORDER", "BY",
    "HAVING", "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT", "AS",
    "CREATE", "TABLE", "INSERT", "UPDATE", "DELETE", "ALTER", "DROP",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CONSTRAINT", "INDEX",
    "NOT", "NULL", "DEFAULT", "CHECK", "UNIQUE", "AUTO_INCREMENT",
    "INT", "INTEGER", "VARCHAR", "TEXT", "REAL", "BOOLEAN", "DATE",
    "TIMESTAMP", "DECIMAL", "FLOAT", "DOUBLE", "CHAR", "BLOB",
}


def needs_quotes(identifier: str) -> bool:
    """检查标识符是否需要引号"""
    if not identifier:
        return False
    if identifier.upper() in SQL_RESERVED_WORDS:
        return True
    for char in identifier:
        if not (char.isalnum() or char == '_'):
            return True
    return False


def format_identifier(identifier: str) -> str:
    """格式化标识符（添加引号if needed）"""
    if identifier is None:
        return ""
    if needs_quotes(identifier):
        return f'"{identifier}"'
    return identifier


def load_tables_json(path: Union[str, Path]) -> Dict[str, Dict]:
    """加载tables.json文件，返回db_id到信息的映射"""
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tables_map = {}
    for item in data:
        db_id = item.get('db_id')
        if db_id:
            tables_map[db_id] = item

    return tables_map


def extract_column_indices_from_sql(sql: str, db_info: Dict) -> Set[int]:
    """从SQL语句中提取使用的列索引"""
    sql_lower = sql.lower()
    used_indices = set()

    for idx, (table_idx, column_name) in enumerate(db_info.get("column_names_original", [])):
        if table_idx < 0:
            continue

        table_name = db_info["table_names_original"][table_idx]

        # 检查多种格式
        patterns = [
            f"{table_name}.{column_name}",
            f"`{table_name}`.`{column_name}`",
            f'"{table_name}"."{column_name}"',
        ]

        for pattern in patterns:
            if pattern.lower() in sql_lower:
                used_indices.add(idx)
                break

        # 检查单独的列名
        if column_name.lower() in sql_lower:
            column_count = sum(
                1 for _, col in db_info.get("column_names_original", [])
                if col == column_name
            )
            if column_count == 1:
                used_indices.add(idx)

    return used_indices


def normalize_identifier(s: str) -> str:
    """
    去除所有标点符号、空格等非字母数字字符，只保留字母和数字
    用于模糊匹配时的标准化

    Args:
        s: 输入字符串

    Returns:
        只包含字母数字的小写字符串
    """
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()


def select_columns_from_list(db_info: Dict, column_list: List[str], strict_match: bool = True) -> Set[int]:
    """
    从列名列表选择列索引

    匹配逻辑：
    1. 首先尝试精确匹配 (table.column 格式)
    2. 如果精确匹配失败，去除所有标点符号后再次匹配
    3. 如果仍然匹配不上，跳过该项

    Args:
        db_info: 数据库信息
        column_list: 列名列表 (格式: "table.column" 或 "table")
        strict_match: 是否启用严格匹配模式，确保每个预测项都对应schema中的列

    Returns:
        匹配到的列索引集合
    """
    if not db_info or not column_list:
        return set()

    # 构建列名到索引的映射 (原始格式)
    column_map = {}
    # 构建标准化后的列名到原始key的映射 (用于二次匹配)
    normalized_column_map = {}
    for idx, (table_idx, column_name) in enumerate(db_info.get("column_names_original", [])):
        if table_idx < 0:
            continue
        table_name = db_info["table_names_original"][table_idx]
        key = f"{table_name}.{column_name}".lower()
        column_map[key] = idx
        # 标准化后的key映射到原始key
        normalized_key = normalize_identifier(key)
        normalized_column_map[normalized_key] = key

    # 构建表名到所有列索引的映射
    table_map = {}
    # 构建标准化后的表名到原始表名的映射
    normalized_table_map = {}
    for idx, (table_idx, _) in enumerate(db_info.get("column_names_original", [])):
        if table_idx < 0:
            continue
        table_name = db_info["table_names_original"][table_idx].lower()
        if table_name not in table_map:
            table_map[table_name] = []
            normalized_table_map[normalize_identifier(table_name)] = table_name
        table_map[table_name].append(idx)

    selected_indices = set()
    unmatched_items = []

    for item in column_list:
        item_lower = item.strip().lower()
        # 清理引号 - 处理 table."column" 或 table.`column` 格式
        item_lower = item_lower.replace('"', '').replace('`', '').replace("'", '').strip()

        matched = False

        # 第一步：精确匹配
        if item_lower in column_map:
            selected_indices.add(column_map[item_lower])
            matched = True
        elif item_lower in table_map:
            selected_indices.update(table_map[item_lower])
            matched = True
        else:
            # 第二步：去除标点符号后的二次匹配
            normalized_item = normalize_identifier(item_lower)

            if normalized_item in normalized_column_map:
                # 找到匹配，还原回原始schema格式
                original_key = normalized_column_map[normalized_item]
                selected_indices.add(column_map[original_key])
                matched = True
            elif normalized_item in normalized_table_map:
                # 表名匹配
                original_table = normalized_table_map[normalized_item]
                selected_indices.update(table_map[original_table])
                matched = True
            else:
                # 仅在非严格模式下尝试部分匹配
                if not strict_match:
                    for key, idx in column_map.items():
                        if item_lower in key or key in item_lower:
                            selected_indices.add(idx)
                            matched = True
                            break

        if not matched:
            # 匹配不上，跳过该项
            unmatched_items.append(item)

    return selected_indices


def parse_column_list(output: str) -> List[str]:
    """
    从检索器输出中提取列名列表

    兼容以下格式：
    1. ```[table.col, ...]```
    2. 裸列表 [table.col, ...]
    3. JSON: {"columns": ["table.col", ...]}
    4. JSON: {"table": ["col1", "col2"]}
    """
    if not output:
        return []

    stripped_output = output.strip()
    if not stripped_output:
        return []

    # 优先尝试 JSON 格式
    json_candidates = []
    if stripped_output.startswith("{") and stripped_output.endswith("}"):
        json_candidates.append(stripped_output)

    json_block_match = re.search(r'```json\s*\n?(.*?)\n?```', stripped_output, re.DOTALL)
    if json_block_match:
        json_candidates.insert(0, json_block_match.group(1).strip())

    for json_str in json_candidates:
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            if isinstance(parsed.get("columns"), list):
                return [str(col).strip() for col in parsed["columns"] if str(col).strip()]

            flattened_columns = []
            valid_table_map = True
            for table_name, columns in parsed.items():
                if not isinstance(columns, list):
                    valid_table_map = False
                    break
                for column_name in columns:
                    column_name = str(column_name).strip()
                    if column_name:
                        flattened_columns.append(f"{table_name}.{column_name}")
            if valid_table_map and flattened_columns:
                return flattened_columns

    def _parse_list_content(list_content: str) -> List[str]:
        return [
            col.strip().strip('"').strip("'")
            for col in list_content.split(',')
            if col.strip()
        ]

    code_block_match = re.search(
        r'```(?:python|sql|text)?\s*\n?\s*\[([^\]]*)\]\s*\n?```',
        stripped_output,
        re.DOTALL
    )
    if code_block_match:
        return _parse_list_content(code_block_match.group(1))

    if stripped_output.startswith('[') and stripped_output.endswith(']'):
        return _parse_list_content(stripped_output[1:-1])

    list_match = re.search(r'\[([^\[\]]*)\]', stripped_output, re.DOTALL)
    if list_match:
        return _parse_list_content(list_match.group(1))

    return []


def parse_retriever_output(output: str, db_info: Dict, strict_match: bool = True) -> Set[int]:
    """
    解析检索器输出并映射到 schema 列索引
    """
    if not output or not db_info:
        return set()

    columns = parse_column_list(output)
    if not columns:
        return set()

    selected_indices = select_columns_from_list(db_info, columns, strict_match=strict_match)
    if not selected_indices:
        return set()

    return ensure_necessary_columns(db_info, selected_indices)


def ensure_necessary_columns(db_info: Dict, base_indices: Set[int], add_primary_keys: bool = True) -> Set[int]:
    """
    直接返回检索器预测的列，不添加任何额外的主键或外键

    Args:
        db_info: 数据库信息
        base_indices: 基础列索引集合
        add_primary_keys: 保留参数以兼容旧代码，但不再使用

    Returns:
        原始的列索引集合（不做任何修改）
    """
    if not base_indices:
        return {
            idx for idx, (table_idx, _) in enumerate(db_info.get("column_names_original", []))
            if table_idx >= 0
        }

    return set(base_indices)


def build_ddl_schema(
    db_info: Dict,
    column_indices: Optional[Set[int]] = None,
    include_foreign_keys: bool = True
) -> str:
    """构建DDL格式的Schema"""
    if not db_info:
        return ""

    # 确保包含必要的列
    if column_indices is not None:
        column_indices = ensure_necessary_columns(db_info, column_indices)
    else:
        column_indices = {
            idx for idx, (table_idx, _) in enumerate(db_info.get("column_names_original", []))
            if table_idx >= 0
        }

    # 获取主键信息
    primary_keys = set()
    for pk in db_info.get("primary_keys", []):
        if isinstance(pk, int):
            primary_keys.add(pk)
        elif isinstance(pk, (list, tuple)):
            primary_keys.update(pk)

    # 构建DDL
    ddl_tables = []

    for table_idx, table_name in enumerate(db_info.get("table_names_original", [])):
        table_columns = []
        table_pk_columns = []
        table_fk_constraints = []

        for col_idx, (col_table_idx, col_name) in enumerate(db_info.get("column_names_original", [])):
            if col_table_idx != table_idx or col_idx not in column_indices:
                continue

            col_type = db_info.get("column_types", [])[col_idx] if col_idx < len(db_info.get("column_types", [])) else "text"

            col_def = f"    {format_identifier(col_name)} {col_type}"
            table_columns.append(col_def)

            if col_idx in primary_keys:
                table_pk_columns.append(col_name)

            if include_foreign_keys:
                for source_idx, target_idx in db_info.get("foreign_keys", []):
                    if source_idx == col_idx:
                        if target_idx >= len(db_info["column_names_original"]):
                            continue
                        # 只有当目标列也在所选列中时才声明外键
                        if target_idx not in column_indices:
                            continue
                        target_table_idx = db_info["column_names_original"][target_idx][0]
                        target_table_name = db_info["table_names_original"][target_table_idx]
                        target_column_name = db_info["column_names_original"][target_idx][1]

                        constraint_name = f"fk_{table_name.lower().replace(' ', '_')}_{col_name.lower().replace(' ', '_')}"
                        fk_constraint = (
                            f"    CONSTRAINT {constraint_name} "
                            f"FOREIGN KEY ({format_identifier(col_name)}) "
                            f"REFERENCES {format_identifier(target_table_name)} "
                            f"({format_identifier(target_column_name)})"
                        )
                        table_fk_constraints.append(fk_constraint)

        if table_columns:
            ddl = f"CREATE TABLE {format_identifier(table_name)} (\n"
            ddl += ",\n".join(table_columns)

            if table_pk_columns:
                pk_list = ", ".join(format_identifier(pk) for pk in table_pk_columns)
                ddl += f",\n    PRIMARY KEY ({pk_list})"

            if table_fk_constraints:
                ddl += ",\n" + ",\n".join(table_fk_constraints)

            ddl += "\n);"
            ddl_tables.append(ddl)

    return "\n\n".join(ddl_tables)
