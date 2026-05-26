"""
Generate prompts using GenLink's SQL column extraction method.
Uses sqlglot to extract columns from gold SQL, then adds PK/FK info.

改进：通过预处理-提取-还原的方式处理带特殊字符的列名
"""

import json
import os
import sys
import re
import sqlite3
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import OrderedDict

from sqlglot import parse_one, expressions
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from prepare_eval_data import (
    sample_table_values,
    retrieve_relevant_hits,
    retrieve_question_related_db_values,
    obtain_n_grams,
    deduplicate_dicts,
    format_identifier
)
from pyserini.search.lucene import LuceneSearcher


# ============== 改进的列提取方法（支持特殊字符列名） ==============

def make_safe_name(name: str) -> str:
    """将带特殊字符的名称转换为安全名称（只保留字母数字下划线）"""
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if safe and safe[0].isdigit():
        safe = '_' + safe
    return safe


def build_column_mappings(db_path: str):
    """
    从数据库构建列名映射

    Returns:
        original_to_safe: dict, 原始列名 -> 安全列名
        safe_to_original_lower: dict, 安全列名(小写) -> 原始列名
        db_schema_safe: dict, 用安全名构建的 schema dict（供 sqlglot 使用）
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall() if t[0] != 'sqlite_sequence']

    original_to_safe = {}       # 原始名 -> 安全名
    safe_to_original_lower = {} # 安全名(小写) -> 原始名
    db_schema_safe = {}         # 用安全名构建的 schema dict

    for table in tables:
        cursor.execute(f'PRAGMA table_info(`{table}`);')
        table_info = cursor.fetchall()

        db_schema_safe[table] = {}
        for row in table_info:
            col_name = row[1]
            col_type = row[2] or 'TEXT'
            safe_col = make_safe_name(col_name)

            # 构建安全名 schema（小写 key）
            db_schema_safe[table][safe_col.lower()] = col_type

            # 记录映射
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
        original_to_safe: dict, 原始列名 -> 安全列名
        safe_to_original_lower: dict, 安全列名(小写) -> 原始列名
        db_schema_safe: dict, 用安全名构建的 schema dict（供 sqlglot 使用）
    """
    original_to_safe = {}       # 原始名 -> 安全名
    safe_to_original_lower = {} # 安全名(小写) -> 原始名
    db_schema_safe = {}         # 用安全名构建的 schema dict

    table_names = db_info.get("table_names_original", [])
    column_names = db_info.get("column_names_original", [])
    column_types = db_info.get("column_types", [])

    # 初始化每个表的 schema
    for table_name in table_names:
        db_schema_safe[table_name] = {}

    # 遍历所有列
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:  # 跳过 * 列
            continue

        table_name = table_names[table_idx]
        col_type = column_types[col_idx] if col_idx < len(column_types) else 'TEXT'
        safe_col = make_safe_name(col_name)

        # 构建安全名 schema（小写 key）
        db_schema_safe[table_name][safe_col.lower()] = col_type

        # 记录映射
        original_to_safe[col_name] = safe_col
        safe_to_original_lower[safe_col.lower()] = col_name

    return original_to_safe, safe_to_original_lower, db_schema_safe


def extract_sql_columns_with_db_info(db_info: dict, sql: str) -> dict:
    """
    使用 db_info (tables.json) 提取 SQL 中的列（当数据库文件不可用时使用）

    通过预处理-提取-还原的方式处理带特殊字符的列名：
    1. 从 db_info 构建列名映射
    2. 预处理：将 SQL 中的特殊列名替换为安全名
    3. 提取：使用 sqlglot 的 qualify + find_all 提取列
    4. 还原：将安全名映射回原始列名，将表别名映射回原始表名

    Args:
        db_info: tables.json 中的单个数据库信息
        sql: SQL 查询语句

    Returns:
        dict: {table_name: [column_names]}
    """
    if not sql or not sql.strip():
        return {}

    try:
        # 1. 从 db_info 构建映射
        original_to_safe, safe_to_original_lower, db_schema_safe = build_column_mappings_from_db_info(db_info)

        # 如果 schema 为空，返回空
        if not db_schema_safe:
            return {}

        # 2. 预处理 SQL
        processed_sql = preprocess_sql(sql, original_to_safe)

        # 3. 使用 sqlglot 解析
        try:
            parsed = parse_one(processed_sql, read='sqlite')
            qualified = qualify(
                parsed,
                schema=db_schema_safe,
                qualify_columns=True,
                validate_qualify_columns=False
            )
        except Exception:
            # qualify 失败时，尝试直接解析
            qualified = parse_one(processed_sql, read='sqlite')

        # 4-5. 按作用域提取别名映射和列（修复 CTE 别名冲突）
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

    except Exception as e:
        # 出错时返回空
        return {}


def preprocess_sql(sql: str, original_to_safe: dict) -> str:
    """将 SQL 中带特殊字符的列名替换为安全名"""
    processed = sql
    # 按长度降序排序，避免短名称先匹配
    for orig, safe in sorted(original_to_safe.items(), key=lambda x: -len(x[0])):
        # 匹配 `原始名` 或 "原始名"
        processed = re.sub(rf'`{re.escape(orig)}`', safe, processed)
        processed = re.sub(rf'"{re.escape(orig)}"', safe, processed)
    return processed


def _extract_columns_core(sql: str, original_to_safe: dict, safe_to_original_lower: dict, db_schema_safe: dict) -> dict:
    """
    核心列提取逻辑（供 extract_sql_columns_improved 和 extract_sql_columns_with_db_info 共用）

    Args:
        sql: SQL 查询语句
        original_to_safe: 原始列名 -> 安全列名 映射
        safe_to_original_lower: 安全列名(小写) -> 原始列名 映射
        db_schema_safe: 用安全名构建的 schema dict

    Returns:
        dict: {table_name: [column_names]}
    """
    if not sql or not sql.strip():
        return {}

    try:
        # 1. 预处理 SQL
        processed_sql = preprocess_sql(sql, original_to_safe)

        # 2. 使用 sqlglot 解析
        try:
            parsed = parse_one(processed_sql, read='sqlite')
            qualified = qualify(
                parsed,
                schema=db_schema_safe,
                qualify_columns=True,
                validate_qualify_columns=False
            )
        except Exception:
            # qualify 失败时，尝试直接解析
            qualified = parse_one(processed_sql, read='sqlite')

        # 3. 按作用域提取别名映射和列（修复 CTE 别名冲突）
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

    except Exception as e:
        return {}


def extract_sql_columns_improved(db_path: str, sql: str) -> dict:
    """
    改进的 SQL 列提取方法

    通过预处理-提取-还原的方式处理带特殊字符的列名：
    1. 预处理：将 SQL 中的特殊列名替换为安全名
    2. 提取：使用 sqlglot 的 qualify + find_all 提取列
    3. 还原：将安全名映射回原始列名，将表别名映射回原始表名

    Args:
        db_path: 数据库文件路径
        sql: SQL 查询语句

    Returns:
        dict: {table_name: [column_names]}
    """
    if not sql or not sql.strip():
        return {}

    try:
        # 1. 构建映射
        original_to_safe, safe_to_original_lower, db_schema_safe = build_column_mappings(db_path)

        # 2. 预处理 SQL
        processed_sql = preprocess_sql(sql, original_to_safe)

        # 3. 使用 sqlglot 解析
        try:
            parsed = parse_one(processed_sql, read='sqlite')
            qualified = qualify(
                parsed,
                schema=db_schema_safe,
                qualify_columns=True,
                validate_qualify_columns=False
            )
        except Exception:
            # qualify 失败时，尝试直接解析
            qualified = parse_one(processed_sql, read='sqlite')

        # 4-5. 按作用域提取别名映射和列（修复 CTE 别名冲突）
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

    except Exception as e:
        # 出错时返回空
        return {}


# ============== Prompt Template (same as original) ==============

INPUT_PROMPT_TEMPLATE = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{db_engine}

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
'''


def extract_columns_with_genlink(db_path: str, sql: str):
    """
    使用改进的方法从 SQL 中提取列（支持特殊字符列名）

    Returns:
        - extracted_schema: dict {table_name: [col_names]}
        - used_tables: set of table names used in SQL
    """
    try:
        extracted_schema = extract_sql_columns_improved(db_path, sql)
        used_tables = set(extracted_schema.keys())
        return extracted_schema, used_tables
    except Exception as e:
        print(f"Column extraction failed: {e}")
        return {}, set()


def get_column_index(db_info: dict, table_name: str, column_name: str) -> int:
    """Get column index in db_info given table and column name."""
    table_name_lower = table_name.lower()
    column_name_lower = column_name.lower()

    for col_idx, (table_idx, col_name) in enumerate(db_info["column_names_original"]):
        if table_idx >= 0:
            t_name = db_info["table_names_original"][table_idx]
            if t_name.lower() == table_name_lower and col_name.lower() == column_name_lower:
                return col_idx
    return -1


def get_table_index(db_info: dict, table_name: str) -> int:
    """Get table index in db_info given table name."""
    table_name_lower = table_name.lower()
    for idx, t_name in enumerate(db_info["table_names_original"]):
        if t_name.lower() == table_name_lower:
            return idx
    return -1


def obtain_db_details_genlink(db_info: dict, sampled_db_values_dict: dict,
                               relavant_db_values_dict: dict,
                               extracted_schema: dict, used_tables: set):
    """
    Generate DDL-style schema using GenLink extracted columns.

    Key features:
    1. Only include columns extracted by GenLink
    2. Add PRIMARY KEY for used tables (even if PK column not in extracted columns)
    3. Add FOREIGN KEY only if BOTH source and target columns are in extracted columns
    4. Use both relavant_db_values (from Lucene) and sampled_db_values for examples
    """
    db_details = []

    # Build column index set for extracted columns
    extracted_col_indices = set()
    for table_name, columns in extracted_schema.items():
        for col_name in columns:
            col_idx = get_column_index(db_info, table_name, col_name)
            if col_idx >= 0:
                extracted_col_indices.add(col_idx)

    # Add primary key columns for used tables (even if not extracted)
    pk_col_indices_to_add = set()
    for table_name in used_tables:
        table_idx = get_table_index(db_info, table_name)
        if table_idx < 0:
            continue

        for pk_idx in db_info["primary_keys"]:
            if isinstance(pk_idx, int):
                pk_table_idx = db_info["column_names_original"][pk_idx][0]
                if pk_table_idx == table_idx:
                    pk_col_indices_to_add.add(pk_idx)
            elif isinstance(pk_idx, list):
                for p_idx in pk_idx:
                    pk_table_idx = db_info["column_names_original"][p_idx][0]
                    if pk_table_idx == table_idx:
                        pk_col_indices_to_add.add(p_idx)

    # Merge extracted columns with PK columns
    all_used_col_indices = extracted_col_indices | pk_col_indices_to_add

    # Process each table
    for outer_table_idx, table_name in enumerate(db_info["table_names_original"]):
        # Skip tables not used
        if table_name not in used_tables and table_name.lower() not in [t.lower() for t in used_tables]:
            continue

        column_info_list = []
        pk_columns = []
        fk_info = []

        for column_idx, ((inner_table_idx, column_name), (_, column_comment), column_type) in enumerate(zip(
            db_info["column_names_original"], db_info["column_names"], db_info["column_types"]
        )):
            if inner_table_idx != outer_table_idx:
                continue

            if column_idx not in all_used_col_indices:
                continue

            # Get sample values (same logic as original: relavant first, then sampled)
            column_values = []
            tc_key = f"{table_name}.{column_name}".lower()
            if tc_key in relavant_db_values_dict:
                column_values.extend(relavant_db_values_dict[tc_key])
            if tc_key in sampled_db_values_dict:
                column_values.extend(sampled_db_values_dict[tc_key])
            column_values = list(dict.fromkeys(column_values))[:6]  # dedup and limit

            # Format column info
            if column_name.lower() in [column_comment.lower(), column_comment.lower().replace(" ", "_"), column_comment.lower().replace(" ", "")] \
                or column_comment.strip() == "":
                column_info = f'    {format_identifier(column_name)} {column_type},'
                if len(column_values) > 0:
                    column_info += f" -- example: {column_values}"
            else:
                column_info = f'    {format_identifier(column_name)} {column_type}, -- {column_comment}'
                if len(column_values) > 0:
                    column_info += f", example: {column_values}"

            column_info_list.append(column_info)

            # Check if this column is a primary key
            for pk_idx in db_info["primary_keys"]:
                if isinstance(pk_idx, int) and column_idx == pk_idx:
                    pk_columns.append(column_name)
                elif isinstance(pk_idx, list) and column_idx in pk_idx:
                    pk_columns.append(column_name)

            # Check foreign keys - only add if BOTH source and target are in used columns
            for (source_column_idx, target_column_idx) in db_info["foreign_keys"]:
                if column_idx == source_column_idx:
                    # Check if target column is also in our used columns
                    if target_column_idx in all_used_col_indices:
                        source_table_idx = db_info["column_names_original"][source_column_idx][0]
                        source_table_name = db_info["table_names_original"][source_table_idx]
                        source_column_name = db_info["column_names_original"][source_column_idx][1]
                        target_table_idx = db_info["column_names_original"][target_column_idx][0]
                        target_table_name = db_info["table_names_original"][target_table_idx]
                        target_column_name = db_info["column_names_original"][target_column_idx][1]
                        fk_info.append(
                            f'    FOREIGN KEY ({format_identifier(source_column_name)}) '
                            f'REFERENCES {format_identifier(target_table_name)} ({format_identifier(target_column_name)}),'
                        )

        if len(column_info_list) > 0:
            pk_columns = list(OrderedDict.fromkeys(pk_columns))
            if len(pk_columns) > 0:
                pk_info = ['    PRIMARY KEY (' + ', '.join([f'{format_identifier(col)}' for col in pk_columns]) + '),']
            else:
                pk_info = []
            fk_info = list(OrderedDict.fromkeys(fk_info))

            table_ddl = f'CREATE TABLE {format_identifier(table_name)} (\n'
            table_ddl += "\n".join(column_info_list + pk_info + fk_info)
            if table_ddl.endswith(","):
                table_ddl = table_ddl[:-1]
            table_ddl += "\n);"

            db_details.append(table_ddl)

    return "\n\n".join(db_details)


def process_dataset(args):
    print(f"Loading dataset from {args.input_data_file}")
    with open(args.input_data_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Loading tables from {args.tables}")
    with open(args.tables, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    # Build db_id -> db_info mapping
    used_db_ids = list(set([data["db_id"] for data in dataset]))
    db_id2db_info = {}
    for db_info in tables_data:
        if db_info["db_id"] in used_db_ids:
            db_id2db_info[db_info["db_id"]] = db_info

    # Sample db values
    print("Sampling database values...")
    db_id2sampled_db_values = {}
    for db_id in tqdm(used_db_ids):
        db_file = os.path.join(args.db_path, db_id, db_id + ".sqlite")
        if os.path.exists(db_file) and db_id in db_id2db_info:
            sampled_values = sample_table_values(
                db_file,
                db_id2db_info[db_id]["table_names_original"],
                args.value_limit_num
            )
            db_id2sampled_db_values[db_id] = sampled_values
        else:
            db_id2sampled_db_values[db_id] = {}

    # Load db content index if available (for relevant value retrieval)
    db_id2searcher = {}
    if args.db_content_index_path and os.path.exists(args.db_content_index_path):
        print("Loading db content index...")
        for db_id in tqdm(used_db_ids):
            index_path = os.path.join(args.db_content_index_path, db_id)
            if os.path.exists(index_path):
                db_id2searcher[db_id] = LuceneSearcher(index_path)

    # Process each data point
    print("Generating prompts using GenLink column extraction...")
    new_dataset = []
    extraction_stats = {"success": 0, "fallback": 0}

    for data_idx, data in enumerate(tqdm(dataset)):
        db_id = data["db_id"]
        db_info = db_id2db_info.get(db_id)

        if db_info is None:
            print(f"Warning: db_info not found for {db_id}")
            continue

        # Get question with evidence
        evidence = data.get("evidence", "")
        if evidence.strip():
            question = evidence + "\n" + data["question"]
        else:
            question = data["question"]

        # Get relevant db values using Lucene index (same as original)
        relavant_db_values_dict = {}
        if db_id in db_id2searcher:
            queries = obtain_n_grams(question, 8) + [question]
            queries = list(dict.fromkeys(queries))
            query2hits = retrieve_relevant_hits(db_id2searcher[db_id], queries)
            hits = []
            for q in queries:
                hits.extend(query2hits.get(q, []))
            hits = deduplicate_dicts(hits)
            relavant_db_values_dict = retrieve_question_related_db_values(hits, question)

        sampled_db_values = db_id2sampled_db_values.get(db_id, {})
        gold_sql = data.get("SQL", "")

        # Use GenLink to extract columns
        db_path = os.path.join(args.db_path, db_id, db_id + ".sqlite")
        extracted_schema, used_tables = extract_columns_with_genlink(db_path, gold_sql)

        if extracted_schema:
            extraction_stats["success"] += 1
            db_details = obtain_db_details_genlink(
                db_info, sampled_db_values, relavant_db_values_dict,
                extracted_schema, used_tables
            )
        else:
            # Fallback to full schema if extraction fails
            extraction_stats["fallback"] += 1
            print(f"Warning: GenLink extraction failed for idx={data_idx}, using full schema")
            # Use all tables and columns as fallback
            all_tables = set(db_info["table_names_original"])
            all_schema = {t: [] for t in all_tables}
            for col_idx, (table_idx, col_name) in enumerate(db_info["column_names_original"]):
                if table_idx >= 0:
                    t_name = db_info["table_names_original"][table_idx]
                    all_schema[t_name].append(col_name)
            db_details = obtain_db_details_genlink(
                db_info, sampled_db_values, relavant_db_values_dict,
                all_schema, all_tables
            )

        # Generate prompt
        input_seq = INPUT_PROMPT_TEMPLATE.format(
            db_engine="SQLite",
            db_details=db_details,
            question=question
        )

        new_data = {
            "question_id": data.get("question_id", data_idx),
            "db_id": db_id,
            "question": data["question"],
            "evidence": evidence,
            "SQL": gold_sql,
            "difficulty": data.get("difficulty", ""),
            "input_seq": input_seq,
            "output_seq": gold_sql,
            "extracted_schema": extracted_schema  # Keep for debugging
        }
        new_dataset.append(new_data)

    # Save output
    print(f"\nExtraction stats: {extraction_stats}")
    print(f"Saving {len(new_dataset)} prompts to {args.output_data_file}")
    with open(args.output_data_file, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, indent=2, ensure_ascii=False, fp=f)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, required=True,
                        help="Input dev/test data file (e.g., dev.json)")
    parser.add_argument("--output_data_file", type=str, required=True,
                        help="Output prompts file")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to database directory")
    parser.add_argument("--tables", type=str, required=True,
                        help="Path to tables.json")
    parser.add_argument("--value_limit_num", type=int, default=2,
                        help="Number of sample values per column")
    parser.add_argument("--db_content_index_path", type=str, default="",
                        help="Path to db content index for value retrieval")

    args = parser.parse_args()
    process_dataset(args)
