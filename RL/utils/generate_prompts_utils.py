"""
Extracted utility functions from generate_prompts.py for schema generation and prompt building.
"""

import json
import sqlite3
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path


SQL_RESERVED_WORDS = {
    'IDENTIFIED', 'FOREIGN', 'CONSTRAINT', 'USER', 'POSITION', 'DESCRIBE', 'CHECK',
    'RECURSIVE', 'REAL', 'CONTINUE', 'GLOBAL', 'RLIKE', 'INSENSITIVE', 'BOOLEAN',
    'CHAR', 'ROLE', 'CASE', 'SCHEMA', 'CLOB', 'RESIGNAL', 'ROW', 'DEC', 'TOP',
    'EXCEPT', 'SENSITIVE', 'OUT', 'RENAME', 'READS', 'BLOB', 'INT', 'EXTERNAL',
    'LOCALTIMESTAMP', 'DECLARE', 'DO', 'AS', 'OVER', 'CONDITION', 'SELECT',
    'SAVEPOINT', 'WITHIN', 'ELSEIF', 'UNLOCK', 'DATABASE', 'TRIGGER', 'ACCESS',
    'FALSE', 'BREAK', 'ITERATE', 'SMALLINT', 'ASC', 'YEAR', 'DELETE', 'ROLLBACK',
    'ON', 'ESCAPE', 'CREATE', 'MONTH', 'SPECIFIC', 'SESSION', 'SQLSTATE', 'HOLD',
    'SET', 'EXPLAIN', 'RETURN', 'ROWNUM', 'BINARY', 'SYSDATE', 'SQLWARNING',
    'EXTEND', 'CAST', 'FOR', 'TERMINATED', 'VIEW', 'TRAILING', 'HOUR', 'VARYING',
    'RESTRICT', 'RIGHT', 'DISTINCT', 'JOIN', 'UNKNOWN', 'VALUES', 'TABLE', 'OR',
    'DOUBLE', 'DROP', 'COMMIT', 'PRECISION', 'LANGUAGE', 'START', 'INTERSECT',
    'IGNORE', 'NULL', 'CURRENT_DATE', 'LOCK', 'INTO', 'NEW', 'DESC', 'STATIC',
    'MODIFIES', 'GRANT', 'VALUE', 'LIMIT', 'MODULE', 'DATE', 'LOCALTIME',
    'PERCENT', 'REPEAT', 'FULL', 'USAGE', 'ORDER', 'WHEN', 'PRIMARY', 'BETWEEN',
    'CURSOR', 'DECIMAL', 'HAVING', 'IF', 'FILTER', 'INDEX', 'ILIKE', 'VARCHAR',
    'EXEC', 'USING', 'ROWS', 'PLACING', 'WHILE', 'EXECUTE', 'EACH', 'LEFT',
    'FLOAT', 'COLLATE', 'CURRENT_TIME', 'OPEN', 'RANGE', 'CROSS', 'FUNCTION',
    'TIME', 'BOTH', 'NOT', 'CONVERT', 'NCHAR', 'KEY', 'DEFAULT', 'LIKE',
    'ANALYZE', 'EXISTS', 'IN', 'BIT', 'INOUT', 'SUM', 'NUMERIC', 'AFTER',
    'LEAVE', 'INSERT', 'TO', 'COUNT', 'THEN', 'BEFORE', 'OUTER', 'COLUMN',
    'ONLY', 'END', 'PROCEDURE', 'OFFSET', 'ADD', 'INNER', 'RELEASE', 'FROM',
    'DAY', 'NO', 'CALL', 'BY', 'LOCAL', 'ZONE', 'TRUE', 'EXIT', 'LEADING',
    'INTEGER', 'MERGE', 'OLD', 'AVG', 'MIN', 'SQL', 'LOOP', 'SIGNAL',
    'REFERENCES', 'MINUTE', 'UNIQUE', 'GENERATED', 'ALL', 'MATCH', 'CASCADE',
    'UNION', 'COMMENT', 'FETCH', 'UNDO', 'UPDATE', 'WHERE', 'ELSE', 'PARTITION',
    'BIGINT', 'CHARACTER', 'CURRENT_TIMESTAMP', 'ALTER', 'INTERVAL', 'REVOKE',
    'CONNECT', 'WITH', 'TIMESTAMP', 'GROUP', 'BEGIN', 'CURRENT', 'REGEXP',
    'NATURAL', 'SOME', 'SQLEXCEPTION', 'MAX', 'SUBSTRING', 'OF', 'AND',
    'REPLACE', 'IS',
}
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_]')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def needs_backticks(identifier):
    if identifier.upper() in SQL_RESERVED_WORDS:
        return True
    if SPECIAL_CHARS_PATTERN.search(identifier):
        return True
    return False


def format_identifier(identifier):
    if needs_backticks(identifier):
        return f'"{identifier}"'
    return identifier


def sample_table_values(db_file_dir, table_names, limit_num):
    db_values_dict = dict()
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()
    safe_limit = limit_num if isinstance(limit_num, int) and limit_num > 0 else 20

    for table_name in table_names:
        try:
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns = cursor.fetchall()
            column_names = [column[1] for column in columns]

            for column_name in column_names:
                if not isinstance(column_name, str) or column_name.strip() == "":
                    continue
                query = f"""
                SELECT "{column_name}"
                FROM (
                    SELECT DISTINCT "{column_name}"
                    FROM "{table_name}"
                    WHERE "{column_name}" IS NOT NULL and "{column_name}" != ''
                ) AS unique_values
                LIMIT {safe_limit};
                """
                cursor.execute(query)
                values = cursor.fetchall()
                values = [value[0] for value in values]

                for idx in range(len(values)):
                    if isinstance(values[idx], str):
                        values[idx] = values[idx][:40]

                if len(values) > 0:
                    db_values_dict[f"{table_name}.{column_name}".lower()] = values
        except Exception:
            continue

    cursor.close()
    conn.close()
    return db_values_dict


def obtain_db_details(db_info, sampled_db_values_dict, relavant_db_values_dict,
                      used_column_idx_list, mode="dev"):
    """
    Generate DDL-style schema.
    used_column_idx_list: list/set of column indices to include (None = all columns)
    """
    db_details = []

    if used_column_idx_list is None:
        used_column_idx_list = list(range(len(db_info["column_names_original"])))
    else:
        used_column_idx_list = list(used_column_idx_list)

    for outer_table_idx, table_name in enumerate(db_info["table_names_original"]):
        column_info_list = []
        pk_columns = []
        fk_info = []

        for column_idx, ((inner_table_idx, column_name), (_, column_comment), column_type) in enumerate(zip(
            db_info["column_names_original"], db_info["column_names"], db_info["column_types"]
        )):
            if inner_table_idx == outer_table_idx:
                if column_idx not in used_column_idx_list:
                    continue

                column_values = []
                if f"{table_name}.{column_name}".lower() in relavant_db_values_dict:
                    column_values.extend(relavant_db_values_dict[f"{table_name}.{column_name}".lower()])
                if f"{table_name}.{column_name}".lower() in sampled_db_values_dict:
                    column_values.extend(sampled_db_values_dict[f"{table_name}.{column_name}".lower()])
                column_values = list(dict.fromkeys(column_values))  # dedup
                column_values = column_values[:6]

                if column_name.lower() in [
                    column_comment.lower(),
                    column_comment.lower().replace(" ", "_"),
                    column_comment.lower().replace(" ", "")
                ] or column_comment.strip() == "":
                    column_info = f'    {format_identifier(column_name)} {column_type},'
                    if len(column_values) > 0:
                        column_info += f" -- example: {column_values}"
                else:
                    column_info = f'    {format_identifier(column_name)} {column_type}, -- {column_comment}'
                    if len(column_values) > 0:
                        column_info += f", example: {column_values}"

                column_info_list.append(column_info)

                for primary_keys_idx in db_info["primary_keys"]:
                    if isinstance(primary_keys_idx, int):
                        if column_idx == primary_keys_idx:
                            pk_columns.append(column_name)
                    elif isinstance(primary_keys_idx, list):
                        if column_idx in primary_keys_idx:
                            pk_columns.append(column_name)

                for (source_column_idx, target_column_idx) in db_info["foreign_keys"]:
                    if column_idx == source_column_idx and target_column_idx in used_column_idx_list:
                        source_table_idx = db_info["column_names_original"][source_column_idx][0]
                        source_table_name = db_info["table_names_original"][source_table_idx]
                        source_column_name = db_info["column_names_original"][source_column_idx][1]
                        target_table_idx_val = db_info["column_names_original"][target_column_idx][0]
                        target_table_name = db_info["table_names_original"][target_table_idx_val]
                        target_column_name = db_info["column_names_original"][target_column_idx][1]
                        fk_str = (
                            f'    CONSTRAINT fk_{source_table_name.lower().replace(" ", "_")}'
                            f'_{source_column_name.lower().replace(" ", "_")}'
                            f' FOREIGN KEY ({format_identifier(source_column_name)})'
                            f' REFERENCES {format_identifier(target_table_name)}'
                            f' ({format_identifier(target_column_name)}),'
                        )
                        fk_info.append(fk_str)

        if len(column_info_list) > 0:
            pk_columns = list(OrderedDict.fromkeys(pk_columns))
            if len(pk_columns) > 0:
                pk_info = [
                    '    PRIMARY KEY ('
                    + ', '.join([f'{format_identifier(c)}' for c in pk_columns])
                    + '),'
                ]
            else:
                pk_info = []
            fk_info = list(OrderedDict.fromkeys(fk_info))

            table_ddl = f'CREATE TABLE {format_identifier(table_name)} (\n'
            table_ddl += "\n".join(column_info_list + pk_info + fk_info)
            if table_ddl.endswith(","):
                table_ddl = table_ddl[:-1]
            table_ddl += "\n);"

            db_details.append(table_ddl)

    db_details = "\n\n".join(db_details)
    return db_details


@lru_cache(maxsize=None)
def load_prompt_template(filename: str) -> str:
    """Load prompt templates from prompts/ so files are the source of truth."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


INPUT_PROMPT_TEMPLATE = load_prompt_template("generator_prompt.txt")
RETRIEVER_PROMPT_TEMPLATE = load_prompt_template("retriever_prompt.txt")
