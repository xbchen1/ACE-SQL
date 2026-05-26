"""
准备BIRD评估数据
使用训练时的prompt格式生成检索器输入

修改：使用与训练数据相同的 obtain_db_details 生成 schema（包含示例值）
      增加 Lucene 值检索（与 OmniSQL 一致）
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import OrderedDict
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk import ngrams

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    BIRD_DEV_DATA,
    BIRD_TABLES,
    BIRD_DB_PATH,
    BIRD_DB_CONTENT_INDEX_PATH,
    RETRIEVER_PROMPT_PATH,
    DEFAULT_OUTPUT_DIR,
    VALUE_LIMIT_NUM,
)
from src.utils.schema_utils import (
    load_tables_json,
    build_ddl_schema,
    extract_column_indices_from_sql
)

# 默认 db_content_index 路径
DEFAULT_DB_CONTENT_INDEX_PATH = BIRD_DB_CONTENT_INDEX_PATH


# ============== 从 generate_prompts.py 复制的函数（与训练数据生成一致） ==============

SQL_RESERVED_WORDS = {'IDENTIFIED', 'FOREIGN', 'CONSTRAINT', 'USER', 'POSITION', 'DESCRIBE', 'CHECK', 'RECURSIVE', 'REAL', 'CONTINUE', 'GLOBAL', 'RLIKE', 'INSENSITIVE', 'BOOLEAN', 'CHAR', 'ROLE', 'CASE', 'SCHEMA', 'CLOB', 'RESIGNAL', 'ROW', 'DEC', 'TOP', 'EXCEPT', 'SENSITIVE', 'OUT', 'RENAME', 'READS', 'BLOB', 'INT', 'EXTERNAL', 'LOCALTIMESTAMP', 'DECLARE', 'DO', 'AS', 'OVER', 'CONDITION', 'SELECT', 'SAVEPOINT', 'WITHIN', 'ELSEIF', 'UNLOCK', 'DATABASE', 'TRIGGER', 'ACCESS', 'FALSE', 'BREAK', 'ITERATE', 'SMALLINT', 'ASC', 'YEAR', 'DELETE', 'ROLLBACK', 'ON', 'ESCAPE', 'CREATE', 'MONTH', 'SPECIFIC', 'SESSION', 'SQLSTATE', 'HOLD', 'SET', 'EXPLAIN', 'RETURN', 'ROWNUM', 'BINARY', 'SYSDATE', 'SQLWARNING', 'EXTEND', 'CAST', 'FOR', 'TERMINATED', 'VIEW', 'TRAILING', 'HOUR', 'VARYING', 'RESTRICT', 'RIGHT', 'DISTINCT', 'JOIN', 'UNKNOWN', 'VALUES', 'TABLE', 'OR', 'DOUBLE', 'DROP', 'COMMIT', 'PRECISION', 'LANGUAGE', 'START', 'INTERSECT', 'IGNORE', 'NULL', 'CURRENT_DATE', 'LOCK', 'INTO', 'NEW', 'DESC', 'STATIC', 'MODIFIES', 'GRANT', 'VALUE', 'LIMIT', 'MODULE', 'DATE', 'LOCALTIME', 'PERCENT', 'REPEAT', 'FULL', 'USAGE', 'ORDER', 'WHEN', 'PRIMARY', 'BETWEEN', 'CURSOR', 'DECIMAL', 'HAVING', 'IF', 'FILTER', 'INDEX', 'ILIKE', 'VARCHAR', 'EXEC', 'USING', 'ROWS', 'PLACING', 'WHILE', 'EXECUTE', 'EACH', 'LEFT', 'FLOAT', 'COLLATE', 'CURRENT_TIME', 'OPEN', 'RANGE', 'CROSS', 'FUNCTION', 'TIME', 'BOTH', 'NOT', 'CONVERT', 'NCHAR', 'KEY', 'DEFAULT', 'LIKE', 'ANALYZE', 'EXISTS', 'IN', 'BIT', 'INOUT', 'SUM', 'NUMERIC', 'AFTER', 'LEAVE', 'INSERT', 'TO', 'COUNT', 'THEN', 'BEFORE', 'OUTER', 'COLUMN', 'ONLY', 'END', 'PROCEDURE', 'OFFSET', 'ADD', 'INNER', 'RELEASE', 'FROM', 'DAY', 'NO', 'CALL', 'BY', 'LOCAL', 'ZONE', 'TRUE', 'EXIT', 'LEADING', 'INTEGER', 'MERGE', 'OLD', 'AVG', 'MIN', 'SQL', 'LOOP', 'SIGNAL', 'REFERENCES', 'MINUTE', 'UNIQUE', 'GENERATED', 'ALL', 'MATCH', 'CASCADE', 'UNION', 'COMMENT', 'FETCH', 'UNDO', 'UPDATE', 'WHERE', 'ELSE', 'PARTITION', 'BIGINT', 'CHARACTER', 'CURRENT_TIMESTAMP', 'ALTER', 'INTERVAL', 'REVOKE', 'CONNECT', 'WITH', 'TIMESTAMP', 'GROUP', 'BEGIN', 'CURRENT', 'REGEXP', 'NATURAL', 'SOME', 'SQLEXCEPTION', 'MAX', 'SUBSTRING', 'OF', 'AND', 'REPLACE', 'IS'}
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_]')


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
    """从数据库采样列值"""
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
        except Exception as e:
            continue

    cursor.close()
    conn.close()
    return db_values_dict


def obtain_db_details(db_info, sampled_db_values_dict, relavant_db_values_dict,
                      used_column_idx_list, mode="dev"):
    """生成 DDL 格式的 schema（与训练数据一致，包含示例值）"""
    db_details = []

    if used_column_idx_list is None:
        used_column_idx_list = list(range(len(db_info["column_names_original"])))

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
                        target_table_idx = db_info["column_names_original"][target_column_idx][0]
                        target_table_name = db_info["table_names_original"][target_table_idx]
                        target_column_name = db_info["column_names_original"][target_column_idx][1]
                        fk_info.append(f'    CONSTRAINT fk_{source_table_name.lower().replace(" ", "_")}_{source_column_name.lower().replace(" ", "_")} FOREIGN KEY ({format_identifier(source_column_name)}) REFERENCES {format_identifier(target_table_name)} ({format_identifier(target_column_name)}),')

        if len(column_info_list) > 0:
            pk_columns = list(OrderedDict.fromkeys(pk_columns))
            if len(pk_columns) > 0:
                pk_info = ['    PRIMARY KEY (' + ', '.join([f'{format_identifier(column_name)}' for column_name in pk_columns]) + '),']
            else:
                pk_info = []
            fk_info = list(OrderedDict.fromkeys(fk_info))

            table_ddl = ""
            table_ddl += f'CREATE TABLE {format_identifier(table_name)} (\n'
            table_ddl += "\n".join(column_info_list + pk_info + fk_info)
            if table_ddl.endswith(","):
                table_ddl = table_ddl[:-1]
            table_ddl += "\n);"

            db_details.append(table_ddl)

    db_details = "\n\n".join(db_details)
    return db_details


def load_retriever_prompt_template(prompt_path: str) -> str:
    """加载检索器提示词模板"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============== Lucene 值检索（来自 OmniSQL） ==============

def obtain_n_grams(sequence, max_n):
    """返回 sequence 的所有 n-gram (n <= max_n)"""
    tokens = word_tokenize(sequence)
    all_n_grams = []
    for n in range(1, max_n + 1):
        all_n_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    return all_n_grams


def calculate_substring_match_percentage(query, target):
    """计算 query 在 target 中的最长子串匹配率"""
    query = query.lower()
    target = target.lower()
    substrings = []
    for i in range(len(query)):
        for j in range(i + 1, len(query) + 1):
            substrings.append(query[i:j])
    max_matched_substring_len = max(
        [len(substring) for substring in substrings if substring in target]
    )
    return max_matched_substring_len / len(query)


def _get_hit_raw(hit):
    """兼容不同版本 pyserini 获取 hit 的 raw JSON"""
    if hasattr(hit, 'raw') and hit.raw:
        return hit.raw
    doc = hit.lucene_document
    return doc.get('raw')


def retrieve_relevant_hits(searcher, queries):
    """使用 Lucene 批量检索相关数据库值"""
    queries = list(dict.fromkeys(queries))
    q_ids = [f"{idx}" for idx in range(len(queries))]
    query2hits = dict()
    search_results = searcher.batch_search(queries, q_ids, k=10, threads=60)
    for query, q_id in zip(queries, q_ids):
        hits = search_results[q_id]
        raw_list = []
        for hit in hits:
            raw = _get_hit_raw(hit)
            if raw and raw not in raw_list:
                raw_list.append(raw)
        hits = [json.loads(r) for r in raw_list]
        query2hits[query] = hits
    return query2hits


def deduplicate_dicts(dict_list):
    """去重 dict 列表"""
    seen = set()
    unique_dicts = []
    for d in dict_list:
        dict_tuple = frozenset(d.items())
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_dicts.append(d)
    return unique_dicts


def retrieve_question_related_db_values(hits, question):
    """从检索结果中筛选与 question 高度匹配的数据库值"""
    high_score_hits = []
    for idx, hit in enumerate(hits):
        table_name, column_name, c_id = hit["id"].split("-**-")
        score = calculate_substring_match_percentage(hit["contents"], question)
        if score > 0.85:
            high_score_hits.append({
                "table_dot_column_lower_case": f"{table_name}.{column_name}".lower(),
                "db_value": hit["contents"],
                "score": score,
                "index": idx,
            })
    high_score_hits = sorted(
        high_score_hits,
        key=lambda x: (x["score"], len(x["db_value"]), x["index"]),
        reverse=True
    )
    high_score_hits = high_score_hits[:20]

    relavant_db_values_dict = dict()
    for hit in high_score_hits:
        key = hit["table_dot_column_lower_case"]
        if key in relavant_db_values_dict:
            relavant_db_values_dict[key].append(hit["db_value"])
        else:
            relavant_db_values_dict[key] = [hit["db_value"]]

    return relavant_db_values_dict


def prepare_bird_eval_data(
    dev_data_path: str,
    tables_path: str,
    retriever_prompt_path: str,
    output_path: str,
    db_path: str = BIRD_DB_PATH,
    value_limit_num: int = VALUE_LIMIT_NUM,
    max_samples: Optional[int] = None,
    db_content_index_path: str = None
):
    """
    准备BIRD评估数据

    Args:
        dev_data_path: BIRD dev.json路径
        tables_path: BIRD dev_tables.json路径
        retriever_prompt_path: 检索器提示词模板路径
        output_path: 输出文件路径
        db_path: 数据库文件目录
        value_limit_num: 每列采样值数量
        max_samples: 最大样本数（用于测试）
    """
    print(f"加载BIRD数据: {dev_data_path}")
    with open(dev_data_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    print(f"加载表Schema: {tables_path}")
    tables_map = load_tables_json(tables_path)

    print(f"加载检索器提示词模板: {retriever_prompt_path}")
    retriever_prompt_template = load_retriever_prompt_template(retriever_prompt_path)

    # 预采样数据库值（与训练数据生成方式一致）
    print("预采样数据库值...")
    db_id_to_sampled_values = {}
    used_db_ids = set(s.get('db_id', '') for s in dev_data[:max_samples] if s.get('db_id'))

    for db_id in tqdm(used_db_ids, desc="采样数据库值"):
        db_file = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        db_info = tables_map.get(db_id, {})
        if os.path.exists(db_file) and db_info.get("table_names_original"):
            try:
                sampled = sample_table_values(db_file, db_info["table_names_original"], value_limit_num)
                db_id_to_sampled_values[db_id] = sampled
            except Exception as e:
                print(f"警告: 采样 {db_id} 失败: {e}")
                db_id_to_sampled_values[db_id] = {}
        else:
            db_id_to_sampled_values[db_id] = {}

    # 加载 Lucene 索引
    db_id2searcher = {}
    if db_content_index_path and os.path.exists(db_content_index_path):
        from pyserini.search.lucene import LuceneSearcher
        print(f"加载 Lucene 索引: {db_content_index_path}")
        for db_id in tqdm(used_db_ids, desc="加载Lucene索引"):
            index_path = os.path.join(db_content_index_path, db_id)
            if os.path.exists(index_path):
                try:
                    db_id2searcher[db_id] = LuceneSearcher(index_path)
                except Exception as e:
                    print(f"警告: 加载 {db_id} Lucene 索引失败: {e}")
        print(f"成功加载 {len(db_id2searcher)}/{len(used_db_ids)} 个 Lucene 索引")
    else:
        print("未指定 db_content_index_path 或路径不存在，跳过 Lucene 值检索")

    eval_samples = []
    skipped = 0

    for idx, sample in enumerate(tqdm(dev_data, desc="准备评估数据")):
        if max_samples and idx >= max_samples:
            break

        question_id = sample.get("question_id", idx)
        db_id = sample.get("db_id", "")
        question = sample.get("question", "")
        evidence = sample.get("evidence", "")
        gold_sql = sample.get("SQL", "")
        difficulty = sample.get("difficulty", "")

        # 获取数据库Schema信息
        db_info = tables_map.get(db_id)
        if not db_info:
            print(f"警告: 跳过样本 {question_id}, 找不到数据库 {db_id}")
            skipped += 1
            continue

        # 获取采样的数据库值
        sampled_values = db_id_to_sampled_values.get(db_id, {})

        # 如果有evidence，将其添加到question中（与训练数据格式一致）
        if evidence and evidence.strip():
            full_question = f"{evidence.strip()}\n{question}"
        else:
            full_question = question

        # Lucene 值检索
        relavant_db_values_dict = {}
        if db_id in db_id2searcher:
            queries = obtain_n_grams(full_question, 8) + [full_question]
            queries = list(dict.fromkeys(queries))
            query2hits = retrieve_relevant_hits(db_id2searcher[db_id], queries)
            hits = []
            for query in queries:
                hits.extend(query2hits.get(query, []))
            hits = deduplicate_dicts(hits)
            relavant_db_values_dict = retrieve_question_related_db_values(hits, full_question)

        # 使用 obtain_db_details 构建 Schema（与训练数据一致，包含示例值）
        full_schema = obtain_db_details(db_info, sampled_values, relavant_db_values_dict, None)

        # 从SQL提取黄金标准Schema（用于后续评估，仍使用 build_ddl_schema）
        gold_indices = extract_column_indices_from_sql(gold_sql, db_info)
        gold_context = build_ddl_schema(db_info, gold_indices) if gold_indices else full_schema

        # 构建检索器输入提示词（使用训练时的格式）
        prompt = retriever_prompt_template.replace("{question}", full_question).replace("{schema}", full_schema)

        eval_sample = {
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "SQL": gold_sql,
            "difficulty": difficulty,
            "input_seq": prompt,  # 用于推理
            "gold_context": gold_context,  # 用于评估
            "db_info_raw": json.dumps(db_info, ensure_ascii=False)
        }
        eval_samples.append(eval_sample)

    print(f"成功准备 {len(eval_samples)} 个评估样本，跳过 {skipped} 个")

    # 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, indent=2, ensure_ascii=False)

    print(f"评估数据已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="准备BIRD评估数据")
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default=BIRD_DEV_DATA,
        help="BIRD dev.json路径"
    )
    parser.add_argument(
        "--tables_path",
        type=str,
        default=BIRD_TABLES,
        help="BIRD dev_tables.json路径"
    )
    parser.add_argument(
        "--retriever_prompt_path",
        type=str,
        default=RETRIEVER_PROMPT_PATH,
        help="检索器提示词模板路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "retriever_prompts.json"),
        help="输出文件路径"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（用于测试）"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=BIRD_DB_PATH,
        help="数据库文件目录"
    )
    parser.add_argument(
        "--value_limit_num",
        type=int,
        default=VALUE_LIMIT_NUM,
        help="每列采样值数量"
    )
    parser.add_argument(
        "--db_content_index_path",
        type=str,
        default=DEFAULT_DB_CONTENT_INDEX_PATH,
        help="Lucene 索引路径（用于值检索）"
    )

    args = parser.parse_args()

    prepare_bird_eval_data(
        dev_data_path=args.dev_data_path,
        tables_path=args.tables_path,
        retriever_prompt_path=args.retriever_prompt_path,
        output_path=args.output_path,
        db_path=args.db_path,
        value_limit_num=args.value_limit_num,
        max_samples=args.max_samples,
        db_content_index_path=args.db_content_index_path
    )


if __name__ == "__main__":
    main()
