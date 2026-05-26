"""
根据检索器合并输出，生成SQL提示词

流程:
1. 读取合并后的检索器输出（包含 merged_column_indices 和 db_info_raw）
2. 采样数据库值 + Lucene 值检索
3. 使用 obtain_db_details 构建精简 DDL schema
4. 使用 INPUT_PROMPT_TEMPLATE 生成 SQL 提示词
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import BIRD_DB_PATH, BIRD_TABLES, BIRD_DB_CONTENT_INDEX_PATH, VALUE_LIMIT_NUM
from prompt_templates import INPUT_PROMPT_TEMPLATE

# 从 prepare_eval_data.py 导入函数（包括 Lucene 值检索）
from prepare_eval_data import (
    sample_table_values, obtain_db_details,
    obtain_n_grams, retrieve_relevant_hits, deduplicate_dicts,
    retrieve_question_related_db_values, _get_hit_raw
)

from src.utils.schema_utils import parse_retriever_output

# 默认 db_content_index 路径
DEFAULT_DB_CONTENT_INDEX_PATH = BIRD_DB_CONTENT_INDEX_PATH


# ============== 主逻辑 ==============

def generate_sql_prompts(
    retriever_output_path: str,
    db_path: str,
    output_path: str,
    value_limit_num: int = 2,
    db_content_index_path: str = None
):
    """
    根据检索器合并输出生成SQL提示词

    Args:
        retriever_output_path: 合并后的检索器输出文件路径
        db_path: 数据库目录路径
        output_path: 输出文件路径
        value_limit_num: 每列采样值数量
        db_content_index_path: Lucene 索引路径（可选）
    """
    print(f"加载检索器输出: {retriever_output_path}")
    with open(retriever_output_path, 'r', encoding='utf-8') as f:
        retriever_data = json.load(f)

    # 解析每个样本的 db_info
    db_info_cache = {}
    for item in retriever_data:
        db_id = item['db_id']
        if db_id not in db_info_cache and 'db_info_raw' in item:
            db_info_raw = item['db_info_raw']
            if isinstance(db_info_raw, str):
                db_info_cache[db_id] = json.loads(db_info_raw)
            else:
                db_info_cache[db_id] = db_info_raw

    used_db_ids = list(set([item['db_id'] for item in retriever_data]))

    # 预先采样数据库值
    print("采样数据库值...")
    db_id2sampled_values = {}
    for db_id in tqdm(used_db_ids, desc="采样数据库值"):
        db_file = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        db_info = db_info_cache.get(db_id)
        if os.path.exists(db_file) and db_info and db_info.get("table_names_original"):
            try:
                sampled_values = sample_table_values(
                    db_file,
                    db_info["table_names_original"],
                    value_limit_num
                )
                db_id2sampled_values[db_id] = sampled_values
            except Exception as e:
                print(f"警告: 采样 {db_id} 失败: {e}")
                db_id2sampled_values[db_id] = {}
        else:
            db_id2sampled_values[db_id] = {}

    # Lucene 值检索
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

    # 生成SQL提示词
    print("生成SQL提示词...")
    sql_prompts = []

    for idx, item in enumerate(tqdm(retriever_data, desc="生成SQL提示词")):
        question_id = item.get("question_id", idx)
        db_id = item.get("db_id", "")
        question = item.get("question", "")
        evidence = item.get("evidence", "")
        gold_sql = item.get("SQL", "")
        difficulty = item.get("difficulty", "")
        retriever_response = item.get("retriever_response", "")

        db_info = db_info_cache.get(db_id)
        if db_info is None:
            print(f"Warning: db_info not found for {db_id}, skipping sample {question_id}")
            continue

        # 获取合并后的列索引
        merged_indices = item.get('merged_column_indices')
        if merged_indices is not None:
            pred_indices = [int(i) for i in merged_indices]
        else:
            # 回退: 从检索器响应文本解析列索引
            pred_text = item.get('merged_output', retriever_response)
            pred_indices = list(parse_retriever_output(pred_text, db_info))

        # 获取采样的数据库值
        sampled_values = db_id2sampled_values.get(db_id, {})

        # 构建完整问题
        if evidence and evidence.strip():
            full_question = f"{evidence.strip()}\n{question}"
        else:
            full_question = question

        # Lucene 值检索（逐样本）
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

        # 使用 obtain_db_details 生成带值采样的精简 schema
        ddl_schema = obtain_db_details(db_info, sampled_values, relavant_db_values_dict, pred_indices)

        # 使用模板生成提示词
        input_seq = INPUT_PROMPT_TEMPLATE.format(
            db_engine="SQLite",
            db_details=ddl_schema,
            question=full_question
        )

        sql_prompt = {
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "SQL": gold_sql,
            "difficulty": difficulty,
            "input_seq": input_seq,
            "output_seq": gold_sql,
            "selected_schema": ddl_schema,
            "retriever_response": retriever_response
        }
        sql_prompts.append(sql_prompt)

    # 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sql_prompts, f, indent=2, ensure_ascii=False)

    print(f"生成 {len(sql_prompts)} 条SQL提示词 -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="根据检索器输出生成SQL提示词")
    parser.add_argument(
        "--retriever_output_path",
        type=str,
        required=True,
        help="合并后的检索器输出文件路径"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=BIRD_DB_PATH,
        help="数据库目录路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出文件路径"
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

    generate_sql_prompts(
        retriever_output_path=args.retriever_output_path,
        db_path=args.db_path,
        output_path=args.output_path,
        value_limit_num=args.value_limit_num,
        db_content_index_path=args.db_content_index_path
    )


if __name__ == "__main__":
    main()
