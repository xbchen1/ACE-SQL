"""生成Spider SQL提示词（基于检索器输出）- 对齐Bird逻辑"""
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.schema_utils import parse_retriever_output
from prompt_templates import INPUT_PROMPT_TEMPLATE

# 从prepare_eval_data.py导入Bird的函数
from prepare_eval_data import sample_table_values, obtain_db_details

def generate_spider_sql_prompts(
    data_file: str,
    tables_file: str,
    db_path: str,
    retriever_output_file: str,
    output_file: str,
    value_limit: int = 2
):
    """生成Spider SQL提示词（使用Bird的obtain_db_details）"""
    with open(data_file, 'r') as f:
        data = json.load(f)

    with open(tables_file, 'r') as f:
        tables = json.load(f)

    with open(retriever_output_file, 'r') as f:
        retriever_outputs = json.load(f)

    db_map = {db['db_id']: db for db in tables}

    # 预先采样数据库值
    print("采样数据库值...")
    db_id2sampled_values = {}
    used_db_ids = list(set([item['db_id'] for item in data]))
    for db_id in tqdm(used_db_ids):
        db_file = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_file) and db_id in db_map:
            sampled_values = sample_table_values(
                db_file,
                db_map[db_id]["table_names_original"],
                value_limit
            )
            db_id2sampled_values[db_id] = sampled_values
        else:
            db_id2sampled_values[db_id] = {}

    sql_prompts = []
    for item, ret_out in zip(data, retriever_outputs):
        db_id = item['db_id']
        question = item['question']
        db_info = db_map[db_id]

        # 解析检索器输出
        if isinstance(ret_out, dict):
            merged_indices = ret_out.get('merged_column_indices')
            if merged_indices is not None:
                pred_indices = [int(idx) for idx in merged_indices]
            else:
                pred_text = ret_out.get('merged_output', '')
                pred_indices = list(parse_retriever_output(pred_text, db_info))
        else:
            pred_indices = list(parse_retriever_output(ret_out, db_info))

        # 使用obtain_db_details生成带值采样的schema
        sampled_values = db_id2sampled_values.get(db_id, {})
        ddl_schema = obtain_db_details(db_info, sampled_values, {}, pred_indices)

        # 使用Bird的模板格式
        input_seq = INPUT_PROMPT_TEMPLATE.format(
            db_engine="SQLite",
            db_details=ddl_schema,
            question=question
        )

        sql_prompts.append({
            "db_id": db_id,
            "question": question,
            "input_seq": input_seq,
            "selected_schema": ddl_schema,
            "merged_column_indices": pred_indices
        })

    with open(output_file, 'w') as f:
        json.dump(sql_prompts, f, indent=2, ensure_ascii=False)

    print(f"生成 {len(sql_prompts)} 条SQL提示词 -> {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--retriever_output", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--value_limit", type=int, default=2)
    args = parser.parse_args()

    generate_spider_sql_prompts(
        args.data_file, args.tables_file, args.db_path,
        args.retriever_output, args.output_file, args.value_limit
    )
