"""准备Spider数据集的检索器提示词"""
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# 从prepare_eval_data.py导入Bird的函数
from prepare_eval_data import sample_table_values, obtain_db_details

def prepare_spider_retriever_prompts(
    data_file: str,
    tables_file: str,
    db_path: str,
    output_file: str,
    prompt_template_file: str,
    value_limit: int = 2,
    prompt_format: str = "list"
):
    """准备Spider检索器提示词（使用Bird的obtain_db_details）"""
    with open(data_file, 'r') as f:
        data = json.load(f)

    with open(tables_file, 'r') as f:
        tables = json.load(f)

    with open(prompt_template_file, 'r') as f:
        prompt_template = f.read()

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

    prompts = []
    for item in data:
        db_id = item['db_id']
        question = item['question']
        db_info = db_map[db_id]

        # 使用obtain_db_details生成带值采样的schema
        sampled_values = db_id2sampled_values.get(db_id, {})
        full_schema = obtain_db_details(db_info, sampled_values, {}, None)

        instruction = prompt_template.replace("{question}", question).replace("{schema}", full_schema)
        if prompt_format == "json":
            instruction += "\n\nPlease output in JSON format: {\"columns\": [...]}"

        prompts.append({
            "db_id": db_id,
            "question": question,
            "input_seq": instruction,
            "db_info_raw": json.dumps(db_info, ensure_ascii=False)
        })

    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"生成 {len(prompts)} 条检索器提示词 -> {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--prompt_template", required=True)
    parser.add_argument("--value_limit", type=int, default=2)
    parser.add_argument("--prompt_format", default="list", choices=["list", "json"])
    args = parser.parse_args()

    prepare_spider_retriever_prompts(
        args.data_file, args.tables_file, args.db_path,
        args.output_file, args.prompt_template, args.value_limit, args.prompt_format
    )
