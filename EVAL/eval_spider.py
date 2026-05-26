#!/usr/bin/env python3
"""Spider评估 - 直接使用OmniSQL的评估逻辑"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OMNISQL_EVAL_DIR = Path(os.environ.get("OMNISQL_EVAL_DIR", PROJECT_ROOT / "omnisql_eval"))
sys.path.insert(0, str(OMNISQL_EVAL_DIR))
from evaluate_spider import run_spider_eval

sys.path.insert(0, str(PROJECT_ROOT))
from spider_config import *
from src.utils.schema_utils import build_ddl_schema

def ensure_spider_retriever_metadata(retriever_output_file: str, tables_file: str):
    """为 Spider 检索结果补齐 db_info_raw，兼容 merge_retriever_outputs.py。"""
    if not os.path.exists(retriever_output_file):
        return

    with open(retriever_output_file, 'r', encoding='utf-8') as f:
        retriever_data = json.load(f)

    if not retriever_data:
        return

    needs_update = any(not sample.get("db_info_raw") for sample in retriever_data if isinstance(sample, dict))
    if not needs_update:
        return

    with open(tables_file, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    db_map = {db["db_id"]: db for db in tables}

    updated = False
    for sample in retriever_data:
        if not isinstance(sample, dict):
            continue
        if sample.get("db_info_raw"):
            continue
        db_info = db_map.get(sample.get("db_id"))
        if db_info is None:
            continue
        sample["db_info_raw"] = json.dumps(db_info, ensure_ascii=False)
        updated = True

    if updated:
        with open(retriever_output_file, 'w', encoding='utf-8') as f:
            json.dump(retriever_data, f, indent=2, ensure_ascii=False)


def generate_full_schema_sql_prompts(
    data_file: str,
    tables_file: str,
    output_file: str
):
    """使用完整 schema 直接生成 SQL 提示词。"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(tables_file, 'r', encoding='utf-8') as f:
        tables = json.load(f)

    db_map = {db["db_id"]: db for db in tables}

    sql_prompts = []
    for item in data:
        db_id = item["db_id"]
        question = item["question"]
        db_info = db_map[db_id]

        all_indices = {
            idx for idx, (table_idx, _) in enumerate(db_info["column_names_original"])
            if table_idx >= 0
        }
        ddl_schema = build_ddl_schema(db_info, all_indices, include_foreign_keys=True)
        instruction = f"-- Database: {db_id}\n{ddl_schema}\n\n-- Question: {question}\n-- SQL:"

        sql_prompts.append({
            "db_id": db_id,
            "question": question,
            "input_seq": instruction,
            "selected_schema": ddl_schema,
            "schema_mode": "full_schema",
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sql_prompts, f, indent=2, ensure_ascii=False)

    print(f"生成 {len(sql_prompts)} 条完整 Schema SQL 提示词 -> {output_file}")

def evaluate_spider(
    model_path: str,
    output_dir: str,
    split: str = "dev",
    num_samples: int = 1,
    temperature: float = 0.0,
    merge_mode: str = "union",
    prompt_format: str = "list",
    skip_inference: bool = False,
    pipeline_mode: str = "rag"
):
    """评估Spider"""
    os.makedirs(output_dir, exist_ok=True)

    if split == "dev":
        data_file = SPIDER_DEV_DATA
        tables_file = SPIDER_TABLES
        db_path = SPIDER_DB_PATH
        gold_file = SPIDER_DEV_GOLD
    else:
        data_file = SPIDER_TEST_DATA
        tables_file = SPIDER_TEST_TABLES
        db_path = SPIDER_TEST_DB_PATH
        gold_file = SPIDER_TEST_GOLD

    if pipeline_mode == "rag":
        retriever_prompts = os.path.join(output_dir, "retriever_prompts.json")
        retriever_outputs = os.path.join(output_dir, "retriever_outputs.json")
        merged_outputs = os.path.join(output_dir, "retriever_merged.json")
        sql_prompts = os.path.join(output_dir, "sql_prompts.json")
        sql_outputs = os.path.join(output_dir, "sql_outputs.json")
        infer_num_samples = num_samples
    else:
        retriever_prompts = None
        retriever_outputs = None
        merged_outputs = None
        sql_prompts = os.path.join(output_dir, "sql_prompts_full_schema.json")
        sql_outputs = os.path.join(output_dir, "sql_outputs_full_schema.json")
        infer_num_samples = 1

    if not skip_inference:
        # Step 1-5: 推理流程
        import subprocess

        if pipeline_mode == "rag":
            print("\n[Step 1/6] 准备检索器提示词...")
            if not os.path.exists(retriever_prompts):
                subprocess.run([
                    "python3", "prepare_spider_data.py",
                    "--data_file", data_file,
                    "--tables_file", tables_file,
                    "--db_path", db_path,
                    "--output_file", retriever_prompts,
                    "--prompt_template", RETRIEVER_PROMPT_PATH,
                    "--value_limit", str(VALUE_LIMIT_NUM),
                    "--prompt_format", prompt_format
                ], check=True)

            print("\n[Step 2/6] 检索器推理...")
            if not os.path.exists(retriever_outputs):
                subprocess.run([
                    "python3", "retriever_infer.py",
                    "--model_path", model_path,
                    "--input_file", retriever_prompts,
                    "--output_file", retriever_outputs,
                    "--temperature", str(temperature),
                    "--max_tokens", str(RETRIEVER_MAX_TOKENS),
                    "--num_samples", str(num_samples)
                ], check=True)

            print("\n[Step 3/6] 合并检索结果...")
            if not os.path.exists(merged_outputs):
                ensure_spider_retriever_metadata(retriever_outputs, tables_file)
                subprocess.run([
                    "python3", "merge_retriever_outputs.py",
                    "--input_file", retriever_outputs,
                    "--output_file", merged_outputs,
                    "--mode", merge_mode
                ], check=True)

            print("\n[Step 4/6] 生成SQL提示词...")
            if not os.path.exists(sql_prompts):
                subprocess.run([
                    "python3", "generate_spider_sql_prompts.py",
                    "--data_file", data_file,
                    "--tables_file", tables_file,
                    "--db_path", db_path,
                    "--retriever_output", merged_outputs,
                    "--output_file", sql_prompts,
                    "--value_limit", str(VALUE_LIMIT_NUM)
                ], check=True)
        else:
            if num_samples != 1:
                print(f"\n[Info] pipeline_mode=full_schema 时固定使用单次推理，num_samples 从 {num_samples} 调整为 1")

            print("\n[Step 1/6] 生成完整 Schema SQL 提示词...")
            if not os.path.exists(sql_prompts):
                generate_full_schema_sql_prompts(
                    data_file=data_file,
                    tables_file=tables_file,
                    output_file=sql_prompts,
                )

        print("\n[Step 5/6] SQL生成...")
        if not os.path.exists(sql_outputs):
            subprocess.run([
                "python3", "sql_infer.py",
                "--model_path", model_path,
                "--input_file", sql_prompts,
                "--output_file", sql_outputs,
                "--temperature", str(temperature),
                "--max_tokens", str(SQL_MAX_TOKENS),
                "--num_samples", str(infer_num_samples)
            ], check=True)

    # Step 6: 使用OmniSQL评估
    print("\n[Step 6/6] 评估 (使用OmniSQL逻辑)...")

    # sql_infer.py已经输出了正确的格式：[{"pred_sqls": [...], ...}, ...]
    # 直接使用即可
    with open(sql_outputs, 'r') as f:
        preds = json.load(f)

    # 调用OmniSQL的run_spider_eval
    mode = "major_voting" if infer_num_samples > 1 else "greedy_search"
    ex_acc, ts_acc = run_spider_eval(gold_file, sql_outputs, db_path, "", mode, True)

    result = {
        "split": split,
        "pipeline_mode": pipeline_mode,
        "ex_acc": ex_acc,
        "ts_acc": ts_acc
    }
    with open(os.path.join(output_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n准确率: {ex_acc}%")
    return ex_acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--merge_mode", default="union", choices=["union", "maj"])
    parser.add_argument("--prompt_format", default="list", choices=["list", "json"])
    parser.add_argument("--pipeline_mode", default="rag", choices=["rag", "full_schema"])
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()

    evaluate_spider(
        args.model_path, args.output_dir, args.split,
        args.num_samples, args.temperature, args.merge_mode,
        args.prompt_format, args.skip_inference, args.pipeline_mode
    )
