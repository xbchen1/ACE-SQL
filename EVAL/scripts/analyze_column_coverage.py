#!/usr/bin/env python3
"""
Analyze column coverage and noise rate between predicted SQLs and gold SQLs.

This script calculates:
1. Coverage rate: How many gold columns are covered by predicted columns
2. Noise rate: How many predicted columns are outside the gold columns

Two modes:
- MAJ mode: Merge columns from all 8 predicted SQLs
- Greedy mode: Use only the first predicted SQL
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import BIRD_TABLES, BIRD_DB_PATH

from generate_prompts_genlink import extract_sql_columns_improved


def extract_columns_from_sql_with_db_path(sql: str, db_id: str, db_path: str):
    """
    使用改进后的方法从 SQL 中提取列

    Args:
        sql: SQL 查询语句
        db_id: 数据库 ID
        db_path: 数据库根目录

    Returns:
        set of (table_name, column_name) tuples
    """
    if not sql or not sql.strip():
        return set()

    db_file = os.path.join(db_path, db_id, db_id + ".sqlite")
    if not os.path.exists(db_file):
        return set()

    try:
        extracted_schema = extract_sql_columns_improved(db_file, sql)
        # 转换为 (table_name, column_name) 元组集合
        columns = set()
        for table_name, col_names in extracted_schema.items():
            for col_name in col_names:
                columns.add((table_name, col_name))
        return columns
    except Exception as e:
        return set()


def normalize_column_set(columns):
    """Normalize column set to lowercase for comparison."""
    return set((t.lower(), c.lower()) for t, c in columns)


def calculate_metrics(gold_columns, pred_columns):
    """
    Calculate coverage rate and noise rate.

    Coverage rate = |pred ∩ gold| / |gold|  (how many gold columns are covered)
    Noise rate = |pred - gold| / |pred|  (how many pred columns are noise)
    """
    gold_norm = normalize_column_set(gold_columns)
    pred_norm = normalize_column_set(pred_columns)

    if len(gold_norm) == 0:
        coverage = 1.0 if len(pred_norm) == 0 else 0.0
    else:
        covered = len(gold_norm & pred_norm)
        coverage = covered / len(gold_norm)

    if len(pred_norm) == 0:
        noise = 0.0
    else:
        extra = len(pred_norm - gold_norm)
        noise = extra / len(pred_norm)

    return coverage, noise


def analyze_file(input_file, tables_file, db_path, output_file=None):
    """Analyze a single SQL output file."""

    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading tables from {tables_file}")
    with open(tables_file, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    # Build db_id -> db_info mapping
    db_id2info = {db["db_id"]: db for db in tables_data}

    # Statistics
    maj_coverages = []
    maj_noises = []
    greedy_coverages = []
    greedy_noises = []

    detailed_results = []

    for idx, item in enumerate(data):
        db_id = item.get("db_id", "")
        gold_sql = item.get("SQL", "")
        pred_sqls = item.get("pred_sqls", [])

        if db_id not in db_id2info:
            print(f"Warning: db_id '{db_id}' not found in tables, skipping item {idx}")
            continue

        # Extract gold columns using improved method
        gold_columns = extract_columns_from_sql_with_db_path(gold_sql, db_id, db_path)

        # MAJ mode: merge all pred_sqls columns
        maj_columns = set()
        for pred_sql in pred_sqls:
            pred_cols = extract_columns_from_sql_with_db_path(pred_sql, db_id, db_path)
            maj_columns.update(pred_cols)

        # Greedy mode: use first pred_sql
        if pred_sqls:
            greedy_columns = extract_columns_from_sql_with_db_path(pred_sqls[0], db_id, db_path)
        else:
            greedy_columns = set()

        # Calculate metrics
        maj_coverage, maj_noise = calculate_metrics(gold_columns, maj_columns)
        greedy_coverage, greedy_noise = calculate_metrics(gold_columns, greedy_columns)

        maj_coverages.append(maj_coverage)
        maj_noises.append(maj_noise)
        greedy_coverages.append(greedy_coverage)
        greedy_noises.append(greedy_noise)

        detailed_results.append({
            "question_id": item.get("question_id", idx),
            "db_id": db_id,
            "gold_columns": list(gold_columns),
            "gold_column_count": len(gold_columns),
            "maj_columns": list(maj_columns),
            "maj_column_count": len(maj_columns),
            "greedy_columns": list(greedy_columns),
            "greedy_column_count": len(greedy_columns),
            "maj_coverage": maj_coverage,
            "maj_noise": maj_noise,
            "greedy_coverage": greedy_coverage,
            "greedy_noise": greedy_noise,
        })

    # Calculate averages
    n = len(maj_coverages)
    avg_maj_coverage = sum(maj_coverages) / n if n > 0 else 0
    avg_maj_noise = sum(maj_noises) / n if n > 0 else 0
    avg_greedy_coverage = sum(greedy_coverages) / n if n > 0 else 0
    avg_greedy_noise = sum(greedy_noises) / n if n > 0 else 0

    summary = {
        "input_file": input_file,
        "total_samples": n,
        "maj_mode": {
            "avg_coverage": avg_maj_coverage,
            "avg_noise": avg_maj_noise,
        },
        "greedy_mode": {
            "avg_coverage": avg_greedy_coverage,
            "avg_noise": avg_greedy_noise,
        }
    }

    # Print summary
    print("\n" + "="*60)
    print(f"Analysis Results for: {os.path.basename(input_file)}")
    print("="*60)
    print(f"Total samples: {n}")
    print()
    print("MAJ Mode (merge all 8 pred_sqls):")
    print(f"  Average Coverage Rate: {avg_maj_coverage:.4f} ({avg_maj_coverage*100:.2f}%)")
    print(f"  Average Noise Rate:    {avg_maj_noise:.4f} ({avg_maj_noise*100:.2f}%)")
    print()
    print("Greedy Mode (first pred_sql only):")
    print(f"  Average Coverage Rate: {avg_greedy_coverage:.4f} ({avg_greedy_coverage*100:.2f}%)")
    print(f"  Average Noise Rate:    {avg_greedy_noise:.4f} ({avg_greedy_noise*100:.2f}%)")
    print("="*60)

    # Save detailed results if output file specified
    if output_file:
        output_data = {
            "summary": summary,
            "detailed_results": detailed_results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze column coverage and noise rate")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input SQL output file (json)")
    parser.add_argument("--tables", "-t", type=str,
                        default=BIRD_TABLES,
                        help="Path to tables.json")
    parser.add_argument("--db_path", "-d", type=str,
                        default=BIRD_DB_PATH,
                        help="Path to database directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for detailed results (optional)")

    args = parser.parse_args()

    analyze_file(args.input, args.tables, args.db_path, args.output)


if __name__ == "__main__":
    main()
