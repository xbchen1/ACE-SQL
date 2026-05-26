"""合并Spider检索器输出（union或majority voting）"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.schema_utils import parse_retriever_output

def merge_retriever_outputs(
    retriever_output_file: str,
    tables_file: str,
    output_file: str,
    merge_mode: str = "union",
    num_samples: int = 1
):
    """合并检索器输出"""
    with open(retriever_output_file, 'r') as f:
        outputs = json.load(f)

    with open(tables_file, 'r') as f:
        tables = json.load(f)

    db_map = {db['db_id']: db for db in tables}

    merged = []
    for i in range(0, len(outputs), num_samples):
        group = outputs[i:i+num_samples]
        db_id = group[0]['db_id']
        db_info = db_map[db_id]

        if merge_mode == "union":
            merged_indices = set()
            for item in group:
                pred_indices = parse_retriever_output(item['pred_text'], db_info)
                merged_indices.update(pred_indices)

            # 转换为列名列表
            merged_cols = [db_info['column_names_original'][idx] for idx in sorted(merged_indices)]
            merged_output = str(merged_cols)
        else:
            # majority voting逻辑
            from collections import Counter
            all_indices = []
            for item in group:
                pred_indices = parse_retriever_output(item['pred_text'], db_info)
                all_indices.extend(pred_indices)

            counter = Counter(all_indices)
            threshold = num_samples // 2
            merged_indices = {idx for idx, cnt in counter.items() if cnt > threshold}
            merged_cols = [db_info['column_names_original'][idx] for idx in sorted(merged_indices)]
            merged_output = str(merged_cols)

        merged.append({
            "db_id": db_id,
            "question": group[0]['question'],
            "merged_output": merged_output,
            "num_samples": len(group)
        })

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"合并 {len(outputs)} -> {len(merged)} 条检索结果 -> {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="union", choices=["union", "maj"])
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    merge_retriever_outputs(args.input, args.tables, args.output, args.mode, args.num_samples)
