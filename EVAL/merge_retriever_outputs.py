"""
合并多个检索器输出
支持两种模式：
  - union (U): 取并集
  - maj: 对每个列进行 majority voting

用于 maj-voting 模式
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import GENERATE_PROMPTS_DIR

# 引用 schema_utils 中的方法
from src.utils.schema_utils import (
    parse_column_list as parse_schema_column_list,
    select_columns_from_list,
    ensure_necessary_columns
)


def parse_json_format(output: str) -> Dict[str, List[str]]:
    """
    解析 JSON 格式的检索器输出 (retriever_v2 格式)

    格式示例:
    ```json
    {
        "table1": ["column1", "column2"],
        "table2": ["column1", "column2"]
    }
    ```

    Args:
        output: 检索器生成的输出

    Returns:
        表名到列名列表的字典，解析失败返回空字典
    """
    if not output:
        return {}

    # 尝试提取 ```json ... ``` 代码块
    json_block_pattern = re.compile(r'```json\s*\n?(.*?)\n?```', re.DOTALL)
    match = json_block_pattern.search(output)

    if match:
        json_str = match.group(1).strip()
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 尝试直接解析整个输出中的 JSON 对象
    # 查找 { ... } 模式
    brace_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    matches = brace_pattern.findall(output)

    for json_str in matches:
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                # 验证是否是表-列格式
                is_valid = all(
                    isinstance(v, list) and all(isinstance(c, str) for c in v)
                    for v in result.values()
                )
                if is_valid:
                    return result
        except json.JSONDecodeError:
            continue

    return {}


def parse_column_list(output: str) -> List[str]:
    """
    解析检索器输出，提取列名列表
    兼容两类输入：
      1. 原始模型输出中的代码块列表 ```[...]```
      2. 已抽取过一次的裸列表 [...]

    Args:
        output: 检索器生成的输出

    Returns:
        列名列表 (格式: "table.column")
    """
    return parse_schema_column_list(output)


def merge_column_indices_union(db_info: Dict, parsed_columns_list: List[str], add_primary_keys: bool = True) -> Set[int]:
    """
    合并多个检索结果的列索引，取并集

    Args:
        db_info: 数据库信息
        parsed_columns_list: 多个检索结果的解析列列表
        add_primary_keys: 是否添加主键

    Returns:
        合并后的列索引集合
    """
    merged_indices = set()

    for parsed_output in parsed_columns_list:
        columns = parse_column_list(parsed_output)
        if columns:
            # 使用严格匹配模式，确保每个预测项都对应schema中的列
            indices = select_columns_from_list(db_info, columns, strict_match=True)
            merged_indices.update(indices)

    # 确保包含必要的列（外键等，不额外添加主键由add_primary_keys控制）
    if merged_indices:
        merged_indices = ensure_necessary_columns(db_info, merged_indices, add_primary_keys=add_primary_keys)

    return merged_indices


def merge_column_indices_maj(
    db_info: Dict,
    parsed_columns_list: List[str],
    add_primary_keys: bool = True,
    vote_threshold: float = 0.5,
) -> Set[int]:
    """
    合并多个检索结果的列索引，使用按列 majority voting。

    每个 rollout 先解析成列索引集合；同一个列在一个 rollout 中最多计 1 票。
    默认阈值 0.5 表示列需要出现在 ceil(num_rollouts * 0.5) 个 rollout 中。

    Args:
        db_info: 数据库信息
        parsed_columns_list: 多个检索结果的解析列列表
        add_primary_keys: 是否添加主键

    Returns:
        投票选出的列索引集合
    """
    if not parsed_columns_list:
        return set()

    if not 0 < vote_threshold <= 1:
        raise ValueError(f"vote_threshold must be in (0, 1], got {vote_threshold}")

    vote_counts = Counter()
    for parsed_output in parsed_columns_list:
        columns = parse_column_list(parsed_output)
        if columns:
            # 使用严格匹配模式，确保每个预测项都对应schema中的列
            indices = select_columns_from_list(db_info, columns, strict_match=True)
            if indices:
                vote_counts.update(set(indices))

    if not vote_counts:
        return set()

    cutoff = max(1, math.ceil(len(parsed_columns_list) * vote_threshold))
    result = {idx for idx, votes in vote_counts.items() if votes >= cutoff}
    if result:
        result = ensure_necessary_columns(db_info, result, add_primary_keys=add_primary_keys)

    return result


def merge_column_indices(
    db_info: Dict,
    parsed_columns_list: List[str],
    mode: str = "union",
    add_primary_keys: bool = True,
    maj_vote_threshold: float = 0.5,
) -> Set[int]:
    """
    合并多个检索结果的列索引

    Args:
        db_info: 数据库信息
        parsed_columns_list: 多个检索结果的解析列列表
        mode: 合并模式，"union" 或 "maj"
        add_primary_keys: 是否添加主键

    Returns:
        合并后的列索引集合
    """
    if mode == "maj":
        return merge_column_indices_maj(
            db_info,
            parsed_columns_list,
            add_primary_keys=add_primary_keys,
            vote_threshold=maj_vote_threshold,
        )
    else:
        return merge_column_indices_union(db_info, parsed_columns_list, add_primary_keys=add_primary_keys)


def merge_retriever_outputs(
    input_file: str,
    output_file: str,
    mode: str = "union",
    add_primary_keys: bool = True,
    maj_vote_threshold: float = 0.5,
):
    """
    合并检索器的多个输出

    Args:
        input_file: 检索器输出文件路径（包含 parsed_columns_list）
        output_file: 输出文件路径
        mode: 合并模式，"union" (取并集) 或 "maj" (majority voting)
        add_primary_keys: 是否添加主键
    """
    mode_desc = "取并集" if mode == "union" else "majority voting"
    print(f"加载检索器输出: {input_file}")
    print(f"合并模式: {mode} ({mode_desc})")
    print(f"添加主键: {add_primary_keys}")
    if mode == "maj":
        print(f"列投票阈值: ceil(num_rollouts * {maj_vote_threshold})")

    with open(input_file, 'r', encoding='utf-8') as f:
        retriever_data = json.load(f)

    merged_results = []
    fallback_count = 0

    print(f"合并 {len(retriever_data)} 个样本的检索结果...")

    for idx, sample in enumerate(retriever_data):
        db_info = json.loads(sample.get("db_info_raw", "{}"))

        # 获取多个检索结果
        parsed_columns_list = sample.get("parsed_columns_list", [])

        if not parsed_columns_list:
            # 兼容单样本模式
            parsed_columns = sample.get("parsed_columns", "")
            if parsed_columns:
                parsed_columns_list = [parsed_columns]

        # 合并列索引
        merged_indices = merge_column_indices(
            db_info,
            parsed_columns_list,
            mode=mode,
            add_primary_keys=add_primary_keys,
            maj_vote_threshold=maj_vote_threshold,
        )

        if not merged_indices:
            fallback_count += 1

        # 构建输出
        result = {
            **sample,
            "merged_column_indices": sorted(merged_indices) if merged_indices else None,
            "num_retriever_samples": len(parsed_columns_list),
            "retriever_merge_mode": mode,
            "retriever_maj_vote_threshold": maj_vote_threshold if mode == "maj" else None,
            "merged_column_count": len(merged_indices),
            # 保留原始的 retriever_response 字段用于兼容
            "retriever_response": sample.get("retriever_responses", [sample.get("retriever_response", "")])[0] if sample.get("retriever_responses") else sample.get("retriever_response", "")
        }
        merged_results.append(result)

    print(f"合并完成！{fallback_count} 个样本将使用完整 Schema 作为 fallback")

    # 保存结果
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"结果保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="合并检索器输出")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="检索器输出文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="union",
        choices=["union", "maj"],
        help="合并模式: union (取并集) 或 maj (majority voting)，默认 union"
    )
    parser.add_argument(
        "--add_primary_keys",
        action="store_true",
        default=True,
        help="是否添加主键（默认添加）"
    )
    parser.add_argument(
        "--no_add_primary_keys",
        action="store_true",
        help="不添加主键"
    )
    parser.add_argument(
        "--maj_vote_threshold",
        type=float,
        default=0.5,
        help="maj 模式按列投票阈值比例，保留 votes >= ceil(num_rollouts * threshold) 的列，默认 0.5"
    )

    args = parser.parse_args()

    # 处理 add_primary_keys 参数
    add_primary_keys = not args.no_add_primary_keys

    merge_retriever_outputs(
        input_file=args.input_file,
        output_file=args.output_file,
        mode=args.mode,
        add_primary_keys=add_primary_keys,
        maj_vote_threshold=args.maj_vote_threshold,
    )


if __name__ == "__main__":
    main()
