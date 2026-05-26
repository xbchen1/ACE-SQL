"""
Retriever output parsing and Generator prompt building utilities.
"""

import re
from typing import Dict, List, Set, Optional

from utils.schema_utils import (
    select_columns_from_list,
    ensure_necessary_columns,
)
from utils.generate_prompts_utils import (
    obtain_db_details,
    INPUT_PROMPT_TEMPLATE,
)


def parse_column_list(output: str) -> List[str]:
    """
    Parse retriever output to extract column name list.
    Simple regex: find last [...] in text.
    """
    if not output:
        return []

    # Find last [...] pattern
    list_pattern = re.compile(r'\[([^\]]+)\]')
    matches = list(list_pattern.finditer(output))
    if not matches:
        return []

    list_content = matches[-1].group(1)
    columns = [
        col.strip().strip('"').strip("'")
        for col in list_content.split(',')
        if col.strip()
    ]
    return columns


def parse_retriever_output(text: str, db_info: dict, add_primary_keys: bool = False) -> Set[int]:
    """
    Parse [table.col, ...] from retriever output into column index set.
    Returns empty set if parsing fails (no fallback at individual sample level).

    Args:
        text: Retriever output text
        db_info: Database schema info
        add_primary_keys: Whether to add primary/foreign keys (default: False)
    """
    column_list = parse_column_list(text)
    if not column_list:
        return set()

    pred_indices = select_columns_from_list(db_info, column_list, strict_match=True)

    if not pred_indices:
        return set()

    # Only add primary/foreign keys if explicitly requested
    if add_primary_keys:
        pred_indices = ensure_necessary_columns(db_info, pred_indices, add_primary_keys=True)

    return pred_indices


def build_generator_prompt(
    question: str,
    evidence: str,
    db_info: dict,
    pred_indices: Set[int],
    sampled_values: dict,
    relavant_values: dict = None,
) -> str:
    """
    Build pruned schema DDL + generator prompt.
    Uses obtain_db_details with selected column indices.
    """
    if relavant_values is None:
        relavant_values = {}

    # Build question with evidence
    if evidence and evidence.strip():
        full_question = evidence.strip() + "\n" + question.strip()
    else:
        full_question = question.strip()

    # Generate pruned DDL schema
    db_details = obtain_db_details(
        db_info=db_info,
        sampled_db_values_dict=sampled_values,
        relavant_db_values_dict=relavant_values,
        used_column_idx_list=pred_indices,
    )

    # Fill in the generator prompt template
    prompt = INPUT_PROMPT_TEMPLATE.format(
        db_engine="SQLite",
        db_details=db_details,
        question=full_question,
    )
    return prompt
