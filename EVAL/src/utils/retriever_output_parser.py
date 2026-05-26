"""Shared retriever output parsing helpers."""

import re
from typing import Dict, List

from .schema_utils import select_columns_from_list, ensure_necessary_columns


def parse_retriever_output_to_columns(output: str, db_info: Dict) -> List[int]:
    """Parse retriever output into selected schema column indices."""
    if not output or not db_info:
        return []

    code_block_pattern = re.compile(
        r'```(?:python|sql|text)?\s*\n?\s*\[([^\]]*)\]\s*\n?```',
        re.DOTALL
    )
    match = code_block_pattern.search(output)

    if not match:
        list_pattern = re.compile(r'\[([^\]]+)\]')
        match = list_pattern.search(output)

    if match:
        list_content = match.group(1)
        columns = [
            col.strip().strip('"').strip("'")
            for col in list_content.split(',')
            if col.strip()
        ]
        column_indices = select_columns_from_list(db_info, columns)
        if column_indices:
            column_indices = ensure_necessary_columns(db_info, column_indices)
            return list(column_indices)

    if 'CREATE TABLE' in output.upper():
        table_pattern = re.compile(r'CREATE\s+TABLE\s+[`"]?(\w+)[`"]?', re.IGNORECASE)
        tables = table_pattern.findall(output)
        if tables:
            selected_indices = set()
            lower_tables = {table.lower() for table in tables}
            for idx, (table_idx, _) in enumerate(db_info.get("column_names_original", [])):
                if table_idx < 0:
                    continue
                table_name = db_info["table_names_original"][table_idx]
                if table_name.lower() in lower_tables:
                    selected_indices.add(idx)
            if selected_indices:
                selected_indices = ensure_necessary_columns(db_info, selected_indices)
                return list(selected_indices)

    return []
