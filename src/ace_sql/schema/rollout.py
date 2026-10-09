"""Schema aggregation and shared generator prompts for two-pass rollouts."""
import json
from ace_sql.data.rl import as_object, validate_metadata
from ace_sql.schema.retrieval import parse_retriever_output, build_generator_prompt
GROUP_SIZE = 8

def role_group_id(row_key: str, role: str) -> str:
    """Distinct stable groups keep retriever and generator normalization apart."""
    if role not in {"retriever", "generator"}:
        raise ValueError(f"unknown role: {role}")
    return json.dumps([row_key, role], ensure_ascii=False, separators=(",", ":"))


def union_columns(ret_texts: list[str], db_info: dict, group_size: int = GROUP_SIZE) -> set[int]:
    """Keep every parsed column proposed by any retriever sample (Algorithm 1)."""
    if group_size <= 0 or len(ret_texts) != group_size:
        raise ValueError(f"expected {group_size} retriever samples, got {len(ret_texts)}")
    return set().union(*(parse_retriever_output(text, db_info) for text in ret_texts))


def generator_prompt(ret_texts: list[str], metadata: dict,
                     group_size: int = GROUP_SIZE) -> tuple[list[dict], set[int]]:
    """One union schema shared by all N generator samples, with no fallback."""
    _, extra = validate_metadata(metadata)
    db_info = as_object(extra["db_info"], "db_info")
    selected = union_columns(ret_texts, db_info, group_size)
    text = build_generator_prompt(
        question=extra["question"], evidence=extra.get("evidence", ""),
        db_info=db_info, pred_indices=selected,
        sampled_values=as_object(extra["sampled_values"], "sampled_values"),
        relavant_values=as_object(extra["relavant_values"], "relavant_values"),
    )
    return [{"role": "user", "content": text}], selected
