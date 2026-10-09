"""Execution reward and SQL-column attribution, independent of the trainer."""
import os
import re
from typing import Optional, Tuple, Set
from ace_sql.schema.sql_columns import extract_sql_columns_from_db
from ace_sql.data.paths import resolve_db_path
from .execution import execute_sql_single
from .grading import grade

SQL_EXEC_TIMEOUT = 30
LENGTH_FREE_TOKENS = 512
LENGTH_CAP_TOKENS = 2048
MAX_LENGTH_PENALTY = 0.5

def compute_length_penalty(resp_len: Optional[int]) -> float:
    """Linear penalty: 0 up to 512 tokens, then down to -0.5 at 2048 tokens."""
    if resp_len is None:
        return 0.0
    capped_len = min(max(int(resp_len), 0), LENGTH_CAP_TOKENS)
    if capped_len <= LENGTH_FREE_TOKENS:
        return 0.0
    span = max(LENGTH_CAP_TOKENS - LENGTH_FREE_TOKENS, 1)
    return -MAX_LENGTH_PENALTY * ((capped_len - LENGTH_FREE_TOKENS) / span)

def extract_sql_from_response(solution_str: str) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Extract SQL from generator response.
    Requires <think>...</think> tag and ```sql ... ``` block.
    Returns: (pred_sql, answer_text, think_text, format_correct)
    """
    # Must have <think> tag — hard requirement
    has_think = '<think>' in solution_str and '</think>' in solution_str
    if not has_think:
        return None, None, None, False

    # Extract think text for diagnostics and format validation.
    think_match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else None

    # Answer = everything after last </think>
    answer_text = solution_str.split('</think>')[-1].strip()

    # Extract SQL only from answer_text (after </think>), not from think block
    sql_matches = list(re.finditer(r'```sql(.*?)```', answer_text, re.DOTALL | re.IGNORECASE))
    if not sql_matches:
        return None, answer_text, think_text, False

    pred_sql = sql_matches[-1].group(1).strip()
    return pred_sql, answer_text, think_text, True

def compute_sql_reward(solution_str: str, ground_truth: dict, gen_resp_len: Optional[int] = None,
                       length_penalty_enabled: bool = True) -> Tuple[float, float, dict]:
    """
    Strict two-bin SQL reward:
      - correct SQL with the required response format: 1 plus length penalty
      - malformed, unexecutable, or executable but incorrect SQL: 0

    Returns: (total_score, score_without_length_term, detail). SQL length is
    kept as a diagnostic only and does not affect the score.
    """
    ZERO_REWARD = 0.0
    MATCH_REWARD = 1.0

    gold_sql = re.sub(r'\s+', ' ', ground_truth.get('sql', ''))
    db_path = resolve_db_path(
        ground_truth.get('db_path', ''),
        ground_truth.get('db_id', ''),
    )

    pred_sql, _answer_text, _think_text, format_correct = extract_sql_from_response(solution_str)

    raw_length_score = compute_length_penalty(gen_resp_len) if length_penalty_enabled else 0.0

    if not format_correct or not pred_sql:
        sql_detail = {
            "format_score": ZERO_REWARD,
            "exec_score": 0.0,
            "exec_status": "N/A",
            "result_score": 0.0,
            "length_score": 0.0,
            "raw_length_score": raw_length_score,
            "gen_resp_len": 0 if gen_resp_len is None else int(gen_resp_len),
        }
        return ZERO_REWARD, ZERO_REWARD, sql_detail

    format_score = 0.0
    pred_sql = re.sub(r'\s+', ' ', pred_sql)

    exec_score = 0.0
    result_score = 0.0
    exec_status = 'Unexecutable'
    try:
        gold_result, _gold_error = execute_sql_single(gold_sql, db_path, SQL_EXEC_TIMEOUT)
        if gold_result is None:
            exec_status = 'Gold Error'
        else:
            pred_result, _pred_error = execute_sql_single(pred_sql, db_path, SQL_EXEC_TIMEOUT)
            if pred_result is None:
                exec_status = 'Unexecutable'
            else:
                is_correct, _ = grade(list(gold_result), list(pred_result), grading_method="ordered")
                exec_status = 'Match' if is_correct else 'Mismatch'
    except Exception:
        exec_status = 'Unexecutable'

    if exec_status in ['Unexecutable', 'Gold Error']:
        total_no_len = ZERO_REWARD
    elif exec_status == 'Mismatch':
        total_no_len = ZERO_REWARD
    elif exec_status == 'Match':
        result_score = MATCH_REWARD
        total_no_len = MATCH_REWARD
    else:
        total_no_len = ZERO_REWARD

    length_score = raw_length_score if exec_status == 'Match' else 0.0
    total = total_no_len + length_score
    sql_detail = {
        "format_score": format_score,
        "exec_score": exec_score,
        "exec_status": exec_status,
        "result_score": result_score,
        "length_score": length_score,
        "raw_length_score": raw_length_score,
        "gen_resp_len": 0 if gen_resp_len is None else int(gen_resp_len),
    }
    return total, total_no_len, sql_detail

def extract_columns_from_sql(sql_text: str, db_path: str, db_info: dict) -> Tuple[bool, Set[int], str]:
    """Extract column indices used in SQL."""
    db_path = resolve_db_path(db_path, db_info.get("db_id"))
    if not db_path or not os.path.exists(db_path):
        return False, set(), "db_path not found"
    try:
        columns_dict = extract_sql_columns_from_db(db_path, sql_text)
        if not columns_dict:
            return False, set(), "extract failed"
        table_names = db_info.get("table_names_original", [])
        column_names = db_info.get("column_names_original", [])
        col_indices = set()
        for tbl, cols in columns_dict.items():
            for col_name in cols:
                for idx, (tidx, cname) in enumerate(column_names):
                    if tidx < 0:
                        continue
                    if (table_names[tidx].lower() == tbl.lower() and
                        cname.lower() == col_name.lower()):
                        col_indices.add(idx)
                        break
        return True, col_indices, ""
    except Exception as e:
        return False, set(), str(e)
