"""
Joint reward computation for two-pass GRPO training.
V4: Empirical attribution-based retriever reward + simple SQL reward.
"""

import re
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from rewards.retriever_utils import parse_column_list
from utils.schema_utils import select_columns_from_list
from utils.Schema_From_SQL import extract_sql_columns_from_db
from utils.path_utils import resolve_db_path
from scripts.sql_utils import execute_sql_single
from scripts.grader import grade

# SQL execution timeout for training (seconds).
SQL_EXEC_TIMEOUT = 30

# Shared response-length penalty for retriever and generator rewards.
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


# ============== SQL Reward Helpers ==============

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


def compute_sql_reward(solution_str: str, ground_truth: dict, gen_resp_len: Optional[int] = None) -> Tuple[float, float, dict]:
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

    raw_length_score = compute_length_penalty(gen_resp_len)

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
                is_correct, _ = grade(list(gold_result), list(pred_result), grading_method="multiset")
                exec_status = 'Match' if is_correct else 'Mismatch'
    except BaseException:
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


# ============== Retriever Reward V4 ==============

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


def compute_empirical_attribution(
    gen_texts: List[str],
    gen_exec_statuses: List[str],
    db_path: str,
    db_info: dict,
    gold_indices: Set[int],
    gold_sql: str,
) -> Tuple[List[Set[int]], Set[int], List[Set[int]]]:
    """
    Compute useful column sets and harmful columns from generator outputs.
    Returns: (all_correct_col_sets, harmful_cols, gen_only_correct_col_sets)
      - all_correct_col_sets: generator Match sets + gold_sql set (for coverage/harmful)
      - gen_only_correct_col_sets: generator Match sets only (for frequency counting)
    """
    gen_only_correct_col_sets = []
    harmful_cols = set()

    for gen_text, exec_status in zip(gen_texts, gen_exec_statuses):
        pred_sql, _, _, _ = extract_sql_from_response(gen_text)
        if not pred_sql:
            continue
        success, col_indices, _ = extract_columns_from_sql(pred_sql, db_path, db_info)
        if not success:
            continue
        if exec_status == "Match":
            gen_only_correct_col_sets.append(col_indices)

    # gold_sql participates in coverage and harmful, but NOT in frequency counting
    all_correct_col_sets = gen_only_correct_col_sets.copy()
    if gold_sql:
        success, gold_sql_cols, _ = extract_columns_from_sql(gold_sql, db_path, db_info)
        if success:
            all_correct_col_sets.append(gold_sql_cols)

    empirical_useful_set = set()
    for col_set in all_correct_col_sets:
        empirical_useful_set.update(col_set)

    for gen_text, exec_status in zip(gen_texts, gen_exec_statuses):
        if exec_status in ["Mismatch", "Unexecutable"]:
            pred_sql, _, _, _ = extract_sql_from_response(gen_text)
            if pred_sql:
                success, wrong_cols, _ = extract_columns_from_sql(pred_sql, db_path, db_info)
                if success:
                    harmful = wrong_cols - gold_indices - empirical_useful_set
                    harmful_cols.update(harmful)

    return all_correct_col_sets, harmful_cols, gen_only_correct_col_sets


def compute_retriever_reward(
    ret_text: str,
    db_info: dict,
    gold_indices: Set[int],
    best_set: Set[int],
    col_weights: Dict[int, float],
    useful_set: Set[int],
    pred_indices: Optional[Set[int]] = None,
    useful_noise_bar: float = 0.1,
) -> tuple:
    """
    V6 retriever reward.  Best-set coverage + two-tier noise treatment.

    Coverage  = |pred ∩ best_set| / |best_set| × 4          [0, 4]
    Irrelevant penalty = (|irrelevant| / |pred|) × (-6)     [-6, 0]
    Useful-noise bonus = avg_w × (useful_n / noise_n) × 4   [0, 4]

    Noise columns = pred − best_set.
    A noise column is *useful* only when its pool weight > useful_noise_bar;
    otherwise it is *irrelevant*.
    Format/empty failures share the same zero-reward floor as non-matches.

    Returns: (reward, ret_detail_dict)
    """
    _fail_detail = {
        "coverage_rate": 0.0, "coverage_score": 0.0,
        "pred_col_cnt": 0,
        "noise_count": 0, "irrelevant_count": 0, "useful_noise_count": 0,
        "irrelevant_penalty": 0.0, "useful_noise_bonus": 0.0,
        "ret_case": "", "ret_length_score": 0.0,
    }

    # --- Format gate ---
    has_think = '<think>' in ret_text and '</think>' in ret_text
    answer_part = ret_text.split('</think>')[-1] if has_think else ret_text
    has_list = bool(re.search(r'\[([^\]]+)\]', answer_part))

    if not has_think or not has_list:
        _fail_detail["ret_case"] = "FORMAT_ERROR"
        return 0.0, _fail_detail

    # --- Parse columns ---
    if pred_indices is None:
        column_list = parse_column_list(ret_text)
        if column_list:
            pred_indices = select_columns_from_list(db_info, column_list, strict_match=True)
        else:
            pred_indices = set()

    if not pred_indices:
        _fail_detail["ret_case"] = "EMPTY"
        return 0.0, _fail_detail

    if not best_set:
        _fail_detail["pred_col_cnt"] = len(pred_indices)
        _fail_detail["ret_case"] = "NO_BEST_SET"
        return 0.0, _fail_detail

    # --- Coverage: fraction of best_set covered × 4 ---
    covered = pred_indices & best_set
    coverage_rate = len(covered) / len(best_set)
    coverage_score = 4.0 * coverage_rate

    # --- Noise classification ---
    noise_cols = pred_indices - best_set
    useful_noise = set()
    irrelevant = set()
    for c in noise_cols:
        if col_weights.get(c, 0.0) > useful_noise_bar:
            useful_noise.add(c)
        else:
            irrelevant.add(c)

    # Irrelevant penalty: proportion of pred that is irrelevant × (-6)
    irrelevant_penalty = (len(irrelevant) / len(pred_indices)) * (-6.0)

    # Useful-noise bonus: avg_weight × (useful_noise / total_noise) × 4
    useful_noise_bonus = 0.0
    if noise_cols and useful_noise:
        avg_w = sum(col_weights.get(c, 0.0) for c in useful_noise) / len(useful_noise)
        useful_ratio = len(useful_noise) / len(noise_cols)
        useful_noise_bonus = avg_w * useful_ratio * 4.0

    total = coverage_score + irrelevant_penalty + useful_noise_bonus

    ret_detail = {
        "coverage_rate": round(coverage_rate, 4),
        "coverage_score": round(coverage_score, 4),
        "pred_col_cnt": len(pred_indices),
        "noise_count": len(noise_cols),
        "irrelevant_count": len(irrelevant),
        "useful_noise_count": len(useful_noise),
        "irrelevant_penalty": round(irrelevant_penalty, 4),
        "useful_noise_bonus": round(useful_noise_bonus, 4),
        "ret_case": f"V6 (cov={coverage_rate:.2f})",
        "ret_length_score": 0.0,
    }

    return total, ret_detail


def _format_gate_ret_text(ret_text: str) -> Tuple[bool, bool]:
    has_think = '<think>' in ret_text and '</think>' in ret_text
    answer_part = ret_text.split('</think>')[-1] if has_think else ret_text
    has_list = bool(re.search(r'\[([^\]]+)\]', answer_part))
    return has_think, has_list


def compute_pool_exact_retriever_reward(
    ret_text: str,
    target_sets: List[Set[int]],
    pred_indices: Optional[Set[int]] = None,
    exact_reward: float = 1.0,
    ret_resp_len: Optional[int] = None,
    response_length: Optional[int] = None,
) -> tuple:
    """
    Sparse retriever reward for the main experiment.

    The target is the highest-frequency column set(s) from the initial pool.
    Any tied max-frequency set is accepted. Format errors, truncated responses,
    and format-correct non-exact responses receive 0. Exact set matches receive
    exact_reward. The shared response length penalty is only applied on exact
    matches.
    """
    raw_length_score = compute_length_penalty(ret_resp_len)
    truncated = (
        ret_resp_len is not None
        and response_length is not None
        and int(ret_resp_len) >= int(response_length)
    )
    fail_detail = {
        "coverage_rate": 0.0, "coverage_score": 0.0,
        "pred_col_cnt": 0,
        "noise_count": 0, "irrelevant_count": 0, "useful_noise_count": 0,
        "irrelevant_penalty": 0.0, "useful_noise_bonus": 0.0,
        "ret_case": "", "ret_length_score": 0.0,
        "ret_length_score_raw": raw_length_score,
        "ret_resp_len": 0 if ret_resp_len is None else int(ret_resp_len),
        "ret_truncated": bool(truncated),
        "pool_target_col_cnt": 0,
        "pool_target_num_ties": len(target_sets),
    }

    has_think, has_list = _format_gate_ret_text(ret_text)
    if not has_think or not has_list:
        fail_detail["ret_case"] = "FORMAT_ERROR"
        fail_detail["coverage_score"] = 0.0
        return 0.0, fail_detail

    if truncated:
        fail_detail["ret_case"] = "TRUNCATED"
        fail_detail["coverage_score"] = 0.0
        return 0.0, fail_detail

    pred_indices = set(pred_indices or set())
    fail_detail["pred_col_cnt"] = len(pred_indices)
    if not pred_indices:
        fail_detail["ret_case"] = "EMPTY"
        return 0.0, fail_detail

    if not target_sets:
        fail_detail["ret_case"] = "NO_POOL_TARGET"
        return 0.0, fail_detail

    def _metric_key(target: Set[int]):
        covered = len(pred_indices & target)
        coverage = covered / max(len(target), 1)
        noise = len(pred_indices - target)
        return (coverage, -noise, -abs(len(pred_indices) - len(target)))

    metric_target = max(target_sets, key=_metric_key)
    covered = pred_indices & metric_target
    noise = pred_indices - metric_target
    coverage_rate = len(covered) / max(len(metric_target), 1)
    exact = any(pred_indices == target for target in target_sets)
    base_reward = float(exact_reward if exact else 0.0)
    length_score = raw_length_score if exact else 0.0
    reward = base_reward + length_score

    ret_detail = {
        "coverage_rate": round(coverage_rate, 4),
        "coverage_score": base_reward,
        "pred_col_cnt": len(pred_indices),
        "noise_count": len(noise),
        "irrelevant_count": len(noise),
        "useful_noise_count": 0,
        "irrelevant_penalty": 0.0,
        "useful_noise_bonus": 0.0,
        "ret_case": "POOL_EXACT_MATCH" if exact else "POOL_EXACT_MISS",
        "ret_length_score": length_score,
        "ret_length_score_raw": raw_length_score,
        "ret_resp_len": 0 if ret_resp_len is None else int(ret_resp_len),
        "ret_truncated": bool(truncated),
        "pool_target_col_cnt": len(metric_target),
        "pool_target_num_ties": len(target_sets),
    }
    return reward, ret_detail


# ============== Reward Detail Aggregation ==============

def aggregate_reward_details(details: List[dict], N: int, prefix: str = "train") -> dict:
    """
    Aggregate per-sample reward details into WandB-ready metrics.

    Args:
        details: List of per-sample detail dicts from compute_rewards().
        N: Number of samples per group (for best-of-N metrics).
        prefix: Metric prefix ("train" or "val").

    Returns:
        Flat dict of aggregated metrics.
    """
    metrics = {}
    n_total = len(details)
    if n_total == 0:
        return metrics

    # --- SQL Reward Decomposition ---
    format_scores = [d.get("format_score", 0.0) for d in details]
    exec_scores = [d.get("exec_score", 0.0) for d in details]
    result_scores = [d.get("result_score", 0.0) for d in details]
    length_scores = [d.get("length_score", 0.0) for d in details]

    metrics[f"{prefix}/sql/format_score_mean"] = float(np.mean(format_scores))
    metrics[f"{prefix}/sql/exec_score_mean"] = float(np.mean(exec_scores))
    metrics[f"{prefix}/sql/result_score_mean"] = float(np.mean(result_scores))
    metrics[f"{prefix}/sql/length_score_mean"] = float(np.mean(length_scores))

    # Execution status distribution
    exec_statuses = [d.get("exec_status", "N/A") for d in details]
    status_map = {"Match": "Match", "Mismatch": "Mismatch",
                  "Unexecutable": "Unexecutable", "Gold Error": "GoldError", "N/A": "NA"}
    for raw_status, key_name in status_map.items():
        count = sum(1 for s in exec_statuses if s == raw_status)
        metrics[f"{prefix}/sql/exec_{key_name}"] = count / n_total

    # --- Retriever Reward Decomposition (V6) ---
    coverage_scores = [d.get("coverage_score", 0.0) for d in details]
    coverage_rates = [d.get("coverage_rate", 0.0) for d in details]
    noise_counts = [d.get("noise_count", 0) for d in details]
    irrelevant_counts = [d.get("irrelevant_count", 0) for d in details]
    useful_noise_counts = [d.get("useful_noise_count", 0) for d in details]
    irrelevant_penalties = [d.get("irrelevant_penalty", 0.0) for d in details]
    useful_noise_bonuses = [d.get("useful_noise_bonus", 0.0) for d in details]
    ret_length_scores = [d.get("ret_length_score", 0.0) for d in details]
    ret_resp_lens = [d.get("ret_resp_len", 0) for d in details]

    metrics[f"{prefix}/ret/coverage_score_mean"] = float(np.mean(coverage_scores))
    metrics[f"{prefix}/ret/coverage_rate_mean"] = float(np.mean(coverage_rates))
    metrics[f"{prefix}/ret/coverage_rate_full"] = sum(1 for c in coverage_rates if c >= 1.0) / n_total
    metrics[f"{prefix}/ret/noise_count_mean"] = float(np.mean(noise_counts))
    metrics[f"{prefix}/ret/irrelevant_count_mean"] = float(np.mean(irrelevant_counts))
    metrics[f"{prefix}/ret/useful_noise_count_mean"] = float(np.mean(useful_noise_counts))
    metrics[f"{prefix}/ret/irrelevant_penalty_mean"] = float(np.mean(irrelevant_penalties))
    metrics[f"{prefix}/ret/useful_noise_bonus_mean"] = float(np.mean(useful_noise_bonuses))
    metrics[f"{prefix}/ret/length_score_mean"] = float(np.mean(ret_length_scores))
    metrics[f"{prefix}/ret/resp_len_mean"] = float(np.mean(ret_resp_lens))
    # V5 compat keys (harmful → irrelevant for monitoring continuity)
    metrics[f"{prefix}/ret/harmful_count_mean"] = float(np.mean(irrelevant_counts))
    metrics[f"{prefix}/ret/harmful_penalty_mean"] = float(np.mean(irrelevant_penalties))
    metrics[f"{prefix}/ret/noise_penalty_mean"] = float(np.mean(irrelevant_penalties))

    # --- Column Selection Statistics ---
    pred_col_counts = [d.get("pred_col_cnt", 0) for d in details]
    gold_col_counts = [d.get("gold_col_cnt", 0) for d in details]

    metrics[f"{prefix}/ret/pred_col_count_mean"] = float(np.mean(pred_col_counts))
    metrics[f"{prefix}/ret/gold_col_count_mean"] = float(np.mean(gold_col_counts))
    mean_gold = float(np.mean(gold_col_counts))
    metrics[f"{prefix}/ret/pred_to_gold_ratio"] = (
        float(np.mean(pred_col_counts)) / mean_gold if mean_gold > 0 else 0.0
    )

    # --- Retriever Case Distribution ---
    ret_cases = [d.get("ret_case", "") for d in details]
    metrics[f"{prefix}/ret/case_empty_ratio"] = sum(1 for c in ret_cases if c == "EMPTY") / n_total
    metrics[f"{prefix}/ret/case_format_error_ratio"] = sum(1 for c in ret_cases if c == "FORMAT_ERROR") / n_total
    metrics[f"{prefix}/ret/case_truncated_ratio"] = sum(1 for c in ret_cases if c == "TRUNCATED") / n_total

    # --- A1: Reward Distribution Statistics ---
    sql_rewards = [d.get("sql_reward", 0.0) for d in details]
    ret_rewards = [d.get("ret_reward", 0.0) for d in details]

    metrics[f"{prefix}/sql/reward_mean"] = float(np.mean(sql_rewards))
    metrics[f"{prefix}/sql/reward_std"] = float(np.std(sql_rewards))
    metrics[f"{prefix}/sql/reward_max"] = float(np.max(sql_rewards))
    metrics[f"{prefix}/sql/reward_min"] = float(np.min(sql_rewards))

    metrics[f"{prefix}/ret/reward_mean"] = float(np.mean(ret_rewards))
    metrics[f"{prefix}/ret/reward_std"] = float(np.std(ret_rewards))
    metrics[f"{prefix}/ret/reward_max"] = float(np.max(ret_rewards))
    metrics[f"{prefix}/ret/reward_min"] = float(np.min(ret_rewards))

    # --- A3: Retriever Exact Match Rate ---
    # V6: exact match = coverage 100% AND no noise columns at all
    exact_matches = sum(
        1 for d in details
        if d.get("coverage_rate", 0.0) >= 1.0
        and d.get("noise_count", 0) == 0
    )
    metrics[f"{prefix}/ret/exact_match_rate"] = exact_matches / n_total

    # --- A4: Irrelevant Column Presence Ratio ---
    has_irrelevant = sum(1 for d in details if d.get("irrelevant_count", 0) > 0)
    metrics[f"{prefix}/ret/has_harmful_ratio"] = has_irrelevant / n_total

    # --- Per-Group Statistics ---
    n_groups = n_total // N if N > 0 else 0
    if n_groups > 0 and N > 0:
        bon_correct = 0
        group_sql_stds = []
        group_ret_stds = []
        group_unique_sets = []
        for g in range(n_groups):
            group_slice = slice(g * N, (g + 1) * N)
            group_statuses = exec_statuses[group_slice]
            if any(s == "Match" for s in group_statuses):
                bon_correct += 1

            # A2: Group-internal reward diversity
            g_sql = sql_rewards[group_slice]
            g_ret = ret_rewards[group_slice]
            group_sql_stds.append(float(np.std(g_sql)))
            group_ret_stds.append(float(np.std(g_ret)))

            # A2: Group-internal column selection diversity
            g_pred_sets = []
            for d in details[group_slice]:
                # Use pred_col_cnt > 0 as proxy for non-empty prediction;
                # build a hashable key from coverage_rate + noise_count + pred_col_cnt
                # to approximate unique column sets without raw indices.
                ret_case = d.get("ret_case", "")
                if ret_case not in ("EMPTY", "FORMAT_ERROR", "NO_CORRECT_SQL", "NO_COVERAGE"):
                    key = (
                        round(d.get("coverage_rate", 0.0), 4),
                        d.get("pred_col_cnt", 0),
                        d.get("noise_count", 0),
                        d.get("irrelevant_count", 0),
                    )
                    g_pred_sets.append(key)
            group_unique_sets.append(len(set(g_pred_sets)) if g_pred_sets else 0)

        metrics[f"{prefix}/group/bon_accuracy"] = bon_correct / n_groups

        # A2: Group diversity metrics
        metrics[f"{prefix}/group/sql_reward_std_mean"] = float(np.mean(group_sql_stds))
        metrics[f"{prefix}/group/ret_reward_std_mean"] = float(np.mean(group_ret_stds))
        metrics[f"{prefix}/group/ret_unique_sets_mean"] = float(np.mean(group_unique_sets))

    return metrics


# ============== Joint Reward Manager ==============

class JointRewardManager:
    """Computes paired retriever + generator rewards with historical column set pool."""

    def __init__(self, max_workers: int = 8, pool_gamma: float = 0.5,
                 initial_pool_path: Optional[str] = None,
                 useful_noise_bar: float = 0.1,
                 retriever_reward_mode: str = "pool_exact",
                 pool_exact_reward: float = 1.0):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Historical correct column set pool: {question_key: {frozenset: float}}
        # Accumulates correct SQL column sets across encounters.
        # Historical counts are decayed before adding current matched rollouts.
        self.col_set_pool: Dict[str, Dict[frozenset, float]] = {}
        self.pool_gamma = pool_gamma
        self.useful_noise_bar = useful_noise_bar
        self.retriever_reward_mode = retriever_reward_mode
        self.pool_exact_reward = float(pool_exact_reward)
        if self.retriever_reward_mode not in {"v6", "pool_exact"}:
            raise ValueError(
                f"Unsupported retriever_reward_mode={self.retriever_reward_mode!r}; "
                "expected 'v6' or 'pool_exact'"
            )

        # Load pre-built pool from JSON if provided
        if initial_pool_path:
            self._load_pool(initial_pool_path)

    def _load_pool(self, path: str):
        """Load pre-built col_set_pool from JSON file.

        Expected format: {question_key: {col_set_str: freq, ...}, ...}
        where col_set_str is a comma-separated string of integer column indices.
        """
        with open(path, "r") as f:
            raw_pool = json.load(f)
        for question_key, freq_dict in raw_pool.items():
            self.col_set_pool[question_key] = {
                frozenset(int(x) for x in k.split(",") if x): v
                for k, v in freq_dict.items()
            }
        print(f"[JointRewardManager] Loaded initial pool: "
              f"{len(self.col_set_pool)} questions, "
              f"{sum(len(v) for v in self.col_set_pool.values())} column sets",
              flush=True)

    def save_pool(self, path: str):
        """Save current col_set_pool to JSON file."""
        serializable = {}
        for question_key, freq_dict in self.col_set_pool.items():
            serializable[question_key] = {
                ",".join(str(x) for x in sorted(k)): v
                for k, v in freq_dict.items()
            }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[JointRewardManager] Saved pool: "
              f"{len(self.col_set_pool)} questions to {path}",
              flush=True)

    @staticmethod
    def _question_key(ground_truth: dict, db_id: str) -> str:
        return f"{db_id}||{ground_truth.get('question', '')}"

    @staticmethod
    def _target_sets_from_pool(pool_freq: Dict[frozenset, float]) -> List[Set[int]]:
        positive_items = [(set_key, float(freq)) for set_key, freq in pool_freq.items() if float(freq) > 0]
        if not positive_items:
            return []
        max_freq = max(freq for _, freq in positive_items)
        return [set(set_key) for set_key, freq in positive_items if abs(freq - max_freq) < 1e-9]

    def _pool_freq_for_question(self, question_key: str) -> Dict[frozenset, float]:
        pool_freq = {}
        if question_key in self.col_set_pool:
            for set_key, freq in self.col_set_pool[question_key].items():
                pool_freq[set_key] = float(freq) * self.pool_gamma
        return pool_freq

    def compute_retriever_only_rewards(
        self,
        ret_texts: List[str],
        ground_truths: List[dict],
        extra_infos: List[dict],
        group_ids: Optional[List[str]] = None,
        repeat_n: Optional[int] = None,
        ret_resp_lens: Optional[List[int]] = None,
        ret_response_length: Optional[int] = None,
    ) -> Tuple[List[float], List[dict]]:
        """Compute sparse pool-exact rewards without generator rollout."""
        n = len(ret_texts)
        if ret_resp_lens is not None and len(ret_resp_lens) != n:
            raise ValueError(f"ret_resp_lens length {len(ret_resp_lens)} does not match number of samples {n}")
        if ret_resp_lens is None:
            ret_resp_lens = [None] * n
        if group_ids is not None:
            if len(group_ids) != n:
                raise ValueError(f"group_ids length {len(group_ids)} does not match number of samples {n}")
            grouped_indices = {}
            ordered_group_ids = []
            for idx, group_id in enumerate(group_ids):
                key = str(group_id)
                if key not in grouped_indices:
                    grouped_indices[key] = []
                    ordered_group_ids.append(key)
                grouped_indices[key].append(idx)
            group_index_lists = [grouped_indices[key] for key in ordered_group_ids]
        else:
            if repeat_n is None or repeat_n <= 0:
                raise ValueError(
                    "compute_retriever_only_rewards requires explicit grouping: pass group_ids, "
                    "or pass a positive repeat_n."
                )
            if n % repeat_n != 0:
                raise ValueError(f"Invalid repeat_n={repeat_n} for {n} samples")
            group_index_lists = [
                list(range(start_idx, start_idx + repeat_n))
                for start_idx in range(0, n, repeat_n)
            ]

        rewards = [0.0] * n
        details = [None] * n
        for sample_indices in group_index_lists:
            first_idx = sample_indices[0]
            gt = ground_truths[first_idx]
            extra = extra_infos[first_idx]
            db_id = gt.get("db_id", "")
            db_info = json.loads(extra.get("db_info", "{}")) if isinstance(extra.get("db_info"), str) else extra.get("db_info", {})
            question_key = self._question_key(gt, db_id)
            pool_freq = self._pool_freq_for_question(question_key)
            target_sets = self._target_sets_from_pool(pool_freq)
            gold_indices = set(gt.get("gold_column_indices", []))

            for idx in sample_indices:
                column_list = parse_column_list(ret_texts[idx])
                pred_indices = select_columns_from_list(db_info, column_list, strict_match=True) if column_list else set()
                ret_reward, ret_detail = compute_pool_exact_retriever_reward(
                    ret_text=ret_texts[idx],
                    target_sets=target_sets,
                    pred_indices=pred_indices,
                    exact_reward=self.pool_exact_reward,
                    ret_resp_len=ret_resp_lens[idx],
                    response_length=ret_response_length,
                )
                rewards[idx] = ret_reward
                details[idx] = {
                    "db_id": db_id,
                    "gold_sql": gt.get("sql", ""),
                    "pred_sql": None,
                    "fmt_ok": False,
                    "sql_reward": 0.0,
                    "sql_no_len": 0.0,
                    "ret_reward": ret_reward,
                    "gold_col_cnt": len(gold_indices),
                    "format_score": 0.0,
                    "exec_score": 0.0,
                    "exec_status": "N/A",
                    "result_score": 0.0,
                    "length_score": 0.0,
                    "gen_resp_len": 0,
                    **ret_detail,
                }
        return rewards, details

    def compute_generator_only_rewards(
        self,
        gen_texts: List[str],
        ground_truths: List[dict],
        gen_resp_lens: Optional[List[int]] = None,
    ) -> Tuple[List[float], List[dict]]:
        """Compute SQL rewards for generator-only alternating epochs."""
        if gen_resp_lens is None:
            gen_resp_lens = [None] * len(gen_texts)
        if len(gen_resp_lens) != len(gen_texts):
            raise ValueError("gen_resp_lens length does not match gen_texts")

        futures = [
            self.executor.submit(compute_sql_reward, gen_text, gt, gen_resp_len)
            for gen_text, gt, gen_resp_len in zip(gen_texts, ground_truths, gen_resp_lens)
        ]
        rewards = []
        details = []
        for gen_text, gt, future in zip(gen_texts, ground_truths, futures):
            sql_reward, sql_no_len, sql_detail = future.result()
            pred_sql, _, _, fmt_ok = extract_sql_from_response(gen_text)
            rewards.append(sql_reward)
            details.append({
                "db_id": gt.get("db_id", ""),
                "gold_sql": gt.get("sql", ""),
                "pred_sql": pred_sql,
                "fmt_ok": fmt_ok,
                "sql_reward": sql_reward,
                "sql_no_len": sql_no_len,
                "ret_reward": 0.0,
                "gold_col_cnt": len(set(gt.get("gold_column_indices", []))),
                "coverage_rate": 0.0,
                "coverage_score": 0.0,
                "pred_col_cnt": 0,
                "noise_count": 0,
                "irrelevant_count": 0,
                "useful_noise_count": 0,
                "irrelevant_penalty": 0.0,
                "useful_noise_bonus": 0.0,
                "ret_case": "GENERATOR_ONLY",
                "ret_length_score": 0.0,
                **sql_detail,
            })
        return rewards, details

    def compute_rewards(
        self,
        ret_texts: List[str],
        gen_texts: List[str],
        ground_truths: List[dict],
        extra_infos: List[dict],
        group_ids: Optional[List[str]] = None,
        repeat_n: Optional[int] = None,
        verbose: bool = False,
        update_pool: bool = True,
        ret_n: Optional[int] = None,
        gen_resp_lens: Optional[List[int]] = None,
        ret_resp_lens: Optional[List[int]] = None,
        ret_response_length: Optional[int] = None,
    ) -> Tuple[List[float], List[float], List[dict]]:
        """
        V4: Group samples by question, compute empirical attribution per group.
        Assumes data is repeated with interleave=True: [q1,q1,q1,q1, q2,q2,q2,q2, ...]

        Args:
            update_pool: If True, update the historical column set pool (training).
                         If False, read-only access to pool (validation).
            ret_n: (Reserved, currently unused.) Number of independent retriever
                   samples per group.  Kept for caller compatibility.
        """
        n = len(ret_texts)
        if gen_resp_lens is not None and len(gen_resp_lens) != n:
            raise ValueError(f"gen_resp_lens length {len(gen_resp_lens)} does not match number of samples {n}")
        if ret_resp_lens is not None and len(ret_resp_lens) != n:
            raise ValueError(f"ret_resp_lens length {len(ret_resp_lens)} does not match number of samples {n}")
        if group_ids is not None:
            if len(group_ids) != n:
                raise ValueError(f"group_ids length {len(group_ids)} does not match number of samples {n}")
            grouped_indices = {}
            ordered_group_ids = []
            for idx, group_id in enumerate(group_ids):
                key = str(group_id)
                if key not in grouped_indices:
                    grouped_indices[key] = []
                    ordered_group_ids.append(key)
                grouped_indices[key].append(idx)
            group_index_lists = [grouped_indices[key] for key in ordered_group_ids]
        else:
            if repeat_n is None or repeat_n <= 0:
                raise ValueError(
                    "compute_rewards requires explicit grouping: pass group_ids, or pass a positive repeat_n. "
                    "Inferring groups from db_id can hide rollout/reward alignment bugs."
                )
            if n % repeat_n != 0:
                raise ValueError(f"Invalid repeat_n={repeat_n} for {n} samples")
            group_index_lists = [
                list(range(start_idx, start_idx + repeat_n))
                for start_idx in range(0, n, repeat_n)
            ]

        # Pass 1: Compute SQL rewards for all samples (parallel)
        if gen_resp_lens is None:
            gen_resp_lens = [None] * n
        if ret_resp_lens is None:
            ret_resp_lens = [None] * n
        futures = [
            self.executor.submit(compute_sql_reward, gen_text, gt, gen_resp_len)
            for gen_text, gt, gen_resp_len in zip(gen_texts, ground_truths, gen_resp_lens)
        ]

        gen_rewards = []
        all_exec_statuses = []
        sql_details_list = []

        for future in futures:
            sql_reward, sql_no_len, sql_detail = future.result()
            gen_rewards.append(sql_reward)
            all_exec_statuses.append(sql_detail["exec_status"])
            sql_details_list.append(sql_detail)

        # Pass 2: Process each question group
        ret_rewards = [0.0] * n
        details = [None] * n

        for sample_indices in group_index_lists:
            first_idx = sample_indices[0]
            first_gt = ground_truths[first_idx]
            first_db_id = first_gt.get("db_id", "")
            first_sql = first_gt.get("sql", "")
            for idx in sample_indices[1:]:
                cur_gt = ground_truths[idx]
                if cur_gt.get("db_id", "") != first_db_id or cur_gt.get("sql", "") != first_sql:
                    raise ValueError(
                        f"Inconsistent ground truth within reward group {sample_indices}: "
                        f"{first_db_id=} vs {cur_gt.get('db_id', '')}, SQL mismatch={cur_gt.get('sql', '') != first_sql}"
                    )

            # Get group data
            group_gen_texts = [gen_texts[idx] for idx in sample_indices]
            group_exec_statuses = [all_exec_statuses[idx] for idx in sample_indices]
            gt = ground_truths[first_idx]
            extra = extra_infos[first_idx]

            db_info = json.loads(extra.get("db_info", "{}")) if isinstance(extra.get("db_info"), str) else extra.get("db_info", {})
            gold_indices = set(gt.get("gold_column_indices", []))
            db_path = resolve_db_path(gt.get("db_path", ""), first_db_id)
            gold_sql = gt.get("sql", "")

            # Stable question key for pool lookup
            question_key = f"{first_db_id}||{gt.get('question', '')}"

            # Empirical attribution for this group (current rollout only)
            correct_sql_col_sets, harmful_cols, gen_only_correct_col_sets = compute_empirical_attribution(
                group_gen_texts, group_exec_statuses, db_path, db_info, gold_indices, gold_sql
            )

            # --- Pool integration ---
            # Step 1: Read existing pool entries with historical decay.
            pool_freq = {}
            if question_key in self.col_set_pool:
                for set_key, freq in self.col_set_pool[question_key].items():
                    pool_freq[set_key] = freq * self.pool_gamma

            # Step 2: Update pool with current rollout (training only)
            if update_pool:
                current_freq = {}
                for col_set in gen_only_correct_col_sets:
                    set_key = frozenset(col_set)
                    current_freq[set_key] = current_freq.get(set_key, 0) + 1

                updated_pool = dict(pool_freq)
                for set_key, freq in current_freq.items():
                    updated_pool[set_key] = updated_pool.get(set_key, 0) + freq
                self.col_set_pool[question_key] = updated_pool

            # Step 3: Merge pool col sets into correct_sql_col_sets
            pool_col_sets = [set(set_key) for set_key in pool_freq if pool_freq[set_key] > 0]
            existing_frozen = {frozenset(cs) for cs in correct_sql_col_sets}
            for pcs in pool_col_sets:
                if frozenset(pcs) not in existing_frozen:
                    correct_sql_col_sets.append(pcs)
                    existing_frozen.add(frozenset(pcs))

            # Step 4 (V6): Compute best_set, col_weights, useful_set from pool
            # Combine pool + current batch frequencies for best_set selection
            all_freq: Dict[frozenset, float] = dict(pool_freq)
            for col_set in gen_only_correct_col_sets:
                sk = frozenset(col_set)
                all_freq[sk] = all_freq.get(sk, 0) + 1

            # best_set = highest-frequency col set; tiebreak: fewest columns
            if all_freq:
                max_f = max(all_freq.values())
                candidates = [s for s, f in all_freq.items() if abs(f - max_f) < 1e-9]
                best_set = set(min(candidates, key=len))
            else:
                best_set = gold_indices.copy()

            # col_weights: per-column importance = appearances / total correct
            total_correct = sum(all_freq.values())
            col_weights: Dict[int, float] = {}
            if total_correct > 0:
                for col_set_key, freq in all_freq.items():
                    for c in col_set_key:
                        col_weights[c] = col_weights.get(c, 0.0) + freq
                for c in col_weights:
                    col_weights[c] /= total_correct

            # useful_set: all columns that appear in any correct SQL (w > 0)
            useful_set: Set[int] = set()
            for col_set in correct_sql_col_sets:
                useful_set.update(col_set)

            # Parse once and cache for reuse
            group_parsed_results = []
            for idx in sample_indices:
                ret_text = ret_texts[idx]
                column_list = parse_column_list(ret_text)
                if column_list:
                    pred_indices = select_columns_from_list(db_info, column_list, strict_match=True)
                else:
                    pred_indices = set()
                group_parsed_results.append((column_list, pred_indices))

            # Compute retriever rewards for each sample in group
            for local_idx, idx in enumerate(sample_indices):
                ret_text = ret_texts[idx]
                gen_text = gen_texts[idx]
                column_list, pred_indices = group_parsed_results[local_idx]

                if self.retriever_reward_mode == "pool_exact":
                    ret_reward, ret_detail = compute_pool_exact_retriever_reward(
                        ret_text=ret_text,
                        target_sets=self._target_sets_from_pool(pool_freq),
                        pred_indices=pred_indices,
                        exact_reward=self.pool_exact_reward,
                        ret_resp_len=ret_resp_lens[idx],
                        response_length=ret_response_length,
                    )
                else:
                    ret_reward, ret_detail = compute_retriever_reward(
                        ret_text=ret_text,
                        db_info=db_info,
                        gold_indices=gold_indices,
                        best_set=best_set,
                        col_weights=col_weights,
                        useful_set=useful_set,
                        pred_indices=pred_indices,
                        useful_noise_bar=self.useful_noise_bar,
                    )

                pred_sql, _, _, fmt_ok = extract_sql_from_response(gen_text)
                detail = {
                    "db_id": gt.get("db_id", ""),
                    "gold_sql": gold_sql,
                    "pred_sql": pred_sql,
                    "fmt_ok": fmt_ok,
                    "sql_reward": gen_rewards[idx],
                    "sql_no_len": gen_rewards[idx] - sql_details_list[idx].get("length_score", 0.0),
                    "ret_reward": ret_reward,
                    "gold_col_cnt": len(gold_indices),
                    **sql_details_list[idx],
                    **ret_detail,
                }

                ret_rewards[idx] = ret_reward
                details[idx] = detail

        return ret_rewards, gen_rewards, details
