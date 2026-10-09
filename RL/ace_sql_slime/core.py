"""Framework-independent ACE-SQL rollout/reward semantics from paper_v3.

The published parsing, prompt template, SQL grading, and historical-pool reward
implementation are reused directly.  This module imports neither verl nor vLLM.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from threading import RLock

from rewards.joint_reward import JointRewardManager
from rewards.retriever_utils import build_generator_prompt, parse_retriever_output
from utils.path_utils import resolve_db_path

from .data import as_object, atomic_write_text, question_key, validate_metadata


GROUP_SIZE = 8
METHOD_VERSION = "arr-20261009-union-updated-pool"


def role_weights(step: int, total_steps: int, ramp_ratio: float = 0.25) -> tuple[float, float]:
    """Complementary paper weights; slime's first rollout/update is step zero."""
    if type(step) is not int or step < 0 or type(total_steps) is not int or total_steps <= 0:
        raise ValueError("step must be a nonnegative integer and total_steps a positive integer")
    if not math.isfinite(ramp_ratio) or not 0 <= ramp_ratio <= 1:
        raise ValueError("ramp_ratio must be finite and in [0, 1]")
    lam = 1.0 if ramp_ratio == 0 else min(1.0, step / (total_steps * ramp_ratio))
    return 1.0 - lam / 2.0, lam / 2.0


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


class RewardEngine:
    """Serialize visits to the historical pool and score aligned role groups.

    Training decays the historical pool and adds current matches before reading
    the retriever target. Validation reads the saved pool without modifying it.
    Failed scoring rolls back the complete pool state.
    """

    def __init__(self, initial_pool_path: str | Path | None = None, max_workers: int = 8,
                 pool_gamma: float = 0.5, group_size: int = GROUP_SIZE,
                 target_mode: str = "online", length_penalty: bool = True):
        if group_size <= 0 or max_workers <= 0:
            raise ValueError("group_size and max_workers must be positive")
        if not math.isfinite(pool_gamma) or not 0 <= pool_gamma <= 1:
            raise ValueError("pool_gamma must be finite and in [0, 1]")
        if target_mode not in {"online", "frozen", "gold", "dense"}:
            raise ValueError("target_mode must be online, frozen, gold, or dense")
        self.group_size = group_size
        self.target_mode = target_mode
        self.length_penalty = bool(length_penalty)
        self.manager = JointRewardManager(
            max_workers=max_workers, pool_gamma=pool_gamma,
            initial_pool_path=str(initial_pool_path) if initial_pool_path else None,
            retriever_reward_mode={"gold": "gold_exact", "dense": "pool_dense"}.get(target_mode, "pool_exact"),
            pool_exact_reward=1.0, length_penalty_enabled=self.length_penalty,
        )
        self._lock = RLock()

    def compute_group(self, ret_texts: list[str], gen_texts: list[str], metadata: dict, *,
                      ret_resp_lens: list[int] | None = None,
                      gen_resp_lens: list[int] | None = None,
                      update_pool: bool = True,
                      ret_response_length: int = 2048) -> tuple[list[float], list[float], list[dict]]:
        n = 1 if not update_pool and len(ret_texts) == len(gen_texts) == 1 else self.group_size
        if len(ret_texts) != n or len(gen_texts) != n:
            raise ValueError(f"both roles must have exactly {n} samples per question")
        gt, extra = validate_metadata(metadata)
        key = question_key(gt)
        database = Path(resolve_db_path(gt["db_path"], gt["db_id"])).resolve()
        if not database.is_file():
            raise FileNotFoundError(f"SQLite database missing for {key}: {database}")
        gt = {**gt, "db_path": str(database)}
        for name, lengths in (("ret_resp_lens", ret_resp_lens), ("gen_resp_lens", gen_resp_lens)):
            if lengths is not None and (len(lengths) != n or any(type(x) is not int or x < 0 for x in lengths)):
                raise ValueError(f"{name} must contain {n} nonnegative integer token lengths")
        with self._lock:
            pool = self.manager.col_set_pool
            if key not in pool:
                raise ValueError(f"no historical pool target for {key}")
            before = deepcopy(pool[key])
            try:
                return self.manager.compute_rewards(
                    ret_texts=ret_texts, gen_texts=gen_texts,
                    ground_truths=[gt] * n, extra_infos=[extra] * n,
                    group_ids=[key] * n, repeat_n=n, ret_n=n,
                    update_pool=update_pool and self.target_mode not in {"frozen", "gold"}, ret_resp_lens=ret_resp_lens,
                    gen_resp_lens=gen_resp_lens, ret_response_length=ret_response_length,
                )
            except BaseException:
                self.manager.col_set_pool[key] = before
                raise

    def state_dict(self) -> dict:
        with self._lock:
            pool = {key: {",".join(map(str, sorted(columns))): frequency
                          for columns, frequency in entries.items()}
                    for key, entries in self.manager.col_set_pool.items()}
            return {"version": 2, "method_version": METHOD_VERSION, "group_size": self.group_size,
                    "target_mode": self.target_mode, "length_penalty": self.length_penalty,
                    "pool_gamma": self.manager.pool_gamma, "pool": pool}

    def load_state_dict(self, state: dict) -> None:
        if (state.get("version") != 2 or state.get("method_version") != METHOD_VERSION
                or state.get("group_size") != self.group_size
                or state.get("target_mode") != self.target_mode
                or state.get("length_penalty") != self.length_penalty):
            raise ValueError("incompatible reward checkpoint version or group size")
        if state.get("pool_gamma") != self.manager.pool_gamma:
            raise ValueError("reward checkpoint pool decay differs from current run")
        pool = {}
        for key, entries in as_object(state.get("pool"), "pool").items():
            pool[key] = {}
            for columns, frequency in as_object(entries, "pool entry").items():
                frequency = float(frequency)
                if not math.isfinite(frequency) or frequency < 0:
                    raise ValueError("pool frequency must be finite and nonnegative")
                pool[key][frozenset(int(col) for col in columns.split(",") if col)] = frequency
        with self._lock:
            self.manager.col_set_pool = pool

    def save_pool(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.state_dict()["pool"], indent=2, allow_nan=False) + "\n")

    def close(self) -> None:
        self.manager.executor.shutdown(wait=True)
