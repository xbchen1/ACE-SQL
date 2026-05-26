"""
VERL-compatible reward wrapper for SQL reward computation.
Registered via custom_reward_function in yaml config.

VERL's NaiveRewardManager calls this with:
    data_source: str  (from non_tensor_batch["data_source"])
    solution_str: str (decoded model response)
    ground_truth: dict (from non_tensor_batch["reward_model"]["ground_truth"])
    extra_info: dict   (from non_tensor_batch["extra_info"])
"""
import os
import sys

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rewards.joint_reward import compute_sql_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """VERL-compatible interface for SQL reward."""
    if "synsql" in data_source:
        score, _, _ = compute_sql_reward(solution_str, ground_truth)
        return float(score)
    raise NotImplementedError(f"Reward not implemented for {data_source=}")
