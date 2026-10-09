"""ACE-SQL's ordinary shared-policy GRPO objective for slime v0.2.3.

The rollout is balanced globally, with N samples of each role per question.
Slime divides its sum of sample means by the combined global batch size. Thus
each sample needs a factor of 2 * role_weight to obtain
    w_ret * mean(L_ret) + w_gen * mean(L_gen).
The reducer weights the complete upstream objective, including reference KL.
There is no gradient projection and no extra optimizer step for either role.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import torch

ROLES = frozenset({"retriever", "generator"})


def validate_loss_configuration(args: Any) -> None:
    """Fail rather than silently change the paper's objective."""
    if getattr(args, "advantage_estimator", "grpo") != "grpo":
        raise ValueError("ACE-SQL requires advantage_estimator=grpo")
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError("ACE-SQL requires sample-mean loss; calculate_per_token_loss is unsupported")
    if getattr(args, "context_parallel_size", 1) != 1 or getattr(args, "allgather_cp", False):
        raise ValueError("ACE-SQL v0.2.3 adapter supports context_parallel_size=1 only")
    if getattr(args, "normalize_advantages", False):
        raise ValueError("ACE-SQL normalizes within question and role; disable normalize_advantages")
    if not getattr(args, "grpo_std_normalization", True):
        raise ValueError("ACE-SQL requires GRPO standard-deviation normalization")
    if float(getattr(args, "kl_coef", 0.0)) != 0.0:
        raise ValueError("ACE-SQL uses reference KL in the loss; reward kl_coef must be 0")
    if getattr(args, "ref_update_interval", None) is not None:
        raise ValueError("ACE-SQL requires a frozen reference policy; disable ref_update_interval")
    for name in ("use_tis", "get_mismatch_metrics", "use_opsm", "use_opd"):
        if getattr(args, name, False):
            raise ValueError(f"ACE-SQL ordinary GRPO does not support {name}")
    if getattr(args, "custom_pg_loss_reducer_function_path", None):
        raise ValueError("ACE-SQL supplies its full-objective reducer; a custom PG-only reducer is unsupported")


def _role_metadata(metadata: Any) -> tuple[str, float]:
    if not isinstance(metadata, Mapping):
        raise ValueError("ACE-SQL requires Sample.train_metadata with role and loss_weight")
    role = metadata.get("role")
    if role not in ROLES:
        raise ValueError(f"Unknown ACE-SQL role {role!r}; expected retriever or generator")
    try:
        weight = float(metadata["loss_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ACE-SQL metadata requires a numeric loss_weight") from exc
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"ACE-SQL loss_weight must be finite and in [0, 1], got {weight}")
    return role, weight


def weighted_sample_reducer(
    batch: Mapping[str, Any],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Preserve slime's response masks and denominators while weighting roles.

    A DP rank or microbatch may contain just one role after length balancing;
    balance is enforced on the complete rollout, never on local microbatches.
    """
    lengths = batch["response_lengths"]
    metadata = batch.get("metadata")
    if metadata is None or len(metadata) != len(lengths):
        raise ValueError(
            "ACE-SQL training metadata is missing or misaligned. Apply "
            "patches/slime-v0.2.3-ace-metadata.patch to the pinned slime checkout."
        )
    weights = [2.0 * _role_metadata(item)[1] for item in metadata]
    if any(int(length) != length or length < 0 for length in lengths):
        raise ValueError("Response lengths must be nonnegative integers")

    def reduce(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 1 or values.numel() != sum(lengths):
            raise ValueError("ACE-SQL expects concatenated response-token values with CP=1")
        factors = torch.repeat_interleave(
            values.new_tensor(weights),
            torch.tensor(lengths, dtype=torch.long, device=values.device),
        )
        return sum_of_sample_mean(values * factors)

    return reduce


def policy_loss(args: Any, batch: Mapping[str, Any], logits: torch.Tensor, sum_of_sample_mean: Callable):
    """Slime custom-loss entrypoint; reuse its clipped PPO and KL implementation."""
    validate_loss_configuration(args)
    from slime.backends.megatron_utils.loss import policy_loss_function

    return policy_loss_function(args, batch, logits, weighted_sample_reducer(batch, sum_of_sample_mean))


def _question_group(sample: Any) -> Any:
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return ("group_index", group_index)
    for metadata in (getattr(sample, "train_metadata", None), getattr(sample, "metadata", None)):
        if isinstance(metadata, Mapping):
            for key in ("row_key", "question_key"):
                if metadata.get(key) is not None:
                    return (key, metadata[key])
    raise ValueError("ACE-SQL reward normalization requires group_index or a metadata row_key/question_key")


def reward_postprocess(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Return raw rewards and role-isolated GRPO advantages in original order.

    Uses float32, unbiased sample std, and std + 1e-6 as in the released normalization convention.
    This hook runs before data-parallel partitioning. It replaces slime's
    positional reshape, which would otherwise normalize both roles together.
    """
    validate_loss_configuration(args)
    group_size = int(getattr(args, "ace_group_size", 8))
    if group_size < 2:
        raise ValueError("ACE-SQL requires ace_group_size >= 2 for unbiased GRPO normalization")
    if not samples:
        raise ValueError("ACE-SQL cannot normalize an empty rollout")
    grouped: dict[Any, list[int]] = defaultdict(list)
    role_weights: dict[str, float] = {}
    raw: list[float] = []
    questions = set()
    for index, sample in enumerate(samples):
        role, weight = _role_metadata(getattr(sample, "train_metadata", None))
        if role in role_weights and not math.isclose(weight, role_weights[role], abs_tol=1e-9):
            raise ValueError("All samples of a role must use the same scheduled weight in one rollout")
        role_weights[role] = weight
        question = _question_group(sample)
        questions.add(question)
        grouped[(question, role)].append(index)
        reward = float(sample.get_reward_value(args))
        if not math.isfinite(reward):
            raise ValueError("ACE-SQL rewards must be finite")
        raw.append(reward)
    for question in questions:
        for role in ROLES:
            size = len(grouped[(question, role)])
            if size != group_size:
                raise ValueError(f"Question {question!r}, role {role}: expected {group_size} samples, got {size}")
    if not math.isclose(sum(role_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("ACE-SQL scheduled retriever/generator weights must sum to 1")
    advantages = [0.0] * len(samples)
    for indices in grouped.values():
        rewards = torch.tensor([raw[index] for index in indices], dtype=torch.float32)
        normalized = (rewards - rewards.mean()) / (rewards.std(unbiased=True) + 1e-6)
        for index, value in zip(indices, normalized.tolist(), strict=True):
            advantages[index] = value
    return raw, advantages
