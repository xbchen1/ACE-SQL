"""
Two-Pass GRPO Ray Trainer for joint Retriever + Generator training.

This trainer subclasses VERL 0.5.5 RayPPOTrainer and overrides fit() to run:
1. Retriever rollout
2. Build generator prompts from retriever output
3. Generator rollout
4. Joint reward computation
5. Joint actor update with alpha/beta loss scaling
"""

import gc
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import open_dict
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rewards.joint_reward import JointRewardManager, aggregate_reward_details
from rewards.retriever_utils import build_generator_prompt, parse_retriever_output

# Fixed validation uses the same rollout and reward path as training.
from trainer.validation_correct import validate_training_rollout


class TwoPassGRPOTrainer(RayPPOTrainer):
    """Two-pass GRPO trainer with joint retriever/generator optimization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SERL-like weighting schedule (kept configurable)
        serl_cfg = self.config.get("serl", {})
        self.constant_loss_scale = bool(serl_cfg.get("constant_loss_scale", False))
        self.loss_weight_schedule = str(serl_cfg.get("loss_weight_schedule", "legacy")).strip().lower()
        valid_loss_weight_schedules = {"legacy", "linear_joint_ramp"}
        if self.loss_weight_schedule not in valid_loss_weight_schedules:
            raise ValueError(
                f"Invalid serl.loss_weight_schedule={self.loss_weight_schedule!r}; "
                f"expected one of {sorted(valid_loss_weight_schedules)}"
            )
        if self.constant_loss_scale and self.loss_weight_schedule != "legacy":
            raise ValueError(
                "serl.constant_loss_scale=True conflicts with "
                f"serl.loss_weight_schedule={self.loss_weight_schedule!r}. "
                "Disable constant_loss_scale for scheduled joint loss weights."
            )
        self.loss_weight_ramp_ratio = float(serl_cfg.get("loss_weight_ramp_ratio", 0.25))
        self.loss_weight_ramp_ratio = min(max(self.loss_weight_ramp_ratio, 1e-6), 1.0)
        self.retriever_loss_weight_start = float(serl_cfg.get("retriever_loss_weight_start", 1.0))
        self.retriever_loss_weight_end = float(serl_cfg.get("retriever_loss_weight_end", 1.0))
        self.generator_loss_weight_start = float(serl_cfg.get("generator_loss_weight_start", 0.0))
        self.generator_loss_weight_end = float(serl_cfg.get("generator_loss_weight_end", 1.0))
        if self.loss_weight_schedule == "linear_joint_ramp":
            print(
                "[SERL] Linear joint loss-weight ramp: "
                f"first {self.loss_weight_ramp_ratio * 100:.0f}% steps, "
                f"retriever {self.retriever_loss_weight_start}->{self.retriever_loss_weight_end}, "
                f"generator {self.generator_loss_weight_start}->{self.generator_loss_weight_end}; "
                "then keep final weights"
            )

        # Initialize reward manager with optional pre-built pool
        initial_pool_path = serl_cfg.get("initial_pool_path", None)
        reward_workers = int(self.config.get("reward", {}).get("num_workers", 8))
        if reward_workers <= 0:
            raise ValueError(f"reward.num_workers must be positive, got {reward_workers}")
        self.reward_manager = JointRewardManager(
            max_workers=reward_workers,
            initial_pool_path=initial_pool_path,
            retriever_reward_mode=serl_cfg.get("retriever_reward_mode", "pool_exact"),
            pool_exact_reward=float(serl_cfg.get("pool_exact_reward", 1.0)),
            pool_gamma=float(serl_cfg.get("pool_gamma", 0.5)),
        )
        print(f"[TwoPass] Reward SQL workers: {reward_workers}")
        self.alpha_init = serl_cfg.get("alpha_init", 0.05)
        self.alpha_max = serl_cfg.get("alpha_max", 0.5)
        self.beta_min = serl_cfg.get("beta_min", 0.5)
        # Fraction of total training progress used to ramp alpha to alpha_max.
        # After this point, alpha stays at alpha_max and beta stays at max(1-alpha, beta_min).
        self.alpha_ramp_ratio = float(serl_cfg.get("alpha_ramp_ratio", 1.0))
        self.alpha_ramp_ratio = min(max(self.alpha_ramp_ratio, 1e-6), 1.0)

        # Three-phase schedule params (all must be non-null to activate)
        _as = serl_cfg.get("alpha_start", None)
        _am = serl_cfg.get("alpha_mid", None)
        _ae = serl_cfg.get("alpha_end", None)
        _p1 = serl_cfg.get("phase1_end", None)
        _p2 = serl_cfg.get("phase2_end", None)
        self.use_three_phase = all(v is not None for v in [_as, _am, _ae, _p1, _p2])

        if self.loss_weight_schedule == "legacy":
            if self.constant_loss_scale:
                print("[SERL] Constant loss scaling: retriever=1.0, generator=1.0")

            if self.use_three_phase:
                self.alpha_start = float(_as)
                self.alpha_mid = float(_am)
                self.alpha_end = float(_ae)
                self.phase1_end = float(_p1)
                self.phase2_end = float(_p2)
                # Validate: 0 < phase1_end < phase2_end < 1
                assert 0 < self.phase1_end < self.phase2_end < 1.0, \
                    f"Invalid phase boundaries: phase1_end={self.phase1_end}, phase2_end={self.phase2_end}, need 0 < p1 < p2 < 1"
                print(f"[SERL] Three-phase schedule: "
                      f"alpha {self.alpha_start}→{self.alpha_mid}→{self.alpha_end}, "
                      f"phases [0,{self.phase1_end}), [{self.phase1_end},{self.phase2_end}), [{self.phase2_end},1.0]")
            else:
                self.alpha_start = self.alpha_mid = self.alpha_end = None
                self.phase1_end = self.phase2_end = None
                print(f"[SERL] Legacy two-phase schedule: "
                      f"alpha {self.alpha_init}→{self.alpha_max} over {self.alpha_ramp_ratio*100:.0f}% steps, "
                      f"beta_min={self.beta_min}")
        else:
            self.alpha_start = self.alpha_mid = self.alpha_end = None
            self.phase1_end = self.phase2_end = None
            self.use_three_phase = False

        # Generator prompt length control
        self.gen_max_prompt_length = serl_cfg.get("gen_max_prompt_length", 4096)
        rollout_prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        rollout_response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        rollout_max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", None)
        rollout_max_model_len = (
            int(rollout_max_model_len)
            if rollout_max_model_len
            else rollout_prompt_length + rollout_response_length
        )
        if int(self.gen_max_prompt_length) > rollout_prompt_length:
            raise ValueError(
                f"serl.gen_max_prompt_length={self.gen_max_prompt_length} must be <= "
                f"actor_rollout_ref.rollout.prompt_length={rollout_prompt_length}"
            )
        min_rollout_model_len = rollout_prompt_length + rollout_response_length
        if rollout_max_model_len < min_rollout_model_len:
            raise ValueError(
                f"actor_rollout_ref.rollout.max_model_len={rollout_max_model_len} must be >= "
                f"prompt_length({rollout_prompt_length}) + response_length({rollout_response_length}) = "
                f"{min_rollout_model_len}; otherwise vLLM silently caps rollout responses."
            )
        self.rollout_response_length = rollout_response_length
        self.retriever_response_length = self._resolve_role_response_length(
            serl_cfg, "retriever_response_length", rollout_response_length
        )
        self.generator_response_length = self._resolve_role_response_length(
            serl_cfg, "generator_response_length", rollout_response_length
        )
        print(
            "[TwoPass] Response lengths: "
            f"global={self.rollout_response_length}, "
            f"retriever={self.retriever_response_length}, "
            f"generator={self.generator_response_length}"
        )
        generator_prompt_mode_raw = str(serl_cfg.get("generator_prompt_mode", "merged_union"))
        generator_prompt_mode = generator_prompt_mode_raw.strip().lower().replace("-", "_")
        generator_prompt_mode_aliases = {
            "maj_vote": "majority_vote",
            "maj_voting": "majority_vote",
            "major_vote": "majority_vote",
            "major_voting": "majority_vote",
            "majority_voting": "majority_vote",
        }
        self.generator_prompt_mode = generator_prompt_mode_aliases.get(
            generator_prompt_mode, generator_prompt_mode
        )
        valid_prompt_modes = {"per_retriever", "merged_union", "majority_vote"}
        if self.generator_prompt_mode not in valid_prompt_modes:
            raise ValueError(
                f"Invalid serl.generator_prompt_mode={generator_prompt_mode_raw!r}; "
                f"expected one of {sorted(valid_prompt_modes)}"
            )
        print(f"[TwoPass] Generator prompt mode: {self.generator_prompt_mode}")
        self.generator_prompt_vote_threshold = float(
            serl_cfg.get("generator_prompt_vote_threshold", 0.5)
        )
        if not 0.0 < self.generator_prompt_vote_threshold <= 1.0:
            raise ValueError(
                "serl.generator_prompt_vote_threshold must be in (0, 1], "
                f"got {self.generator_prompt_vote_threshold}"
            )
        if self.generator_prompt_mode == "majority_vote":
            print(
                "[TwoPass] Generator prompt majority-vote threshold: "
                f"{self.generator_prompt_vote_threshold}"
            )

        phase_schedule_raw = str(serl_cfg.get("training_phase_schedule", "joint"))
        self.training_phase_schedule = [
            phase.strip().lower()
            for phase in phase_schedule_raw.split(",")
            if phase.strip()
        ] or ["joint"]
        valid_phases = {"joint", "retriever", "generator"}
        bad_phases = [phase for phase in self.training_phase_schedule if phase not in valid_phases]
        if bad_phases:
            raise ValueError(
                f"Invalid serl.training_phase_schedule entries={bad_phases}; "
                f"expected entries from {sorted(valid_phases)}"
            )
        self.retriever_only_warmup_epochs = int(serl_cfg.get("retriever_only_warmup_epochs", 0) or 0)
        if self.retriever_only_warmup_epochs < 0:
            raise ValueError(
                "serl.retriever_only_warmup_epochs must be non-negative, "
                f"got {self.retriever_only_warmup_epochs}"
            )
        self.alternating_training = (
            any(phase != "joint" for phase in self.training_phase_schedule)
            or self.retriever_only_warmup_epochs > 0
        )
        print(f"[TwoPass] Training phase schedule: {self.training_phase_schedule}")
        if self.retriever_only_warmup_epochs > 0:
            print(f"[TwoPass] Retriever-only warmup epochs: {self.retriever_only_warmup_epochs}")

        # Separate group sizes for retriever and generator rollouts
        self.gen_n = self.config.actor_rollout_ref.rollout.n
        ret_n_cfg = serl_cfg.get("ret_n", None)
        self.ret_n = int(ret_n_cfg) if ret_n_cfg is not None else self.gen_n
        retriever_only_ret_n_cfg = serl_cfg.get("retriever_only_ret_n", None)
        self.retriever_only_ret_n = (
            int(retriever_only_ret_n_cfg)
            if retriever_only_ret_n_cfg is not None
            else self.ret_n
        )
        if self.ret_n <= 0:
            raise ValueError(f"serl.ret_n must be positive, got {self.ret_n}")
        if self.retriever_only_ret_n <= 0:
            raise ValueError(
                "serl.retriever_only_ret_n must be positive when set, "
                f"got {self.retriever_only_ret_n}"
            )
        uses_generator_phase = any(
            phase in {"joint", "generator"} for phase in self.training_phase_schedule
        )
        self._configure_dual_lora_lr_decay_horizons(uses_generator_phase=uses_generator_phase)
        group_level_prompt_modes = {"merged_union", "majority_vote"}
        if self.ret_n > self.gen_n and uses_generator_phase:
            uses_joint_phase = any(phase == "joint" for phase in self.training_phase_schedule)
            if uses_joint_phase:
                raise NotImplementedError(
                    "Joint two-pass reward mapping requires serl.ret_n <= rollout.n. "
                    f"Got ret_n={self.ret_n}, rollout.n={self.gen_n}, "
                    f"training_phase_schedule={self.training_phase_schedule}. "
                    "Use alternating generator-only phases, increase rollout.n, or reduce serl.ret_n."
                )
            if self.generator_prompt_mode not in group_level_prompt_modes:
                raise NotImplementedError(
                    "Generator-only phases with serl.ret_n > rollout.n require a group-level prompt mode. "
                    f"Got ret_n={self.ret_n}, rollout.n={self.gen_n}, "
                    f"generator_prompt_mode={self.generator_prompt_mode!r}. "
                    "Use merged_union/majority_vote, increase rollout.n, or reduce serl.ret_n."
                )
        print(
            "[TwoPass] Group sizes: "
            f"joint ret_n={self.ret_n}, gen_n={self.gen_n}, "
            f"retriever_only_ret_n={self.retriever_only_ret_n}"
        )
        if self.config.actor_rollout_ref.rollout.mode == "async":
            if not bool(self.config.data.get("return_raw_chat", False)):
                raise ValueError(
                    "Async agent-loop rollout requires data.return_raw_chat=True so raw_prompt is available."
                )
            agent_workers = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
            if agent_workers <= 0:
                raise ValueError(f"Async rollout agent.num_workers must be positive, got {agent_workers}")
            train_batch_size = int(self.config.data.train_batch_size)
            repeat_checks = []
            if uses_generator_phase:
                repeat_checks.extend([
                    ("joint_retriever", self.ret_n),
                    ("generator", self.gen_n),
                ])
            if "retriever" in self.training_phase_schedule or self.retriever_only_warmup_epochs > 0:
                repeat_checks.append(("retriever_only", self.retriever_only_ret_n))
            for phase_name, repeat_n in repeat_checks:
                rollout_batch_size = train_batch_size * int(repeat_n)
                if rollout_batch_size % agent_workers != 0:
                    raise ValueError(
                        f"Async {phase_name} rollout batch size must be divisible by agent workers: "
                        f"data.train_batch_size({train_batch_size}) * n({repeat_n}) = "
                        f"{rollout_batch_size}, agent.num_workers={agent_workers}"
                    )

        base_rollout_temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        self.retriever_epoch_retriever_temperature = float(
            serl_cfg.get("retriever_epoch_retriever_temperature", base_rollout_temperature)
        )
        self.generator_epoch_retriever_temperature = float(
            serl_cfg.get("generator_epoch_retriever_temperature", base_rollout_temperature)
        )
        self.generator_epoch_generator_temperature = float(
            serl_cfg.get("generator_epoch_generator_temperature", base_rollout_temperature)
        )
        self.joint_retriever_temperature = float(
            serl_cfg.get("joint_retriever_temperature", self.retriever_epoch_retriever_temperature)
        )
        self.joint_generator_temperature = float(
            serl_cfg.get("joint_generator_temperature", self.generator_epoch_generator_temperature)
        )
        self.validation_retriever_temperature = float(
            serl_cfg.get("validation_retriever_temperature", base_rollout_temperature)
        )
        self.validation_generator_temperature = float(
            serl_cfg.get("validation_generator_temperature", self.validation_retriever_temperature)
        )
        print(
            "[TwoPass] Rollout temperatures: "
            f"retriever_epoch/retriever={self.retriever_epoch_retriever_temperature}, "
            f"generator_epoch/retriever={self.generator_epoch_retriever_temperature}, "
            f"generator_epoch/generator={self.generator_epoch_generator_temperature}, "
            f"validation/retriever={self.validation_retriever_temperature}, "
            f"validation/generator={self.validation_generator_temperature}"
        )

        # Debug printing can be noisy on large batches
        self.debug_print = bool(serl_cfg.get("debug_print", False))
        self.debug_print_max_samples = int(serl_cfg.get("debug_print_max_samples", 2))
        self.debug_dump_artifacts = bool(serl_cfg.get("debug_dump_artifacts", False))

        # Dual LoRA mode: independent adapters, no gradient projection
        self.dual_lora = self.config.actor_rollout_ref.model.get("dual_lora", {}).get("enabled", False)
        if self.dual_lora:
            print("[TwoPass] Dual LoRA mode ENABLED: retriever and generator use separate adapters")

    def _loss_scales_for_progress(self, progress: float) -> tuple[float, float, float]:
        """Return retriever/generator loss scales and ramp progress."""
        progress = min(max(float(progress), 0.0), 1.0)
        if self.loss_weight_schedule == "linear_joint_ramp":
            ramp_progress = min(progress / self.loss_weight_ramp_ratio, 1.0)
            retriever_weight = self.retriever_loss_weight_start + ramp_progress * (
                self.retriever_loss_weight_end - self.retriever_loss_weight_start
            )
            generator_weight = self.generator_loss_weight_start + ramp_progress * (
                self.generator_loss_weight_end - self.generator_loss_weight_start
            )
            return retriever_weight, generator_weight, ramp_progress

        if self.use_three_phase:
            if progress < self.phase1_end:
                t = progress / self.phase1_end
                alpha = self.alpha_start + t * (self.alpha_mid - self.alpha_start)
            elif progress < self.phase2_end:
                alpha = self.alpha_mid
            else:
                denom = 1.0 - self.phase2_end
                t = min((progress - self.phase2_end) / denom, 1.0)
                alpha = self.alpha_mid + t * (self.alpha_end - self.alpha_mid)
            alpha = min(max(alpha, 0.0), 1.0)
            beta = 1.0 - alpha
        else:
            ramp_progress = min(progress / self.alpha_ramp_ratio, 1.0)
            alpha = self.alpha_init + ramp_progress * (self.alpha_max - self.alpha_init)
            beta = max(1.0 - alpha, self.beta_min)

        if self.constant_loss_scale:
            alpha = 1.0
            beta = 1.0
        return alpha, beta, 1.0

    @staticmethod
    def _resolve_role_response_length(serl_cfg, key: str, global_response_length: int) -> int:
        value = serl_cfg.get(key, None)
        if value is None:
            return int(global_response_length)
        value = int(value)
        if value <= 0:
            raise ValueError(f"serl.{key} must be positive when set, got {value}")
        if value > int(global_response_length):
            raise ValueError(
                f"serl.{key}={value} exceeds actor_rollout_ref.rollout.response_length="
                f"{global_response_length}. Increase data.max_response_length/rollout.response_length first."
            )
        return value

    def _response_length_for_role(self, role: str) -> int:
        role = role.lower()
        if role == "retriever":
            return int(self.retriever_response_length)
        if role == "generator":
            return int(self.generator_response_length)
        raise ValueError(f"Unknown rollout role for response length: {role}")

    def _set_rollout_response_length(
        self,
        batch: DataProto,
        role: str,
        metrics: dict | None = None,
        metric_name: str | None = None,
    ) -> None:
        response_length = self._response_length_for_role(role)
        batch.meta_info["response_length"] = response_length
        if metrics is not None:
            key = metric_name or f"rollout/{role}_response_length"
            metrics[key] = float(response_length)

    def _configure_dual_lora_lr_decay_horizons(self, uses_generator_phase: bool):
        """Set adapter-specific LR decay horizons before actor workers build schedulers."""
        dual_lora_enabled = self.config.actor_rollout_ref.model.get("dual_lora", {}).get("enabled", False)
        if not dual_lora_enabled:
            return

        if self.config.trainer.get("total_training_steps", None) is not None:
            raise ValueError(
                "Dual-LoRA epoch-aware LR decay requires trainer.total_training_steps=null. "
                "Use trainer.total_epochs so retriever/generator scheduler horizons stay aligned "
                "with retriever-only warmup epochs."
            )

        steps_per_epoch = len(self.train_dataloader)
        if steps_per_epoch <= 0:
            raise ValueError(f"Cannot configure Dual-LoRA LR decay: train dataloader is empty ({steps_per_epoch}).")
        total_epochs = int(self.config.trainer.total_epochs)
        if total_epochs <= 0:
            raise ValueError(f"trainer.total_epochs must be positive, got {total_epochs}.")
        if self.retriever_only_warmup_epochs > total_epochs:
            raise ValueError(
                "serl.retriever_only_warmup_epochs cannot exceed trainer.total_epochs: "
                f"{self.retriever_only_warmup_epochs} > {total_epochs}."
            )

        generator_active_epochs = total_epochs - self.retriever_only_warmup_epochs
        if uses_generator_phase and generator_active_epochs <= 0:
            raise ValueError(
                "Generator phase has no active epochs after retriever-only warmup: "
                f"total_epochs={total_epochs}, "
                f"retriever_only_warmup_epochs={self.retriever_only_warmup_epochs}."
            )

        retriever_decay_steps = steps_per_epoch * total_epochs
        generator_decay_steps = steps_per_epoch * max(generator_active_epochs, 1)
        with open_dict(self.config.actor_rollout_ref.actor.optim):
            self.config.actor_rollout_ref.actor.optim.dual_lora_retriever_total_training_steps = (
                retriever_decay_steps
            )
            self.config.actor_rollout_ref.actor.optim.dual_lora_generator_total_training_steps = (
                generator_decay_steps
            )
        print(
            "[TwoPass] Dual-LoRA LR decay horizons: "
            f"steps_per_epoch={steps_per_epoch}, "
            f"retriever={retriever_decay_steps} steps/{total_epochs} epochs, "
            f"generator={generator_decay_steps} steps/{max(generator_active_epochs, 1)} active epochs"
        )

    def _rollout_temperature(self, phase: str, role: str, validation: bool = False) -> float:
        """Select rollout temperature for the current phase and model role."""
        role = role.lower()
        if validation:
            if role == "retriever":
                return self.validation_retriever_temperature
            if role == "generator":
                return self.validation_generator_temperature
            raise ValueError(f"Unknown validation rollout role: {role}")

        phase = phase.lower()
        if phase == "retriever" and role == "retriever":
            return self.retriever_epoch_retriever_temperature
        if phase == "generator" and role == "retriever":
            return self.generator_epoch_retriever_temperature
        if phase == "generator" and role == "generator":
            return self.generator_epoch_generator_temperature
        if phase == "joint" and role == "retriever":
            return self.joint_retriever_temperature
        if phase == "joint" and role == "generator":
            return self.joint_generator_temperature
        raise ValueError(f"Unsupported rollout temperature phase/role: {phase}/{role}")

    def _validate(self, merged=False):
        """Validate on the configured fixed parquet using the training rollout/reward path."""
        return validate_training_rollout(self)

    def _prepare_rollout_batch(self, batch: DataProto) -> DataProto:
        """Prepare generation input in the same style as VERL 0.5.5 fit()."""
        batch_keys_to_pop = [
            k for k in ["input_ids", "attention_mask", "position_ids"] if k in batch.batch.keys()
        ]

        non_tensor_batch_keys_to_pop = []
        for key in [
            "raw_prompt_ids",
            "multi_modal_data",
            "raw_prompt",
            "tools_kwargs",
            "interaction_kwargs",
            "index",
            "agent_name",
        ]:
            if key in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(key)

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        return gen_batch

    def _generate_sequences(self, gen_batch: DataProto) -> DataProto:
        """Run rollout generation under async/sync mode (standalone, with wake/sleep)."""
        if self.async_rollout_mode:
            output = self.async_rollout_manager.generate_sequences(gen_batch)
            expected_prompt_len = int(self.config.actor_rollout_ref.rollout.prompt_length)
        else:
            output = self.actor_rollout_wg.generate_sequences(gen_batch)
            expected_prompt_len = int(gen_batch.batch["input_ids"].shape[1])
        self._assert_rollout_output(
            output,
            expected_size=len(gen_batch),
            expected_prompt_len=expected_prompt_len,
            context="generate_sequences",
        )
        return output

    def _assert_rollout_output(
        self,
        output: DataProto,
        expected_size: int,
        expected_prompt_len: int,
        context: str,
    ) -> None:
        """Fail fast if rollout output no longer matches VERL/vLLM contracts."""
        actual_size = len(output)
        if actual_size != expected_size:
            raise RuntimeError(
                f"{context}: rollout output batch size mismatch: expected {expected_size}, got {actual_size}"
            )

        required = ["prompts", "responses", "input_ids", "attention_mask", "position_ids"]
        missing = [key for key in required if key not in output.batch.keys()]
        if missing:
            raise RuntimeError(f"{context}: rollout output missing tensor keys: {missing}")

        prompt_len = int(expected_prompt_len)
        response_len = int(self.config.actor_rollout_ref.rollout.response_length)
        prompts = output.batch["prompts"]
        responses = output.batch["responses"]
        input_ids = output.batch["input_ids"]
        attention_mask = output.batch["attention_mask"]
        position_ids = output.batch["position_ids"]
        if prompts.shape != (expected_size, prompt_len):
            raise RuntimeError(
                f"{context}: prompts shape mismatch: expected {(expected_size, prompt_len)}, "
                f"got {tuple(prompts.shape)}"
            )
        if responses.shape != (expected_size, response_len):
            raise RuntimeError(
                f"{context}: responses shape mismatch: expected {(expected_size, response_len)}, "
                f"got {tuple(responses.shape)}"
            )
        expected_seq_shape = (expected_size, prompt_len + response_len)
        for key, tensor in [
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
        ]:
            if tensor.shape != expected_seq_shape:
                raise RuntimeError(
                    f"{context}: {key} shape mismatch: expected {expected_seq_shape}, got {tuple(tensor.shape)}"
                )

    # ---- Optimization 1: Single weight-sync for two-pass rollout ----

    def _enter_rollout_context(self):
        """Enter rollout context: sync weights and prepare inference engine once.

        In async mode this performs a single wake_up (FSDP→vLLM weight sync +
        KV-cache allocation) that stays active across multiple generation calls,
        avoiding the redundant second weight sync between retriever and generator.
        """
        if self.async_rollout_mode:
            free_cache = self.config.actor_rollout_ref.rollout.get("free_cache_engine", True)
            if free_cache:
                self.async_rollout_manager.wake_up()

    def _exit_rollout_context(self):
        """Exit rollout context: free inference engine resources."""
        if self.async_rollout_mode:
            free_cache = self.config.actor_rollout_ref.rollout.get("free_cache_engine", True)
            if free_cache:
                self.async_rollout_manager.sleep()

    def _generate_in_context(self, gen_batch: DataProto) -> DataProto:
        """Generate sequences assuming rollout context is already entered.

        In async mode, dispatches directly to agent-loop workers without the
        per-call wake_up/sleep cycle (managed externally by _enter/_exit).
        Falls back to the standard generate path for sync mode.
        """
        if self.async_rollout_mode:
            mgr = self.async_rollout_manager
            chunks = gen_batch.chunk(len(mgr.agent_loop_workers))
            outputs = ray.get([
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(mgr.agent_loop_workers, chunks, strict=True)
            ])
            output = DataProto.concat(outputs)
            metrics_list = [o.meta_info["metrics"] for o in outputs]
            timing = mgr._performance_metrics(metrics_list, output)
            output.meta_info = {"timing": timing}
            self._assert_rollout_output(
                output,
                expected_size=len(gen_batch),
                expected_prompt_len=int(self.config.actor_rollout_ref.rollout.prompt_length),
                context="async_generate_in_context",
            )
            return output
        else:
            # Sync mode: each call enters the sharding-manager on its own
            output = self.actor_rollout_wg.generate_sequences(gen_batch)
            self._assert_rollout_output(
                output,
                expected_size=len(gen_batch),
                expected_prompt_len=int(gen_batch.batch["input_ids"].shape[1]),
                context="sync_generate_in_context",
            )
            return output

    # ---- Optimization 5: Merged old-log-prob computation ----

    def _attach_old_log_probs_merged(
        self, ret_batch: DataProto, gen_batch: DataProto, metrics: dict,
    ) -> tuple:
        """Compute old log-probs for *both* batches in a single dispatched forward pass.

        Pads the shorter batch's sequence dimension so the two can be concatenated,
        runs one ``compute_log_prob`` call, then splits the results.  This halves
        the Ray-dispatch overhead and improves GPU utilisation.

        With ``use_remove_padding=True`` (default), padding tokens are stripped
        before the actual forward pass, so extra padding adds negligible compute.
        """
        ret_resp_len = ret_batch.batch["responses"].shape[1]
        gen_resp_len = gen_batch.batch["responses"].shape[1]

        if ret_resp_len != gen_resp_len:
            raise RuntimeError(
                "Merged old-log-prob expects identical response lengths for retriever and generator batches; "
                f"got retriever={ret_resp_len}, generator={gen_resp_len}"
            )

        ret_temperature = float(ret_batch.meta_info["temperature"])
        gen_temperature = float(gen_batch.meta_info["temperature"])
        if abs(ret_temperature - gen_temperature) > 1e-12:
            metrics["twopass/merged_old_log_prob_skipped_temperature_mismatch"] = 1.0
            ret_batch = self._attach_old_log_probs(ret_batch, metrics, "retriever")
            gen_batch = self._attach_old_log_probs(gen_batch, metrics, "generator")
            return ret_batch, gen_batch
        metrics["twopass/merged_old_log_prob_skipped_temperature_mismatch"] = 0.0

        ret_seq_len = ret_batch.batch["input_ids"].shape[1]
        gen_seq_len = gen_batch.batch["input_ids"].shape[1]
        ret_size = ret_batch.batch["input_ids"].shape[0]
        max_seq_len = max(ret_seq_len, gen_seq_len)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def _left_pad(tensor, target, value=0):
            diff = target - tensor.shape[1]
            return F.pad(tensor, (diff, 0), value=value) if diff > 0 else tensor

        # Build a minimal combined DataProto with only the keys compute_log_prob needs
        combined = DataProto.from_dict(tensors={
            "input_ids": torch.cat([
                _left_pad(ret_batch.batch["input_ids"], max_seq_len, pad_id),
                _left_pad(gen_batch.batch["input_ids"], max_seq_len, pad_id),
            ], dim=0),
            "attention_mask": torch.cat([
                _left_pad(ret_batch.batch["attention_mask"], max_seq_len, 0),
                _left_pad(gen_batch.batch["attention_mask"], max_seq_len, 0),
            ], dim=0),
            "position_ids": torch.cat([
                _left_pad(ret_batch.batch["position_ids"], max_seq_len, 0),
                _left_pad(gen_batch.batch["position_ids"], max_seq_len, 0),
            ], dim=0),
            "responses": torch.cat([
                ret_batch.batch["responses"],
                gen_batch.batch["responses"],
            ], dim=0),
        })
        combined.meta_info["temperature"] = ret_temperature

        # Single dispatched forward pass
        log_prob_output = self.actor_rollout_wg.compute_log_prob(combined)

        # Split results back to individual batches
        all_log_probs = log_prob_output.batch["old_log_probs"]
        ret_batch.batch["old_log_probs"] = all_log_probs[:ret_size]
        gen_batch.batch["old_log_probs"] = all_log_probs[ret_size:]

        # Handle entropy metrics
        if "entropys" in log_prob_output.batch:
            all_entropys = log_prob_output.batch["entropys"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            ret_ent = agg_loss(
                loss_mat=all_entropys[:ret_size],
                loss_mask=ret_batch.batch["response_mask"],
                loss_agg_mode=loss_agg_mode,
            )
            gen_ent = agg_loss(
                loss_mat=all_entropys[ret_size:],
                loss_mask=gen_batch.batch["response_mask"],
                loss_agg_mode=loss_agg_mode,
            )
            metrics["retriever/entropy"] = ret_ent.detach().item()
            metrics["generator/entropy"] = gen_ent.detach().item()

        return ret_batch, gen_batch

    def _attach_old_log_probs(self, batch: DataProto, metrics: dict, prefix: str) -> DataProto:
        """Compute old log-probs and entropy metrics."""
        old_log_prob = self.actor_rollout_wg.compute_log_prob(self._log_prob_view(batch))
        if "entropys" in old_log_prob.batch:
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics[f"{prefix}/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys", None)
        return batch.union(old_log_prob)

    def _attach_ref_log_probs(self, batch: DataProto) -> DataProto:
        """Compute reference policy log-probs when needed."""
        log_prob_batch = self._log_prob_view(batch)
        if not self.ref_in_actor:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(log_prob_batch)
        else:
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(log_prob_batch)
        return batch.union(ref_log_prob)

    def _build_generator_batch(self, ret_batch: DataProto) -> DataProto:
        """
        Parse retriever outputs and build generator prompts.

        ``per_retriever`` mode: generator sample i is conditioned on
        retriever sample i % N_ret from the same question group.  This preserves
        the causal pairing required for retriever credit assignment.

        ``merged_union`` mode merges all N_ret retriever outputs per question.
        ``majority_vote`` keeps columns that appear in at least the configured
        vote threshold of retriever outputs. Both group-level modes replicate
        one prompt for every generator sample.

        per_retriever supports N_ret <= N_gen. Group-level modes also support
        N_ret > N_gen because retriever samples are used only to build the
        shared prompt.
        ret_batch has B*N_ret entries; the returned gen_batch has B*N_gen entries.
        """
        responses = ret_batch.batch["responses"]
        batch_size = responses.shape[0]
        N_ret = self.ret_n
        N_gen = self.gen_n
        if N_ret <= 0 or batch_size % N_ret != 0:
            raise ValueError(f"Invalid retriever batch size={batch_size} for ret_n={N_ret}")
        group_level_prompt_modes = {"merged_union", "majority_vote"}
        if N_ret > N_gen and self.generator_prompt_mode not in group_level_prompt_modes:
            raise NotImplementedError(
                f"ret_n={N_ret} > gen_n={N_gen} is not supported by reward mapping"
            )

        # Decode retriever responses
        if "response_mask" not in ret_batch.batch.keys():
            raise RuntimeError("Retriever response_mask is required before building generator prompts")
        ret_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        ret_resp_lens = torch.sum(ret_batch.batch["response_mask"], dim=-1).detach().cpu().tolist()

        # Group samples by question (interleaved: [q1,q1,..., q2,q2,...])
        num_questions = batch_size // N_ret if N_ret > 0 else batch_size
        gen_prompts = []
        gen_ret_texts = []
        gen_ret_resp_lens = []
        gen_prompt_unique_counts = []
        gen_prompt_pred_col_counts = []
        gen_prompt_empty_schema_count = 0
        vote_cutoffs = []
        vote_selected_counts = []
        vote_candidate_counts = []
        vote_empty_group_count = 0
        vote_bucket_totals = {}

        for q_idx in range(num_questions):
            start_idx = q_idx * N_ret
            end_idx = start_idx + N_ret

            # Get first sample's metadata (all samples in group share same question)
            extra_obj = ret_batch.non_tensor_batch["extra_info"][start_idx]
            extra = json.loads(extra_obj) if isinstance(extra_obj, str) else extra_obj

            db_info_obj = extra.get("db_info", {})
            sampled_values_obj = extra.get("sampled_values", {})
            relavant_values_obj = extra.get("relavant_values", {})
            db_info = json.loads(db_info_obj) if isinstance(db_info_obj, str) else db_info_obj
            sampled_values = (
                json.loads(sampled_values_obj) if isinstance(sampled_values_obj, str) else sampled_values_obj
            )
            relavant_values = (
                json.loads(relavant_values_obj) if isinstance(relavant_values_obj, str) else relavant_values_obj
            )

            question = extra.get("question", "")
            evidence = extra.get("evidence", "")

            group_prompt_texts = []
            if self.generator_prompt_mode in group_level_prompt_modes:
                # All generator samples share one schema derived from the retriever group.
                parsed_group_indices = []
                for i in range(start_idx, end_idx):
                    pred_indices = parse_retriever_output(ret_texts[i], db_info)
                    parsed_group_indices.append(pred_indices)

                if self.generator_prompt_mode == "merged_union":
                    selected_indices = set()
                    for pred_indices in parsed_group_indices:
                        selected_indices.update(pred_indices)
                else:
                    vote_counts = {}
                    for pred_indices in parsed_group_indices:
                        for col_idx in pred_indices:
                            vote_counts[col_idx] = vote_counts.get(col_idx, 0) + 1
                    vote_cutoff = max(1, int(np.ceil(N_ret * self.generator_prompt_vote_threshold)))
                    selected_indices = {
                        col_idx for col_idx, votes in vote_counts.items() if votes >= vote_cutoff
                    }
                    vote_cutoffs.append(vote_cutoff)
                    vote_candidate_counts.append(len(vote_counts))
                    vote_selected_counts.append(len(selected_indices))
                    if not selected_indices:
                        vote_empty_group_count += 1
                    for votes in vote_counts.values():
                        vote_bucket_totals[votes] = vote_bucket_totals.get(votes, 0) + 1

                if not selected_indices:
                    gen_prompt_empty_schema_count += N_gen

                gen_prompt_text = build_generator_prompt(
                    question=question,
                    evidence=evidence,
                    db_info=db_info,
                    pred_indices=selected_indices,
                    sampled_values=sampled_values,
                    relavant_values=relavant_values,
                )

                for gen_i in range(N_gen):
                    ret_idx = start_idx + (gen_i % N_ret)
                    gen_prompts.append([{"role": "user", "content": gen_prompt_text}])
                    gen_ret_texts.append(ret_texts[ret_idx])
                    gen_ret_resp_lens.append(int(ret_resp_lens[ret_idx]))
                    gen_prompt_pred_col_counts.append(len(selected_indices))
                    group_prompt_texts.append(gen_prompt_text)
            else:
                for gen_i in range(N_gen):
                    ret_idx = start_idx + (gen_i % N_ret)
                    pred_indices = parse_retriever_output(ret_texts[ret_idx], db_info)
                    if not pred_indices:
                        gen_prompt_empty_schema_count += 1

                    gen_prompt_text = build_generator_prompt(
                        question=question,
                        evidence=evidence,
                        db_info=db_info,
                        pred_indices=pred_indices,
                        sampled_values=sampled_values,
                        relavant_values=relavant_values,
                    )

                    gen_prompts.append([{"role": "user", "content": gen_prompt_text}])
                    gen_ret_texts.append(ret_texts[ret_idx])
                    gen_ret_resp_lens.append(int(ret_resp_lens[ret_idx]))
                    gen_prompt_pred_col_counts.append(len(pred_indices))
                    group_prompt_texts.append(gen_prompt_text)

            gen_prompt_unique_counts.append(len(set(group_prompt_texts)))

        # Batch tokenize. Async agent-loop rollout consumes raw_prompt_ids when
        # present, so keep these ids aligned with the padded tensors below.
        old_padding_side = self.tokenizer.padding_side
        old_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        all_texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in gen_prompts
        ]
        try:
            encoded_full = self.tokenizer(
                all_texts,
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )
            prompt_token_lens_full = [len(ids) for ids in encoded_full["input_ids"]]
            encoded_unpadded = self.tokenizer(
                all_texts,
                max_length=self.gen_max_prompt_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            raw_prompt_ids = [
                [int(token_id) for token_id in ids]
                for ids in encoded_unpadded["input_ids"]
            ]
            prompt_token_lens = [len(ids) for ids in raw_prompt_ids]
            encoded = self.tokenizer(
                all_texts,
                max_length=self.gen_max_prompt_length,
                truncation=True,
                padding="longest",
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
            self.tokenizer.truncation_side = old_truncation_side

        rollout_prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        if len(raw_prompt_ids) != len(gen_prompts):
            raise RuntimeError(
                f"Generator prompt tokenization mismatch: {len(raw_prompt_ids)} raw_prompt_ids for "
                f"{len(gen_prompts)} prompts"
            )
        too_long = [idx for idx, ids in enumerate(raw_prompt_ids) if len(ids) > rollout_prompt_length]
        if too_long:
            example_idx = too_long[0]
            raise RuntimeError(
                f"Generator raw_prompt_ids exceed rollout prompt_length={rollout_prompt_length}; "
                f"first offending sample={example_idx}, length={len(raw_prompt_ids[example_idx])}"
            )

        gen_input_ids = encoded["input_ids"]
        gen_attention_mask = encoded["attention_mask"]
        for sample_idx, ids in enumerate(raw_prompt_ids):
            mask = gen_attention_mask[sample_idx].bool()
            nonpad = [int(token_id) for token_id in gen_input_ids[sample_idx][mask].tolist()]
            if nonpad != ids:
                raise RuntimeError(
                    f"Generator prompt tensor/raw ids mismatch at sample {sample_idx}: "
                    f"tensor_nonpad_len={len(nonpad)}, raw_len={len(ids)}"
                )
        gen_position_ids = gen_attention_mask.cumsum(-1) - 1
        gen_position_ids = gen_position_ids.clamp(min=0)

        gen_batch = DataProto.from_dict(
            tensors={
                "input_ids": gen_input_ids,
                "attention_mask": gen_attention_mask,
                "position_ids": gen_position_ids,
            }
        )

        # Copy required non-tensor fields (remap from B*N_ret to B*N_gen)
        remap_keys = [
            "uid",
            "reward_model",
            "extra_info",
            "data_source",
            "index",
            "agent_name",
            "tools_kwargs",
            "interaction_kwargs",
            "multi_modal_inputs",
            "multi_modal_data",
        ]
        if N_ret == N_gen:
            for key in remap_keys:
                if key in ret_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch[key] = ret_batch.non_tensor_batch[key]
        else:
            for key in remap_keys:
                if key in ret_batch.non_tensor_batch:
                    orig = ret_batch.non_tensor_batch[key]
                    remapped = []
                    for q in range(num_questions):
                        for i in range(N_gen):
                            remapped.append(orig[q * N_ret + (i % N_ret)])
                    gen_batch.non_tensor_batch[key] = np.array(remapped, dtype=orig.dtype)

        gen_batch.non_tensor_batch["raw_prompt"] = np.array(gen_prompts, dtype=object)
        gen_batch.non_tensor_batch["raw_prompt_ids"] = np.array(raw_prompt_ids, dtype=object)
        gen_batch.non_tensor_batch["ret_texts"] = np.array(gen_ret_texts, dtype=object)
        gen_batch.non_tensor_batch["ret_resp_lens"] = np.array(gen_ret_resp_lens, dtype=object)
        gen_batch.meta_info["generator_prompt_mode"] = self.generator_prompt_mode
        gen_batch.meta_info["generator_prompt_unique_per_group_mean"] = float(
            np.mean(gen_prompt_unique_counts) if gen_prompt_unique_counts else 0.0
        )
        gen_batch.meta_info["generator_prompt_full_schema_fallback_ratio"] = 0.0
        gen_batch.meta_info["generator_prompt_empty_schema_ratio"] = float(
            gen_prompt_empty_schema_count / max(len(gen_prompts), 1)
        )
        gen_batch.meta_info["generator_prompt_pred_col_count_mean"] = float(
            np.mean(gen_prompt_pred_col_counts) if gen_prompt_pred_col_counts else 0.0
        )
        gen_batch.meta_info["generator_prompt_vote_cutoff"] = float(
            vote_cutoffs[0] if vote_cutoffs else 0.0
        )
        gen_batch.meta_info["generator_prompt_vote_candidate_col_count_mean"] = float(
            np.mean(vote_candidate_counts) if vote_candidate_counts else 0.0
        )
        gen_batch.meta_info["generator_prompt_vote_selected_col_count_mean"] = float(
            np.mean(vote_selected_counts) if vote_selected_counts else 0.0
        )
        gen_batch.meta_info["generator_prompt_vote_selected_col_count_max"] = float(
            np.max(vote_selected_counts) if vote_selected_counts else 0.0
        )
        gen_batch.meta_info["generator_prompt_vote_empty_group_ratio"] = float(
            vote_empty_group_count / max(num_questions, 1)
        )
        gen_batch.meta_info["generator_prompt_vote_histogram"] = ",".join(
            f"{votes}:{vote_bucket_totals[votes]}"
            for votes in sorted(vote_bucket_totals)
        )
        for votes in range(1, N_ret + 1):
            gen_batch.meta_info[f"generator_prompt_vote_count_{votes}"] = float(
                vote_bucket_totals.get(votes, 0)
            )
        gen_batch.meta_info["generator_prompt_token_len_mean"] = float(
            np.mean(prompt_token_lens) if prompt_token_lens else 0.0
        )
        gen_batch.meta_info["generator_prompt_token_len_max"] = float(
            np.max(prompt_token_lens) if prompt_token_lens else 0.0
        )
        gen_batch.meta_info["generator_prompt_truncated_ratio"] = float(
            np.mean([length > self.gen_max_prompt_length for length in prompt_token_lens_full])
            if prompt_token_lens_full else 0.0
        )

        return gen_batch

    def _place_rewards_on_tokens(self, batch: DataProto, rewards: list[float]) -> None:
        """Place scalar rewards at the last valid token position."""
        response_mask = batch.batch["response_mask"]
        bs, resp_len = response_mask.shape
        token_level_scores = torch.zeros(bs, resp_len, dtype=torch.float32)

        for i in range(bs):
            valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                token_level_scores[i, valid_positions[-1].item()] = rewards[i]

        batch.batch["token_level_scores"] = token_level_scores

    def _phase_for_epoch(self, epoch: int) -> str:
        if epoch < self.retriever_only_warmup_epochs:
            return "retriever"
        schedule_epoch = epoch - self.retriever_only_warmup_epochs
        return self.training_phase_schedule[schedule_epoch % len(self.training_phase_schedule)]

    def _balance_joint_update_batches(
        self,
        ret_batch: DataProto,
        gen_batch: DataProto,
        metrics: dict,
    ) -> None:
        """Use VERL's native seqlen balancing with one shared group order.

        Balancing retriever and generator batches independently can put the
        heaviest retriever partition and the heaviest generator partition on
        the same FSDP rank.  For shared-adapter/full-model joint update, both
        passes are accumulated on the same worker before one optimizer step, so
        the relevant load is the sum of both passes for a question group.
        """
        ret_bs = int(ret_batch.batch.batch_size[0])
        gen_bs = int(gen_batch.batch.batch_size[0])
        if self.ret_n <= 0 or self.gen_n <= 0:
            raise RuntimeError(f"Invalid joint balance n: ret_n={self.ret_n}, gen_n={self.gen_n}")
        if ret_bs % self.ret_n != 0 or gen_bs % self.gen_n != 0:
            raise RuntimeError(
                "Joint update balance expects grouped batches: "
                f"ret_bs={ret_bs}, ret_n={self.ret_n}, gen_bs={gen_bs}, gen_n={self.gen_n}"
            )
        num_questions = ret_bs // self.ret_n
        if num_questions != gen_bs // self.gen_n:
            raise RuntimeError(
                "Joint update balance requires matching question groups: "
                f"ret_groups={num_questions}, gen_groups={gen_bs // self.gen_n}"
            )

        world_size = int(self.actor_rollout_wg.world_size)
        if num_questions < world_size or num_questions % world_size != 0:
            raise RuntimeError(
                "Joint update balance requires train_batch_size to be divisible by FSDP world size "
                "so DataProto.chunk keeps rank boundaries aligned: "
                f"num_questions={num_questions}, world_size={world_size}"
            )

        ret_lens = ret_batch.batch["attention_mask"].view(ret_bs, -1).sum(-1).tolist()
        gen_lens = gen_batch.batch["attention_mask"].view(gen_bs, -1).sum(-1).tolist()
        joint_group_lens = []
        for q_idx in range(num_questions):
            ret_start = q_idx * self.ret_n
            gen_start = q_idx * self.gen_n
            joint_group_lens.append(
                sum(ret_lens[ret_start: ret_start + self.ret_n])
                + sum(gen_lens[gen_start: gen_start + self.gen_n])
            )

        group_partitions = get_seqlen_balanced_partitions(
            joint_group_lens,
            k_partitions=world_size,
            equal_size=True,
        )
        ret_partitions = [
            [q_idx * self.ret_n + i for q_idx in partition for i in range(self.ret_n)]
            for partition in group_partitions
        ]
        gen_partitions = [
            [q_idx * self.gen_n + i for q_idx in partition for i in range(self.gen_n)]
            for partition in group_partitions
        ]
        ret_batch.reorder(torch.tensor([idx for partition in ret_partitions for idx in partition]))
        gen_batch.reorder(torch.tensor([idx for partition in gen_partitions for idx in partition]))

        metrics.update(
            log_seqlen_unbalance(
                seqlen_list=joint_group_lens,
                partitions=group_partitions,
                prefix="joint/global_group_seqlen",
            )
        )
        metrics.update(
            log_seqlen_unbalance(
                seqlen_list=ret_lens,
                partitions=ret_partitions,
                prefix="retriever/global_seqlen",
            )
        )
        metrics.update(
            log_seqlen_unbalance(
                seqlen_list=gen_lens,
                partitions=gen_partitions,
                prefix="generator/global_seqlen",
            )
        )

    def _print_step_timing(self, step: int, phase: str, timing_raw: dict, metrics: dict) -> None:
        """Print a short timing line because WandB output truncates long metric rows."""
        timing_keys = [
            "step",
            "ret_gen",
            "build_gen",
            "gen_gen",
            "old_log_prob",
            "reward",
            "adv",
            "update_actor_joint",
            "update_actor_dual_lora",
            "update_actor_retriever_only",
            "update_actor_generator_only",
            "testing",
            "save_ckpt",
        ]
        parts = []
        for key in timing_keys:
            value = timing_raw.get(key, None)
            if value is not None:
                parts.append(f"{key}={float(value):.1f}s")
        parts.extend([
            f"ret_n={metrics.get('twopass/effective_ret_n', 0.0):.0f}",
            f"gen_n={metrics.get('twopass/effective_gen_n', 0.0):.0f}",
            f"ret_seq={metrics.get('retriever/global_seqlen/mean', 0.0):.0f}",
            f"gen_seq={metrics.get('generator/global_seqlen/mean', 0.0):.0f}",
        ])
        joint_balanced_max = metrics.get("joint/global_group_seqlen/balanced_max", None)
        if joint_balanced_max is not None:
            parts.extend([
                f"joint_group_seq={metrics.get('joint/global_group_seqlen/mean', 0.0):.0f}",
                f"joint_balanced_max={float(joint_balanced_max):.0f}",
                f"joint_balanced_min={metrics.get('joint/global_group_seqlen/balanced_min', 0.0):.0f}",
            ])
        vote_hist = metrics.get("twopass/generator_prompt_vote_histogram", "")
        if vote_hist:
            parts.extend([
                f"vote_cutoff={metrics.get('twopass/generator_prompt_vote_cutoff', 0.0):.0f}",
                f"vote_candidates={metrics.get('twopass/generator_prompt_vote_candidate_col_count_mean', 0.0):.1f}",
                f"vote_selected={metrics.get('twopass/generator_prompt_vote_selected_col_count_mean', 0.0):.1f}",
                f"vote_selected_max={metrics.get('twopass/generator_prompt_vote_selected_col_count_max', 0.0):.0f}",
                f"vote_empty={metrics.get('twopass/generator_prompt_vote_empty_group_ratio', 0.0):.2f}",
                f"vote_hist={vote_hist}",
            ])
        print(f"[TwoPass][timing] step={step} phase={phase} " + " ".join(parts), flush=True)

    @staticmethod
    def _collect_reward_inputs(batch: DataProto) -> tuple[list[dict], list[dict]]:
        ground_truths = []
        extra_infos = []
        bs = int(batch.batch.batch_size[0])
        for i in range(bs):
            rm_obj = batch.non_tensor_batch["reward_model"][i]
            rm = json.loads(rm_obj) if isinstance(rm_obj, str) else rm_obj
            ground_truths.append(rm.get("ground_truth", {}))

            ex_obj = batch.non_tensor_batch["extra_info"][i]
            extra_infos.append(json.loads(ex_obj) if isinstance(ex_obj, str) else ex_obj)
        return ground_truths, extra_infos

    def _single_phase_update(self, phase_batch: DataProto, metrics: dict, timing_raw: dict, prefix: str) -> None:
        """Run one full-parameter PPO update for a single alternating phase."""
        if self.config.trainer.critic_warmup > self.global_steps:
            return

        multi_turn = self.config.actor_rollout_ref.rollout.multi_turn.enable
        phase_batch.meta_info["multi_turn"] = multi_turn
        update_batch = self._actor_update_view(phase_batch)
        with marked_timer(f"update_actor_{prefix}_only", timing_raw):
            if self.dual_lora:
                update_batch.meta_info["adapter_name"] = prefix
                metrics["twopass/single_phase_update_lora_int_id"] = float(
                    1 if prefix == "retriever" else 2
                )
                actor_output = self.actor_rollout_wg.update_actor_single_lora(update_batch)
            else:
                actor_output = self.actor_rollout_wg.update_actor(update_batch)
        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
        del update_batch

    def _finish_single_phase_batch(
        self,
        phase_batch: DataProto,
        rewards: list[float],
        reward_details: list[dict],
        repeat_n: int,
        metrics: dict,
        timing_raw: dict,
        prefix: str,
    ) -> DataProto:
        """Attach rewards/logprobs/advantages and update the full actor once."""
        self._place_rewards_on_tokens(phase_batch, rewards)
        phase_batch.meta_info["global_steps"] = self.global_steps
        phase_batch.meta_info["loss_scale"] = 1.0
        phase_batch.meta_info["ppo_effective_repeat_n"] = int(repeat_n)
        phase_batch.meta_info["ppo_base_repeat_n"] = int(self.gen_n)
        metrics[f"twopass/{prefix}_ppo_effective_repeat_n"] = float(repeat_n)
        metrics[f"twopass/{prefix}_ppo_base_repeat_n"] = float(self.gen_n)

        metrics[f"reward/{prefix[:3]}_mean"] = float(np.mean(rewards)) if rewards else 0.0
        metrics[f"reward/{prefix[:3]}_std"] = float(np.std(rewards)) if rewards else 0.0
        metrics[f"reward/{prefix[:3]}_max"] = float(np.max(rewards)) if rewards else 0.0
        metrics[f"reward/{prefix[:3]}_min"] = float(np.min(rewards)) if rewards else 0.0
        if prefix == "retriever":
            metrics["reward/ret_mean"] = metrics["reward/ret_mean"]
            metrics["reward/gen_mean"] = 0.0
        else:
            metrics["reward/ret_mean"] = 0.0
            metrics["reward/gen_mean"] = metrics["reward/gen_mean"]

        metrics.update(aggregate_reward_details(reward_details, N=repeat_n, prefix="train"))

        with marked_timer("old_log_prob", timing_raw):
            if self.dual_lora:
                phase_batch.meta_info["adapter_name"] = prefix
                metrics["twopass/single_phase_old_log_prob_lora_int_id"] = float(
                    1 if prefix == "retriever" else 2
                )
            phase_batch = self._attach_old_log_probs(phase_batch, metrics, prefix)

        if self.use_reference_policy:
            with marked_timer("ref_log_prob", timing_raw):
                phase_batch = self._attach_ref_log_probs(phase_batch)

        with marked_timer("adv", timing_raw):
            if self.config.algorithm.use_kl_in_reward:
                phase_batch, phase_kl = apply_kl_penalty(
                    phase_batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics[f"{prefix}/reward_kl_penalty"] = phase_kl["actor/reward_kl_penalty"]
            else:
                phase_batch.batch["token_level_rewards"] = phase_batch.batch["token_level_scores"]

            norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            phase_batch = compute_advantage(
                phase_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=repeat_n,
                norm_adv_by_std_in_grpo=norm_adv,
                config=self.config.algorithm,
            )
            self._add_grpo_diagnostics(phase_batch, metrics, prefix)

        if self.config.trainer.balance_batch:
            self._balance_batch(
                phase_batch,
                metrics=metrics,
                logging_prefix=f"{prefix}/global_seqlen",
            )

        phase_batch.meta_info["global_token_num"] = torch.sum(
            phase_batch.batch["attention_mask"], dim=-1
        ).tolist()

        self._single_phase_update(phase_batch, metrics, timing_raw, prefix)
        return phase_batch

    def _run_retriever_only_step(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        N_ret: int,
    ) -> DataProto:
        """Alternating epoch phase: rollout and train only retriever responses."""
        ret_temperature = self._rollout_temperature("retriever", "retriever")
        metrics["temperature/retriever_rollout"] = ret_temperature
        self._enter_rollout_context()
        try:
            with marked_timer("ret_gen", timing_raw):
                ret_rollout_input = self._prepare_rollout_batch(batch)
                ret_rollout_input.meta_info["global_steps"] = self.global_steps
                ret_rollout_input.meta_info["temperature"] = ret_temperature
                self._set_rollout_response_length(
                    ret_rollout_input, "retriever", metrics, "rollout/retriever_only_response_length"
                )
                if self.dual_lora:
                    ret_rollout_input.meta_info["_lora_int_id"] = 1
                    metrics["twopass/retriever_rollout_lora_int_id"] = 1.0
                ret_rollout_input = ret_rollout_input.repeat(repeat_times=N_ret, interleave=True)
                ret_rollout_output = self._generate_in_context(ret_rollout_input)
                if "timing" in ret_rollout_output.meta_info:
                    timing_raw.update(ret_rollout_output.meta_info["timing"])
                    ret_rollout_output.meta_info.pop("timing", None)

            ret_batch = batch.repeat(repeat_times=N_ret, interleave=True)
            for key, val in ret_batch.non_tensor_batch.items():
                if key not in ret_rollout_output.non_tensor_batch:
                    ret_rollout_output.non_tensor_batch[key] = val
            ret_batch = ret_rollout_output
            if "response_mask" not in ret_batch.batch.keys():
                ret_batch.batch["response_mask"] = compute_response_mask(ret_batch)
            ret_batch.meta_info["temperature"] = ret_temperature
            del ret_rollout_input, ret_rollout_output
        finally:
            self._exit_rollout_context()

        ret_texts = self.tokenizer.batch_decode(ret_batch.batch["responses"], skip_special_tokens=True)
        ret_resp_lens = torch.sum(ret_batch.batch["response_mask"], dim=-1).tolist()
        ground_truths, extra_infos = self._collect_reward_inputs(ret_batch)
        ret_rewards, reward_details = self.reward_manager.compute_retriever_only_rewards(
            ret_texts=ret_texts,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            group_ids=list(ret_batch.non_tensor_batch["uid"]),
            repeat_n=N_ret,
            ret_resp_lens=ret_resp_lens,
            ret_response_length=int(ret_batch.batch["responses"].shape[1]),
        )
        metrics["twopass/skipped_generator_rollout"] = 1.0
        metrics["twopass/phase_retriever_only"] = 1.0
        metrics["twopass/phase_generator_only"] = 0.0
        return self._finish_single_phase_batch(
            ret_batch,
            rewards=ret_rewards,
            reward_details=reward_details,
            repeat_n=N_ret,
            metrics=metrics,
            timing_raw=timing_raw,
            prefix="retriever",
        )

    def _run_generator_only_step(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        N_ret: int,
        N_gen: int,
    ) -> DataProto:
        """Alternating epoch phase: retrieve context, then train only SQL generation."""
        ret_temperature = self._rollout_temperature("generator", "retriever")
        gen_temperature = self._rollout_temperature("generator", "generator")
        metrics["temperature/retriever_rollout"] = ret_temperature
        metrics["temperature/generator_rollout"] = gen_temperature
        self._enter_rollout_context()
        try:
            with marked_timer("ret_gen", timing_raw):
                ret_rollout_input = self._prepare_rollout_batch(batch)
                ret_rollout_input.meta_info["global_steps"] = self.global_steps
                ret_rollout_input.meta_info["temperature"] = ret_temperature
                self._set_rollout_response_length(
                    ret_rollout_input, "retriever", metrics, "rollout/retriever_response_length"
                )
                if self.dual_lora:
                    ret_rollout_input.meta_info["_lora_int_id"] = 1
                    metrics["twopass/retriever_rollout_lora_int_id"] = 1.0
                ret_rollout_input = ret_rollout_input.repeat(repeat_times=N_ret, interleave=True)
                ret_rollout_output = self._generate_in_context(ret_rollout_input)
                if "timing" in ret_rollout_output.meta_info:
                    timing_raw.update(ret_rollout_output.meta_info["timing"])
                    ret_rollout_output.meta_info.pop("timing", None)

            ret_batch = batch.repeat(repeat_times=N_ret, interleave=True)
            for key, val in ret_batch.non_tensor_batch.items():
                if key not in ret_rollout_output.non_tensor_batch:
                    ret_rollout_output.non_tensor_batch[key] = val
            ret_batch = ret_rollout_output
            if "response_mask" not in ret_batch.batch.keys():
                ret_batch.batch["response_mask"] = compute_response_mask(ret_batch)
            ret_batch.meta_info["temperature"] = ret_temperature
            del ret_rollout_input, ret_rollout_output

            with marked_timer("build_gen", timing_raw):
                gen_input = self._build_generator_batch(ret_batch)
                metrics["twopass/generator_prompt_mode_per_retriever"] = float(
                    gen_input.meta_info.get("generator_prompt_mode") == "per_retriever"
                )
                metrics["twopass/generator_prompt_unique_per_group_mean"] = gen_input.meta_info.get(
                    "generator_prompt_unique_per_group_mean", 0.0
                )
                metrics["twopass/generator_prompt_full_schema_fallback_ratio"] = gen_input.meta_info.get(
                    "generator_prompt_full_schema_fallback_ratio", 0.0
                )
                metrics["twopass/generator_prompt_empty_schema_ratio"] = gen_input.meta_info.get(
                    "generator_prompt_empty_schema_ratio", 0.0
                )
                metrics["twopass/generator_prompt_pred_col_count_mean"] = gen_input.meta_info.get(
                    "generator_prompt_pred_col_count_mean", 0.0
                )
                metrics["twopass/generator_prompt_vote_cutoff"] = gen_input.meta_info.get(
                    "generator_prompt_vote_cutoff", 0.0
                )
                metrics["twopass/generator_prompt_vote_candidate_col_count_mean"] = gen_input.meta_info.get(
                    "generator_prompt_vote_candidate_col_count_mean", 0.0
                )
                metrics["twopass/generator_prompt_vote_selected_col_count_mean"] = gen_input.meta_info.get(
                    "generator_prompt_vote_selected_col_count_mean", 0.0
                )
                metrics["twopass/generator_prompt_vote_selected_col_count_max"] = gen_input.meta_info.get(
                    "generator_prompt_vote_selected_col_count_max", 0.0
                )
                metrics["twopass/generator_prompt_vote_empty_group_ratio"] = gen_input.meta_info.get(
                    "generator_prompt_vote_empty_group_ratio", 0.0
                )
                metrics["twopass/generator_prompt_vote_histogram"] = gen_input.meta_info.get(
                    "generator_prompt_vote_histogram", ""
                )
                for votes in range(1, self.ret_n + 1):
                    metrics[f"twopass/generator_prompt_vote_count_{votes}"] = gen_input.meta_info.get(
                        f"generator_prompt_vote_count_{votes}", 0.0
                    )
                metrics["twopass/generator_prompt_token_len_mean"] = gen_input.meta_info.get(
                    "generator_prompt_token_len_mean", 0.0
                )
                metrics["twopass/generator_prompt_token_len_max"] = gen_input.meta_info.get(
                    "generator_prompt_token_len_max", 0.0
                )
                metrics["twopass/generator_prompt_truncated_ratio"] = gen_input.meta_info.get(
                    "generator_prompt_truncated_ratio", 0.0
                )

            with marked_timer("gen_gen", timing_raw):
                gen_input.meta_info["temperature"] = gen_temperature
                gen_input.meta_info["global_steps"] = self.global_steps
                self._set_rollout_response_length(
                    gen_input, "generator", metrics, "rollout/generator_response_length"
                )
                if self.dual_lora:
                    gen_input.meta_info["_lora_int_id"] = 2
                    metrics["twopass/generator_rollout_lora_int_id"] = 2.0
                gen_rollout_output = self._generate_in_context(gen_input)
                if "timing" in gen_rollout_output.meta_info:
                    timing_raw.update(gen_rollout_output.meta_info["timing"])
                    gen_rollout_output.meta_info.pop("timing", None)
        finally:
            self._exit_rollout_context()

        gen_batch = gen_rollout_output
        for key, val in gen_input.non_tensor_batch.items():
            if key not in gen_batch.non_tensor_batch:
                gen_batch.non_tensor_batch[key] = val
        if "response_mask" not in gen_batch.batch.keys():
            gen_batch.batch["response_mask"] = compute_response_mask(gen_batch)
        gen_batch.meta_info["temperature"] = gen_temperature
        del gen_input, gen_rollout_output, ret_batch

        gen_texts = self.tokenizer.batch_decode(gen_batch.batch["responses"], skip_special_tokens=True)
        gen_resp_lens = torch.sum(gen_batch.batch["response_mask"], dim=-1).tolist()
        ground_truths, _extra_infos = self._collect_reward_inputs(gen_batch)
        gen_rewards, reward_details = self.reward_manager.compute_generator_only_rewards(
            gen_texts=gen_texts,
            ground_truths=ground_truths,
            gen_resp_lens=gen_resp_lens,
        )
        metrics["twopass/skipped_generator_rollout"] = 0.0
        metrics["twopass/phase_retriever_only"] = 0.0
        metrics["twopass/phase_generator_only"] = 1.0
        return self._finish_single_phase_batch(
            gen_batch,
            rewards=gen_rewards,
            reward_details=reward_details,
            repeat_n=N_gen,
            metrics=metrics,
            timing_raw=timing_raw,
            prefix="generator",
        )

    def _add_grpo_diagnostics(self, batch: DataProto, metrics: dict, prefix: str) -> None:
        """Log group-level reward and advantage scale after GRPO normalization."""
        response_mask = batch.batch["response_mask"].bool()
        advantages = batch.batch.get("advantages", None)
        token_rewards = batch.batch.get("token_level_rewards", None)
        if advantages is None or token_rewards is None:
            return

        valid_adv = torch.masked_select(advantages.detach().float(), response_mask)
        if valid_adv.numel() > 0:
            metrics[f"{prefix}/adv/abs_mean"] = valid_adv.abs().mean().item()
            metrics[f"{prefix}/adv/max"] = valid_adv.max().item()
            metrics[f"{prefix}/adv/min"] = valid_adv.min().item()

        if "uid" not in batch.non_tensor_batch:
            return

        scores = token_rewards.detach().float().sum(dim=-1).cpu().numpy()
        per_sample_adv = []
        adv_cpu = advantages.detach().float().cpu()
        mask_cpu = response_mask.cpu()
        for i in range(adv_cpu.shape[0]):
            valid = mask_cpu[i]
            if bool(valid.any()):
                per_sample_adv.append(float(adv_cpu[i][valid].mean().item()))
            else:
                per_sample_adv.append(0.0)

        grouped = {}
        ordered_uids = []
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            key = str(uid)
            if key not in grouped:
                grouped[key] = []
                ordered_uids.append(key)
            grouped[key].append(idx)

        reward_stds = []
        adv_stds = []
        for uid in ordered_uids:
            indices = grouped[uid]
            if len(indices) <= 1:
                continue
            reward_stds.append(float(np.std([scores[i] for i in indices], ddof=1)))
            adv_stds.append(float(np.std([per_sample_adv[i] for i in indices], ddof=1)))

        if reward_stds:
            metrics[f"{prefix}/grpo/reward_group_std_mean"] = float(np.mean(reward_stds))
            metrics[f"{prefix}/grpo/reward_group_std_min"] = float(np.min(reward_stds))
            metrics[f"{prefix}/grpo/reward_group_zero_std_ratio"] = float(
                np.mean([std <= 1e-8 for std in reward_stds])
            )
        if adv_stds:
            metrics[f"{prefix}/grpo/adv_group_std_mean"] = float(np.mean(adv_stds))
            metrics[f"{prefix}/grpo/adv_group_std_max"] = float(np.max(adv_stds))

    def _actor_update_view(self, batch: DataProto) -> DataProto:
        """Return the minimal DataProto needed by actor update workers.

        Keeping reward/debug tensors on the driver avoids sending them through
        Ray and, under sequence-parallel preprocessing, avoids moving unused
        tensors to GPU.
        """
        batch_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            batch_keys.append("ref_log_prob")
        batch_keys = [key for key in batch_keys if key in batch.batch.keys()]
        non_tensor_keys = [
            key for key in ["multi_modal_inputs"]
            if key in batch.non_tensor_batch
        ]
        return batch.select(
            batch_keys=batch_keys,
            non_tensor_batch_keys=non_tensor_keys,
        )

    def _log_prob_view(self, batch: DataProto) -> DataProto:
        """Return the minimal DataProto needed for actor/ref log-prob workers."""
        batch_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        batch_keys = [key for key in batch_keys if key in batch.batch.keys()]
        non_tensor_keys = [
            key for key in ["multi_modal_inputs"]
            if key in batch.non_tensor_batch
        ]
        return batch.select(
            batch_keys=batch_keys,
            non_tensor_batch_keys=non_tensor_keys,
        )

    def fit(self):
        """Two-pass GRPO training loop for VERL 0.5.5."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        if self.config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError(
                "Two-pass joint update currently requires fsdp/fsdp2, "
                "because update_actor_joint is only implemented on fsdp workers."
            )

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # Optional validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Two-Pass GRPO")

        # Start from step 1 to align with VERL trainer conventions
        self.global_steps += 1
        self.max_steps_duration = 0
        N_gen = self.gen_n
        N_ret = self.ret_n
        N_retriever_only = self.retriever_only_ret_n

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                profile_steps = self.config.trainer.get("profile_steps", None)
                do_profile = (
                    self.global_steps in profile_steps
                    if profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Assign UIDs for GRPO grouping if absent
                if "uid" not in batch.non_tensor_batch:
                    batch_size = int(batch.batch.batch_size[0])
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(batch_size)],
                        dtype=object,
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                phase = self._phase_for_epoch(epoch)
                ret_temperature = self._rollout_temperature(phase, "retriever")
                gen_temperature = (
                    self._rollout_temperature(phase, "generator")
                    if phase in {"joint", "generator"}
                    else ret_temperature
                )
                batch.meta_info["temperature"] = ret_temperature
                metrics["twopass/phase_joint"] = float(phase == "joint")
                metrics["twopass/phase_retriever"] = float(phase == "retriever")
                metrics["twopass/phase_generator"] = float(phase == "generator")
                metrics["temperature/retriever_rollout"] = ret_temperature
                metrics["temperature/generator_rollout"] = gen_temperature
                metrics["twopass/joint_ret_n"] = float(N_ret)
                metrics["twopass/gen_n"] = float(N_gen)
                metrics["twopass/retriever_only_ret_n"] = float(N_retriever_only)
                metrics["twopass/effective_ret_n"] = float(
                    N_retriever_only if phase == "retriever" else N_ret
                )
                metrics["twopass/effective_gen_n"] = float(0 if phase == "retriever" else N_gen)

                if phase != "joint":
                    with marked_timer("step", timing_raw):
                        if phase == "retriever":
                            phase_batch = self._run_retriever_only_step(
                                batch=batch,
                                metrics=metrics,
                                timing_raw=timing_raw,
                                N_ret=N_retriever_only,
                            )
                            phase_prefix = "retriever"
                        elif phase == "generator":
                            phase_batch = self._run_generator_only_step(
                                batch=batch,
                                metrics=metrics,
                                timing_raw=timing_raw,
                                N_ret=N_ret,
                                N_gen=N_gen,
                            )
                            phase_prefix = "generator"
                        else:
                            raise ValueError(f"Unexpected training phase: {phase}")

                        esi_close = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close
                        ):
                            with marked_timer("save_ckpt", timing_raw):
                                self._save_checkpoint()

                        if self.config.trainer.test_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                        ):
                            with marked_timer("testing", timing_raw):
                                val_metrics = self._validate()
                            if val_metrics:
                                metrics.update(val_metrics)

                    with marked_timer("stop_profile", timing_raw):
                        self._stop_profiling(do_profile)

                    steps_duration = timing_raw.get("step", 0.0)
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                    metrics["training/global_step"] = self.global_steps
                    metrics["training/epoch"] = epoch
                    for k, v in compute_data_metrics(batch=phase_batch, use_critic=False).items():
                        metrics[f"{phase_prefix}/{k}"] = v
                    metrics.update(compute_timing_metrics(batch=phase_batch, timing_raw=timing_raw))
                    metrics.update(
                        compute_throughout_metrics(
                            batch=phase_batch,
                            timing_raw=timing_raw,
                            n_gpus=self.resource_pool_manager.get_n_gpus(),
                        )
                    )

                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=phase_batch)

                    if hasattr(self.train_dataset, "on_batch_end"):
                        self.train_dataset.on_batch_end(batch=phase_batch)

                    self._print_step_timing(self.global_steps, phase, timing_raw, metrics)

                    del phase_batch, batch
                    gc.collect()

                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        ret_r=f"{metrics.get('reward/ret_mean', 0.0):.2f}",
                        gen_r=f"{metrics.get('reward/gen_mean', 0.0):.2f}",
                    )

                    self.global_steps += 1
                    if is_last_step:
                        progress_bar.close()
                        return
                    continue

                with marked_timer("step", timing_raw):
                    # Enter rollout context once for both passes (avoids redundant weight sync)
                    self._enter_rollout_context()
                    try:
                        # ========== Phase 1: Retriever Rollout ==========
                        with marked_timer("ret_gen", timing_raw):
                            ret_rollout_input = self._prepare_rollout_batch(batch)
                            ret_rollout_input.meta_info["global_steps"] = self.global_steps
                            ret_rollout_input.meta_info["temperature"] = ret_temperature
                            self._set_rollout_response_length(
                                ret_rollout_input, "retriever", metrics, "rollout/retriever_response_length"
                            )
                            if self.dual_lora:
                                ret_rollout_input.meta_info["_lora_int_id"] = 1  # retriever adapter
                                metrics["twopass/retriever_rollout_lora_int_id"] = 1.0
                            ret_rollout_input = ret_rollout_input.repeat(repeat_times=N_ret, interleave=True)
                            ret_rollout_output = self._generate_in_context(ret_rollout_input)
                            if "timing" in ret_rollout_output.meta_info:
                                timing_raw.update(ret_rollout_output.meta_info["timing"])
                                ret_rollout_output.meta_info.pop("timing", None)

                        ret_batch = batch.repeat(repeat_times=N_ret, interleave=True)
                        # async rollout returns full tensors; merge non_tensor_batch from original batch
                        for key, val in ret_batch.non_tensor_batch.items():
                            if key not in ret_rollout_output.non_tensor_batch:
                                ret_rollout_output.non_tensor_batch[key] = val
                        ret_batch = ret_rollout_output
                        if "response_mask" not in ret_batch.batch.keys():
                            ret_batch.batch["response_mask"] = compute_response_mask(ret_batch)
                        del ret_rollout_input, ret_rollout_output

                        # === DEBUG: Save I/O to JSON (first 2 steps) ===
                        if self.debug_dump_artifacts and self.global_steps <= 2:
                            _debug_io = []
                            _ret_prompts = ret_batch.batch.get("prompts", None)
                            _ret_responses = ret_batch.batch.get("responses", None)
                            if _ret_responses is not None:
                                for _di in range(_ret_responses.shape[0]):
                                    _p = self.tokenizer.decode(_ret_prompts[_di][_ret_prompts[_di] != self.tokenizer.pad_token_id], skip_special_tokens=True) if _ret_prompts is not None else ""
                                    _r = self.tokenizer.decode(_ret_responses[_di], skip_special_tokens=True)
                                    _debug_io.append({"step": self.global_steps, "phase": "retriever", "sample_idx": _di, "prompt": _p, "response": _r})
                        ret_batch.meta_info["temperature"] = ret_temperature

                        # ========== Phase 2: Build Generator Prompts ==========
                        with marked_timer("build_gen", timing_raw):
                            gen_input = self._build_generator_batch(ret_batch)
                            metrics["twopass/generator_prompt_mode_per_retriever"] = float(
                                gen_input.meta_info.get("generator_prompt_mode") == "per_retriever"
                            )
                            metrics["twopass/generator_prompt_unique_per_group_mean"] = gen_input.meta_info.get(
                                "generator_prompt_unique_per_group_mean", 0.0
                            )
                            metrics["twopass/generator_prompt_full_schema_fallback_ratio"] = gen_input.meta_info.get(
                                "generator_prompt_full_schema_fallback_ratio", 0.0
                            )
                            metrics["twopass/generator_prompt_empty_schema_ratio"] = gen_input.meta_info.get(
                                "generator_prompt_empty_schema_ratio", 0.0
                            )
                            metrics["twopass/generator_prompt_pred_col_count_mean"] = gen_input.meta_info.get(
                                "generator_prompt_pred_col_count_mean", 0.0
                            )
                            metrics["twopass/generator_prompt_vote_cutoff"] = gen_input.meta_info.get(
                                "generator_prompt_vote_cutoff", 0.0
                            )
                            metrics["twopass/generator_prompt_vote_candidate_col_count_mean"] = gen_input.meta_info.get(
                                "generator_prompt_vote_candidate_col_count_mean", 0.0
                            )
                            metrics["twopass/generator_prompt_vote_selected_col_count_mean"] = gen_input.meta_info.get(
                                "generator_prompt_vote_selected_col_count_mean", 0.0
                            )
                            metrics["twopass/generator_prompt_vote_selected_col_count_max"] = gen_input.meta_info.get(
                                "generator_prompt_vote_selected_col_count_max", 0.0
                            )
                            metrics["twopass/generator_prompt_vote_empty_group_ratio"] = gen_input.meta_info.get(
                                "generator_prompt_vote_empty_group_ratio", 0.0
                            )
                            metrics["twopass/generator_prompt_vote_histogram"] = gen_input.meta_info.get(
                                "generator_prompt_vote_histogram", ""
                            )
                            for votes in range(1, self.ret_n + 1):
                                metrics[f"twopass/generator_prompt_vote_count_{votes}"] = gen_input.meta_info.get(
                                    f"generator_prompt_vote_count_{votes}", 0.0
                                )
                            metrics["twopass/generator_prompt_token_len_mean"] = gen_input.meta_info.get(
                                "generator_prompt_token_len_mean", 0.0
                            )
                            metrics["twopass/generator_prompt_token_len_max"] = gen_input.meta_info.get(
                                "generator_prompt_token_len_max", 0.0
                            )
                            metrics["twopass/generator_prompt_truncated_ratio"] = gen_input.meta_info.get(
                                "generator_prompt_truncated_ratio", 0.0
                            )

                        # ========== Phase 3: Generator Rollout ==========
                        with marked_timer("gen_gen", timing_raw):
                            gen_input.meta_info["temperature"] = gen_temperature
                            gen_input.meta_info["global_steps"] = self.global_steps
                            self._set_rollout_response_length(
                                gen_input, "generator", metrics, "rollout/generator_response_length"
                            )
                            if self.dual_lora:
                                gen_input.meta_info["_lora_int_id"] = 2  # generator adapter
                                metrics["twopass/generator_rollout_lora_int_id"] = 2.0
                            gen_rollout_output = self._generate_in_context(gen_input)
                            if "timing" in gen_rollout_output.meta_info:
                                timing_raw.update(gen_rollout_output.meta_info["timing"])
                                gen_rollout_output.meta_info.pop("timing", None)
                    finally:
                        self._exit_rollout_context()

                    # async rollout returns full input_ids/attention_mask/responses,
                    # so use it directly and attach non_tensor_batch from gen_input
                    gen_batch = gen_rollout_output
                    for key, val in gen_input.non_tensor_batch.items():
                        if key not in gen_batch.non_tensor_batch:
                            gen_batch.non_tensor_batch[key] = val
                    if "response_mask" not in gen_batch.batch.keys():
                        gen_batch.batch["response_mask"] = compute_response_mask(gen_batch)
                    del gen_input, gen_rollout_output

                    # === DEBUG: Generator I/O (first 2 steps) ===
                    if self.debug_dump_artifacts and self.global_steps <= 2:
                        _gen_prompts = gen_batch.batch.get("prompts", None)
                        _gen_responses = gen_batch.batch.get("responses", None)
                        if _gen_responses is not None:
                            for _di in range(_gen_responses.shape[0]):
                                _p = self.tokenizer.decode(_gen_prompts[_di][_gen_prompts[_di] != self.tokenizer.pad_token_id], skip_special_tokens=True) if _gen_prompts is not None else ""
                                _r = self.tokenizer.decode(_gen_responses[_di], skip_special_tokens=True)
                                _debug_io.append({"step": self.global_steps, "phase": "generator", "sample_idx": _di, "prompt": _p, "response": _r})

                        # Save I/O to JSON
                        _io_path = os.path.join(PROJECT_ROOT, "scripts", f"debug_io_step{self.global_steps}.json")
                        with open(_io_path, "w") as f:
                            json.dump(_debug_io, f, indent=2, ensure_ascii=False)
                    gen_batch.meta_info["temperature"] = gen_temperature

                    # ========== Phase 4+5+6: Reward (CPU) parallel with Log Probs (GPU) ==========
                    # Prepare reward inputs (CPU decode, no GPU needed)
                    gen_texts = self.tokenizer.batch_decode(gen_batch.batch["responses"], skip_special_tokens=True)
                    ret_texts = list(gen_batch.non_tensor_batch["ret_texts"])
                    ret_resp_lens = list(gen_batch.non_tensor_batch["ret_resp_lens"])
                    gen_resp_lens = torch.sum(gen_batch.batch["response_mask"], dim=-1).tolist()

                    ground_truths = []
                    extra_infos = []
                    bs = len(ret_texts)
                    for i in range(bs):
                        rm_obj = gen_batch.non_tensor_batch["reward_model"][i]
                        rm = json.loads(rm_obj) if isinstance(rm_obj, str) else rm_obj
                        ground_truths.append(rm.get("ground_truth", {}))

                        ex_obj = gen_batch.non_tensor_batch["extra_info"][i]
                        extra_infos.append(json.loads(ex_obj) if isinstance(ex_obj, str) else ex_obj)

                    # Submit reward to background thread (CPU-bound SQL execution)
                    reward_executor = ThreadPoolExecutor(max_workers=1)
                    reward_future = reward_executor.submit(
                        self.reward_manager.compute_rewards,
                        ret_texts=ret_texts,
                        gen_texts=gen_texts,
                        ground_truths=ground_truths,
                        extra_infos=extra_infos,
                        group_ids=list(gen_batch.non_tensor_batch["uid"]),
                        ret_n=N_ret,
                        gen_resp_lens=gen_resp_lens,
                        ret_resp_lens=ret_resp_lens,
                        ret_response_length=int(ret_batch.batch["responses"].shape[1]),
                    )

                    # GPU: old log probs
                    with marked_timer("old_log_prob", timing_raw):
                        if self.dual_lora:
                            ret_batch.meta_info["adapter_name"] = "retriever"
                            gen_batch.meta_info["adapter_name"] = "generator"
                            metrics["twopass/retriever_old_log_prob_lora_int_id"] = 1.0
                            metrics["twopass/generator_old_log_prob_lora_int_id"] = 2.0
                            ret_batch = self._attach_old_log_probs(ret_batch, metrics, "retriever")
                            gen_batch = self._attach_old_log_probs(gen_batch, metrics, "generator")
                        else:
                            ret_batch, gen_batch = self._attach_old_log_probs_merged(
                                ret_batch, gen_batch, metrics,
                            )

                    # GPU: reference log probs
                    if self.use_reference_policy:
                        with marked_timer("ref_log_prob", timing_raw):
                            ret_batch = self._attach_ref_log_probs(ret_batch)
                            gen_batch = self._attach_ref_log_probs(gen_batch)

                    # Collect reward results
                    with marked_timer("reward", timing_raw):
                        ret_rewards, gen_rewards, reward_details = reward_future.result()
                        reward_executor.shutdown(wait=False)

                        # gen_rewards → gen_batch (B*N_gen), direct match
                        self._place_rewards_on_tokens(gen_batch, gen_rewards)

                        # ret_rewards → ret_batch (B*N_ret)
                        # Reward function returns B*N_gen ret_rewards (cycled ret_texts).
                        # Take the first N_ret from each group to map back.
                        if N_ret == N_gen:
                            ret_rewards_mapped = ret_rewards
                        else:
                            num_q = len(ret_rewards) // N_gen
                            ret_rewards_mapped = []
                            for q in range(num_q):
                                for i in range(N_ret):
                                    ret_rewards_mapped.append(ret_rewards[q * N_gen + i])
                        self._place_rewards_on_tokens(ret_batch, ret_rewards_mapped)

                        metrics["reward/ret_mean"] = float(np.mean(ret_rewards_mapped))
                        metrics["reward/gen_mean"] = float(np.mean(gen_rewards))
                        metrics["reward/ret_std"] = float(np.std(ret_rewards_mapped))
                        metrics["reward/gen_std"] = float(np.std(gen_rewards))
                        metrics["reward/ret_max"] = float(np.max(ret_rewards_mapped))
                        metrics["reward/ret_min"] = float(np.min(ret_rewards_mapped))
                        metrics["reward/gen_max"] = float(np.max(gen_rewards))
                        metrics["reward/gen_min"] = float(np.min(gen_rewards))

                        # Aggregate detailed reward metrics for WandB
                        metrics.update(aggregate_reward_details(reward_details, N=N_gen, prefix="train"))

                        # Pool statistics
                        pool = self.reward_manager.col_set_pool
                        metrics["pool/num_questions"] = len(pool)
                        if pool:
                            pool_sizes = [len(v) for v in pool.values()]
                            metrics["pool/sets_per_question_mean"] = float(np.mean(pool_sizes))
                            metrics["pool/total_sets"] = sum(pool_sizes)

                        if self.debug_print:
                            print(f"\n[TwoPass][Step {self.global_steps}] bs={bs}, N_gen={N_gen}, N_ret={N_ret}", flush=True)
                            for idx in range(min(bs, self.debug_print_max_samples)):
                                d = reward_details[idx]
                                print(
                                    f"  sample={idx} db={d['db_id']} "
                                    f"sql={d['sql_reward']:.2f} ret={d['ret_reward']:.2f}",
                                    flush=True,
                                )

                    # === DEBUG: Reward breakdown by group (first 2 steps) ===
                    if self.debug_dump_artifacts and self.global_steps <= 2:
                        _debug_path = os.path.join(PROJECT_ROOT, "scripts", f"debug_rewards_step{self.global_steps}.txt")
                        with open(_debug_path, "w") as _df:
                            num_groups = bs // N_gen

                            # ── Global Summary ──
                            all_sql = [reward_details[i]['sql_reward'] for i in range(bs)]
                            all_ret = [reward_details[i]['ret_reward'] for i in range(bs)]
                            _df.write(f"{'='*90}\n")
                            _df.write(f"STEP {self.global_steps}  |  {num_groups} groups x N_gen={N_gen} N_ret={N_ret}  |  {bs} samples\n")
                            _df.write(f"SQL  avg={np.mean(all_sql):+.2f}  min={np.min(all_sql):+.2f}  max={np.max(all_sql):+.2f}\n")
                            _df.write(f"RET  avg={np.mean(all_ret):+.2f}  min={np.min(all_ret):+.2f}  max={np.max(all_ret):+.2f}\n")
                            exec_stats = [reward_details[i]['exec_status'] for i in range(bs)]
                            for st in ["Match", "Mismatch", "Unexecutable", "N/A"]:
                                cnt = sum(1 for s in exec_stats if s == st)
                                if cnt > 0:
                                    _df.write(f"  {st}: {cnt}/{bs} ({cnt/bs:.0%})")
                            _df.write(f"\n{'='*90}\n\n")

                            # ── Per-Group Detail ──
                            for g in range(num_groups):
                                base = g * N_gen
                                d0 = reward_details[base]
                                g_sql = [reward_details[base+s]['sql_reward'] for s in range(N_gen)]
                                g_ret = [reward_details[base+s]['ret_reward'] for s in range(N_gen)]
                                g_exec = [reward_details[base+s]['exec_status'] for s in range(N_gen)]
                                n_match = sum(1 for s in g_exec if s == "Match")

                                _df.write(f"┌{'─'*88}┐\n")
                                _df.write(f"│ GROUP {g:>2d}  DB: {d0['db_id'][:60]:<60s}  Match: {n_match}/{N_gen} │\n")
                                _df.write(f"│ Gold SQL: {str(d0['gold_sql'])[:76]:<76s} │\n")
                                _df.write(f"├{'─'*88}┤\n")

                                # Quick reward bar for the group
                                _df.write(f"│ {'':4s}{'SQL':>8s} {'RET':>8s} │ {'exec':<13s} │ {'cov':>5s} {'irrel':>5s} {'noise':>5s} │ {'pred':>4s}/{'gold':>4s} │\n")
                                _df.write(f"│ {'':4s}{'────────':>8s} {'────────':>8s} │ {'─────────────':<13s} │ {'─────':>5s} {'─────':>5s} {'─────':>5s} │ {'────':>4s}/{'────':>4s} │\n")

                                for s in range(N_gen):
                                    idx = base + s
                                    d = reward_details[idx]
                                    _df.write(
                                        f"│ S{s}: {d['sql_reward']:+7.2f} {d['ret_reward']:+7.2f}"
                                        f" │ {d['exec_status']:<13s}"
                                        f" │ {d['coverage_rate']:5.2f} {d['irrelevant_count']:5d} {d['noise_count']:5d}"
                                        f" │ {d['pred_col_cnt']:4d}/{d['gold_col_cnt']:4d} │\n"
                                    )

                                _df.write(f"├{'─'*88}┤\n")

                                # Per-sample detail
                                for s in range(N_gen):
                                    idx = base + s
                                    d = reward_details[idx]
                                    _df.write(f"│ S{s} SQL: fmt={d['format_score']:+.0f} exec={d['exec_score']:+.0f}"
                                              f" res={d['result_score']:+.0f} len={d['length_score']:+.2f}"
                                              f"  →  total={d['sql_reward']:+.2f}\n")
                                    _df.write(f"│    RET: cov={d['coverage_score']:+.2f}"
                                              f" irrel={d['irrelevant_penalty']:+.2f} useful={d['useful_noise_bonus']:+.2f}"
                                              f" len={d['ret_length_score']:+.2f}"
                                              f"  →  total={d['ret_reward']:+.2f}\n")
                                    _df.write(f"│    Case: {d['ret_case']}  |  Pred SQL: {str(d['pred_sql'])[:55]}...\n")

                                _df.write(f"└{'─'*88}┘\n\n")

                    # ========== Phase 7: Advantage Computation ==========
                    with marked_timer("adv", timing_raw):
                        if self.config.algorithm.use_kl_in_reward:
                            ret_batch, ret_kl = apply_kl_penalty(
                                ret_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            gen_batch, gen_kl = apply_kl_penalty(
                                gen_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics["retriever/reward_kl_penalty"] = ret_kl["actor/reward_kl_penalty"]
                            metrics["generator/reward_kl_penalty"] = gen_kl["actor/reward_kl_penalty"]
                        else:
                            ret_batch.batch["token_level_rewards"] = ret_batch.batch["token_level_scores"]
                            gen_batch.batch["token_level_rewards"] = gen_batch.batch["token_level_scores"]

                        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        ret_batch = compute_advantage(
                            ret_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=N_ret,
                            norm_adv_by_std_in_grpo=norm_adv,
                            config=self.config.algorithm,
                        )
                        gen_batch = compute_advantage(
                            gen_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=N_gen,
                            norm_adv_by_std_in_grpo=norm_adv,
                            config=self.config.algorithm,
                        )
                        self._add_grpo_diagnostics(ret_batch, metrics, "retriever")
                        self._add_grpo_diagnostics(gen_batch, metrics, "generator")

                    # Balance valid tokens across DP ranks before actor update.
                    # Joint update accumulates retriever and generator gradients
                    # on the same worker, so balance by the combined per-group
                    # load while keeping both passes in the same group order.
                    if self.config.trainer.balance_batch:
                        self._balance_joint_update_batches(ret_batch, gen_batch, metrics)

                    ret_batch.meta_info["global_token_num"] = torch.sum(
                        ret_batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    gen_batch.meta_info["global_token_num"] = torch.sum(
                        gen_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # ========== Phase 8: Retriever/Generator Loss Weight Scheduling ==========
                    progress = (self.global_steps - 1) / max(self.total_training_steps - 1, 1)
                    progress = min(max(progress, 0.0), 1.0)  # clamp to [0, 1]
                    alpha, beta, loss_weight_ramp_progress = self._loss_scales_for_progress(progress)

                    ret_batch.meta_info["loss_scale"] = alpha
                    gen_batch.meta_info["loss_scale"] = beta

                    grad_proj_cfg = self.config.actor_rollout_ref.actor.get("gradient_projection", {})
                    pcgrad_meta = {
                        "use_pcgrad": bool(grad_proj_cfg.get("enabled", False)),
                        "pcgrad_mode": grad_proj_cfg.get("mode", "symmetric"),
                        "normalize_task_grads": bool(grad_proj_cfg.get("normalize_task_grads", False)),
                        "pcgrad_main_task": grad_proj_cfg.get("main_task", "generator"),
                        "pcgrad_aux_task": grad_proj_cfg.get("aux_task", "retriever"),
                        "pcgrad_aux_weight": float(grad_proj_cfg.get("aux_weight", 1.0)),
                        "pcgrad_eps": float(grad_proj_cfg.get("eps", 1e-12)),
                        "pcgrad_main_grad_norm_ema_decay": float(
                            grad_proj_cfg.get("main_grad_norm_ema_decay", 0.95)
                        ),
                        "pcgrad_main_grad_norm_floor_min": float(
                            grad_proj_cfg.get("main_grad_norm_floor_min", 0.0)
                        ),
                        "pcgrad_pre_boost_generator": bool(grad_proj_cfg.get("pre_boost_generator", False)),
                        "pcgrad_pre_boost_target_ratio": float(grad_proj_cfg.get("pre_boost_target_ratio", 1.0)),
                        "pcgrad_pre_boost_max_scale": float(grad_proj_cfg.get("pre_boost_max_scale", 10.0)),
                        "pcgrad_max_ratio": float(grad_proj_cfg.get("max_ratio", 0.3)),
                    }
                    ret_batch.meta_info.update(pcgrad_meta)
                    gen_batch.meta_info.update(pcgrad_meta)
                    metrics["serl/alpha"] = alpha
                    metrics["serl/beta"] = beta
                    metrics["serl/retriever_loss_weight"] = alpha
                    metrics["serl/generator_loss_weight"] = beta
                    metrics["serl/loss_weight_ramp_progress"] = loss_weight_ramp_progress

                    # ========== Phase 9: Actor Update ==========
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        multi_turn = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        ret_batch.meta_info["multi_turn"] = multi_turn
                        gen_batch.meta_info["multi_turn"] = multi_turn
                        ret_update_batch = self._actor_update_view(ret_batch)
                        gen_update_batch = self._actor_update_view(gen_batch)
                        if self.dual_lora:
                            metrics["twopass/retriever_update_lora_int_id"] = 1.0
                            metrics["twopass/generator_update_lora_int_id"] = 2.0
                            with marked_timer("update_actor_dual_lora", timing_raw):
                                joint_output = self.actor_rollout_wg.update_actor_dual_lora(
                                    ret_update_batch,
                                    gen_update_batch,
                                )
                        else:
                            with marked_timer("update_actor_joint", timing_raw):
                                joint_output = self.actor_rollout_wg.update_actor_joint(
                                    ret_update_batch,
                                    gen_update_batch,
                                )
                        joint_metrics = reduce_metrics(joint_output.meta_info["metrics"])
                        metrics.update(joint_metrics)
                        del ret_update_batch, gen_update_batch

                    # Save checkpoint
                    esi_close = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close
                    ):
                        with marked_timer("save_ckpt", timing_raw):
                            self._save_checkpoint()

                    # Optional validation hook
                    if self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics = self._validate()
                        if val_metrics:
                            metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # Step-level bookkeeping
                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                # Data metrics for both retriever and generator batches
                for k, v in compute_data_metrics(batch=ret_batch, use_critic=False).items():
                    metrics[f"retriever/{k}"] = v
                for k, v in compute_data_metrics(batch=gen_batch, use_critic=False).items():
                    metrics[f"generator/{k}"] = v
                metrics.update(compute_timing_metrics(batch=gen_batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=gen_batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=gen_batch)

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=gen_batch)

                self._print_step_timing(self.global_steps, phase, timing_raw, metrics)

                # Free large DataProto objects to reduce inter-step memory residue
                del ret_batch, gen_batch, batch
                gc.collect()

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    ret_r=f"{metrics.get('reward/ret_mean', 0.0):.2f}",
                    gen_r=f"{metrics.get('reward/gen_mean', 0.0):.2f}",
                )

                self.global_steps += 1
                if is_last_step:
                    progress_bar.close()
                    return
