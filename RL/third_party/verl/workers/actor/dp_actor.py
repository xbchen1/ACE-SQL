# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import random
from contextlib import nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.utils.vllm_utils import lora_tensors_fingerprint, trace_lora_path
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_grad_tensors(params, grad_tensors):
    """Write projected gradients back to parameter .grad attributes."""
    for p, g in zip(params, grad_tensors):
        if g is None:
            p.grad = None
            continue
        if g.device != p.device:
            g = g.to(device=p.device, non_blocking=False)
        if p.grad is None:
            p.grad = g
        else:
            p.grad.copy_(g)


class LayerWisePCGradUtility:
    """Task-gradient projection utilities for joint training.

    Supports both:
      1. symmetric, layer-wise PCGrad
      2. generator-dominant, layer-wise one-sided projection
    """

    def __init__(self, model):
        self.named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named_params]
        self.main_task_grad_norm_ema = {}

    @staticmethod
    def _safe_cosine(dot_value: float, norm_a: float, norm_b: float, eps: float = 1e-12) -> float:
        denom = max(norm_a * norm_b, eps)
        return dot_value / denom

    def _apply_generator_dominant_projection(
        self,
        task_grad_buffers: dict,
        main_task: str,
        aux_task: str,
        aux_weight: float,
        eps: float,
        main_task_norm_ema_decay: float,
        main_task_norm_floor_min: float,
    ):
        if main_task not in task_grad_buffers or aux_task not in task_grad_buffers:
            raise ValueError(
                f"Generator-dominant PCGrad expects tasks {{{main_task}, {aux_task}}}, "
                f"got {list(task_grad_buffers.keys())}"
            )
        if main_task == aux_task:
            raise ValueError("Generator-dominant PCGrad requires different main_task and aux_task")

        main_grads = task_grad_buffers[main_task]
        aux_grads = task_grad_buffers[aux_task]

        total_conflict = 0
        total_pairs = len(self.params)
        total_floor_used = 0
        total_zero_main = 0
        active_aux_layers = 0
        cap_scale_sum = 0.0
        cap_scale_min = 1.0
        aux_coeff_sum = 0.0
        ema_sq_sum = 0.0
        floor_sq_sum = 0.0
        effective_cap_sq_sum = 0.0
        uncapped_aux_norm_sq_sum = 0.0
        raw_main_norm_sq_total = 0.0
        raw_aux_norm_sq_total = 0.0
        raw_dot_total = 0.0
        proj_aux_norm_sq_total = 0.0
        proj_aux_dot_main_total = 0.0
        final_norm_sq_total = 0.0
        final_dot_main_total = 0.0
        final_dot_aux_total = 0.0

        # Reuse the auxiliary-task gradient buffer as the final output buffer to
        # avoid keeping extra full-model copies alive during projection.
        final_grads = aux_grads
        aux_weight_value = float(aux_weight)

        for idx, (param_name, _) in enumerate(self.named_params):
            g_main = main_grads[idx]
            g_aux = aux_grads[idx]

            raw_main_norm_sq_value = torch.sum(g_main.float() * g_main.float()).item() if g_main is not None else 0.0
            raw_aux_norm_sq_value = torch.sum(g_aux.float() * g_aux.float()).item() if g_aux is not None else 0.0
            raw_dot_value = (
                torch.sum(g_main.float() * g_aux.float()).item()
                if g_main is not None and g_aux is not None
                else 0.0
            )
            raw_main_norm = raw_main_norm_sq_value ** 0.5
            raw_aux_norm = raw_aux_norm_sq_value ** 0.5

            raw_main_norm_sq_total += raw_main_norm_sq_value
            raw_aux_norm_sq_total += raw_aux_norm_sq_value
            raw_dot_total += raw_dot_value

            ema_key = (main_task, param_name)
            prev_main_ema = self.main_task_grad_norm_ema.get(ema_key, 0.0)
            if raw_main_norm > eps:
                if prev_main_ema > 0.0:
                    main_task_grad_ema = (
                        main_task_norm_ema_decay * prev_main_ema
                        + (1.0 - main_task_norm_ema_decay) * raw_main_norm
                    )
                else:
                    main_task_grad_ema = raw_main_norm
            else:
                main_task_grad_ema = prev_main_ema
            self.main_task_grad_norm_ema[ema_key] = main_task_grad_ema

            generator_grad_is_zero = raw_main_norm <= eps
            if generator_grad_is_zero:
                total_zero_main += 1

            generator_grad_floor = max(main_task_grad_ema, main_task_norm_floor_min)
            if generator_grad_is_zero and generator_grad_floor > 0.0:
                total_floor_used += 1

            effective_cap_norm = raw_main_norm if raw_main_norm > eps else generator_grad_floor
            ema_sq_sum += main_task_grad_ema * main_task_grad_ema
            floor_sq_sum += generator_grad_floor * generator_grad_floor
            effective_cap_sq_sum += effective_cap_norm * effective_cap_norm

            projection_applied = 0.0
            if g_aux is not None and raw_dot_value < 0.0 and raw_main_norm_sq_value > eps:
                proj_coeff = raw_dot_value / raw_main_norm_sq_value
                g_aux.add_(g_main, alpha=-proj_coeff)
                projection_applied = 1.0
                total_conflict += 1

            proj_aux_norm_sq_value = torch.sum(g_aux.float() * g_aux.float()).item() if g_aux is not None else 0.0
            proj_aux_dot_main = (
                torch.sum(g_aux.float() * g_main.float()).item()
                if g_aux is not None and g_main is not None
                else 0.0
            )
            proj_aux_norm = proj_aux_norm_sq_value ** 0.5
            uncapped_aux_norm = aux_weight_value * proj_aux_norm
            uncapped_aux_norm_sq_sum += uncapped_aux_norm * uncapped_aux_norm

            if g_aux is not None and proj_aux_norm > eps:
                active_aux_layers += 1
                if uncapped_aux_norm > eps:
                    retriever_cap_scale = min(1.0, effective_cap_norm / uncapped_aux_norm)
                else:
                    retriever_cap_scale = 1.0
                cap_scale_sum += retriever_cap_scale
                cap_scale_min = min(cap_scale_min, retriever_cap_scale)
            else:
                retriever_cap_scale = 1.0

            aux_coeff = aux_weight_value * retriever_cap_scale if g_aux is not None else 0.0
            aux_coeff_sum += aux_coeff

            if g_aux is None:
                out = g_main.detach().clone() if g_main is not None else None
            else:
                if aux_coeff == 0.0 and g_main is None:
                    out = None
                else:
                    if aux_coeff == 0.0:
                        g_aux.zero_()
                    else:
                        g_aux.mul_(aux_coeff)
                    if g_main is not None:
                        g_aux.add_(g_main)
                    out = g_aux

            final_grads[idx] = out
            main_grads[idx] = None

            proj_aux_norm_sq_total += proj_aux_norm_sq_value
            proj_aux_dot_main_total += proj_aux_dot_main
            final_dot_main_total += raw_main_norm_sq_value + aux_coeff * proj_aux_dot_main
            final_dot_aux_total += proj_aux_dot_main + aux_coeff * proj_aux_norm_sq_value
            final_norm_sq_total += max(
                raw_main_norm_sq_value
                + (aux_coeff * aux_coeff) * proj_aux_norm_sq_value
                + 2.0 * aux_coeff * proj_aux_dot_main,
                0.0,
            )

        raw_main_norm = raw_main_norm_sq_total ** 0.5
        raw_aux_norm = raw_aux_norm_sq_total ** 0.5
        proj_aux_norm = proj_aux_norm_sq_total ** 0.5
        final_norm = final_norm_sq_total ** 0.5
        mean_cap_scale = cap_scale_sum / max(active_aux_layers, 1)
        mean_aux_coeff = aux_coeff_sum / max(active_aux_layers, 1)

        metrics = {
            "pcgrad/conflict_ratio": total_conflict / max(total_pairs, 1),
            "pcgrad/conflict_count": float(total_conflict),
            "pcgrad/total_param_pairs": float(total_pairs),
            "pcgrad/normalize_task_grads": 0.0,
            "pcgrad/aux_projection_applied": total_conflict / max(total_pairs, 1),
            "pcgrad/aux_weight": aux_weight_value,
            "pcgrad/raw_generator_grad_norm": raw_main_norm,
            "pcgrad/raw_retriever_grad_norm": raw_aux_norm,
            "pcgrad/raw_grad_dot": raw_dot_total,
            "pcgrad/raw_grad_cosine": self._safe_cosine(raw_dot_total, raw_main_norm, raw_aux_norm, eps),
            "pcgrad/generator_grad_scale": 1.0,
            "pcgrad/retriever_grad_scale": mean_aux_coeff,
            "pcgrad/generator_grad_ema_norm": ema_sq_sum ** 0.5,
            "pcgrad/generator_grad_floor": floor_sq_sum ** 0.5,
            "pcgrad/effective_cap_norm": effective_cap_sq_sum ** 0.5,
            "pcgrad/retriever_uncapped_norm": uncapped_aux_norm_sq_sum ** 0.5,
            "pcgrad/retriever_cap_scale": mean_cap_scale,
            "pcgrad/retriever_cap_scale_min": cap_scale_min if active_aux_layers > 0 else 1.0,
            "pcgrad/floor_used": total_floor_used / max(total_pairs, 1),
            "pcgrad/generator_grad_is_zero": total_zero_main / max(total_pairs, 1),
            "pcgrad/projected_retriever_grad_norm": proj_aux_norm,
            "pcgrad/projected_dot_with_generator": proj_aux_dot_main_total,
            "pcgrad/projected_cosine_with_generator": self._safe_cosine(
                proj_aux_dot_main_total, raw_main_norm, proj_aux_norm, eps
            ),
            "pcgrad/final_grad_norm": final_norm,
            "pcgrad/final_cosine_with_generator": self._safe_cosine(
                final_dot_main_total, final_norm, raw_main_norm, eps
            ),
            "pcgrad/final_cosine_with_retriever": self._safe_cosine(
                final_dot_aux_total, final_norm, proj_aux_norm, eps
            ),
        }
        return final_grads, metrics

    def _apply_two_task_symmetric_projection(
        self,
        task_grad_buffers: dict,
        task_names: list[str],
        normalize_task_grads: bool,
        eps: float,
    ):
        task_a, task_b = task_names
        grads_a = task_grad_buffers[task_a]
        grads_b = task_grad_buffers[task_b]

        final_grads = []
        total_conflict = 0
        total_pairs = len(self.params)
        raw_a_norm_sq_total = 0.0
        raw_b_norm_sq_total = 0.0
        raw_dot_total = 0.0
        proj_a_norm_sq_total = 0.0
        proj_b_norm_sq_total = 0.0
        proj_dot_total = 0.0
        final_norm_sq_total = 0.0

        for idx, _ in enumerate(self.named_params):
            g_a = grads_a[idx]
            g_b = grads_b[idx]

            raw_a_norm_sq_value = torch.sum(g_a.float() * g_a.float()).item() if g_a is not None else 0.0
            raw_b_norm_sq_value = torch.sum(g_b.float() * g_b.float()).item() if g_b is not None else 0.0
            raw_dot_value = (
                torch.sum(g_a.float() * g_b.float()).item()
                if g_a is not None and g_b is not None
                else 0.0
            )

            raw_a_norm_sq_total += raw_a_norm_sq_value
            raw_b_norm_sq_total += raw_b_norm_sq_value
            raw_dot_total += raw_dot_value

            if raw_dot_value < 0.0:
                total_conflict += 1

            proj_a = g_a.detach().clone() if g_a is not None else None
            proj_b = g_b.detach().clone() if g_b is not None else None

            if raw_dot_value < 0.0:
                if proj_a is not None and raw_b_norm_sq_value > eps:
                    proj_a.add_(g_b, alpha=-(raw_dot_value / raw_b_norm_sq_value))
                if proj_b is not None and raw_a_norm_sq_value > eps:
                    proj_b.add_(g_a, alpha=-(raw_dot_value / raw_a_norm_sq_value))

            proj_a_norm_sq_value = torch.sum(proj_a.float() * proj_a.float()).item() if proj_a is not None else 0.0
            proj_b_norm_sq_value = torch.sum(proj_b.float() * proj_b.float()).item() if proj_b is not None else 0.0
            proj_dot_value = (
                torch.sum(proj_a.float() * proj_b.float()).item()
                if proj_a is not None and proj_b is not None
                else 0.0
            )

            proj_a_norm_sq_total += proj_a_norm_sq_value
            proj_b_norm_sq_total += proj_b_norm_sq_value
            proj_dot_total += proj_dot_value

            if proj_a is None:
                out = proj_b
            elif proj_b is None:
                out = proj_a
            else:
                proj_a.add_(proj_b)
                out = proj_a

            if out is not None:
                final_norm_sq_total += torch.sum(out.float() * out.float()).item()
            final_grads.append(out)
            grads_a[idx] = None
            grads_b[idx] = None

        raw_a_norm = raw_a_norm_sq_total ** 0.5
        raw_b_norm = raw_b_norm_sq_total ** 0.5
        proj_a_norm = proj_a_norm_sq_total ** 0.5
        proj_b_norm = proj_b_norm_sq_total ** 0.5
        final_norm = final_norm_sq_total ** 0.5
        metrics = {
            "pcgrad/conflict_ratio": total_conflict / max(total_pairs, 1),
            "pcgrad/conflict_count": float(total_conflict),
            "pcgrad/total_param_pairs": float(total_pairs),
            "pcgrad/normalize_task_grads": float(normalize_task_grads),
            f"pcgrad/raw_{task_a}_grad_norm": raw_a_norm,
            f"pcgrad/raw_{task_b}_grad_norm": raw_b_norm,
            "pcgrad/raw_grad_dot": raw_dot_total,
            "pcgrad/raw_grad_cosine": self._safe_cosine(raw_dot_total, raw_a_norm, raw_b_norm, eps),
            f"pcgrad/projected_{task_a}_grad_norm": proj_a_norm,
            f"pcgrad/projected_{task_b}_grad_norm": proj_b_norm,
            "pcgrad/projected_grad_dot": proj_dot_total,
            "pcgrad/projected_grad_cosine": self._safe_cosine(proj_dot_total, proj_a_norm, proj_b_norm, eps),
            "pcgrad/final_grad_norm": final_norm,
        }
        return final_grads, metrics

    def _apply_norm_equalized_projection(
        self,
        task_grad_buffers: dict,
        main_task: str,
        aux_task: str,
        max_ratio: float,
        eps: float,
    ):
        """Norm-equalized generator-dominant gradient management.

        Unlike PCGrad which projects away conflicting components (losing signal when
        conflict is pervasive), this mode:
          1. Globally scales the aux (retriever) gradient so its norm is at most
             ``max_ratio * ||main_grad||``.  This fixes the magnitude imbalance
             regardless of per-layer distribution.
          2. Per-layer: removes the component of the aux gradient that conflicts
             with the main (generator) gradient direction.
          3. Combines: ``final = main + aux_projected``.

        The main (generator) gradient is never modified, ensuring that the primary
        task's optimization direction is always preserved in full.
        """
        if main_task not in task_grad_buffers or aux_task not in task_grad_buffers:
            raise ValueError(
                f"norm_equalized expects tasks {{{main_task}, {aux_task}}}, "
                f"got {list(task_grad_buffers.keys())}"
            )

        main_grads = task_grad_buffers[main_task]
        aux_grads = task_grad_buffers[aux_task]

        # --- Step 1: compute global norms ---
        raw_main_norm_sq = 0.0
        raw_aux_norm_sq = 0.0
        for g_m, g_a in zip(main_grads, aux_grads):
            if g_m is not None:
                raw_main_norm_sq += torch.sum(g_m.float() * g_m.float()).item()
            if g_a is not None:
                raw_aux_norm_sq += torch.sum(g_a.float() * g_a.float()).item()
        raw_main_norm = raw_main_norm_sq ** 0.5
        raw_aux_norm = raw_aux_norm_sq ** 0.5

        # --- Step 2: global norm equalization (scale aux down, never up) ---
        target_aux_norm = raw_main_norm * max_ratio
        if raw_aux_norm > target_aux_norm and raw_main_norm > eps:
            aux_scale = target_aux_norm / raw_aux_norm
        else:
            aux_scale = 1.0

        if aux_scale < 1.0:
            for i, g_a in enumerate(aux_grads):
                if g_a is not None:
                    aux_grads[i] = g_a.mul_(aux_scale)

        # --- Step 3: per-layer conflict removal + combine ---
        total_conflict = 0
        total_layers = len(self.params)
        scaled_aux_norm_sq = 0.0
        proj_aux_norm_sq = 0.0
        raw_dot_total = 0.0
        proj_dot_total = 0.0
        final_norm_sq = 0.0

        final_grads = aux_grads

        for idx in range(len(self.named_params)):
            g_main = main_grads[idx]
            g_aux = aux_grads[idx]

            if g_aux is not None:
                scaled_aux_norm_sq += torch.sum(g_aux.float() * g_aux.float()).item()

            if g_main is not None and g_aux is not None:
                dot = torch.sum(g_main.float() * g_aux.float()).item()
                raw_dot_total += dot

                if dot < 0:
                    total_conflict += 1
                    main_norm_sq_layer = torch.sum(g_main.float() * g_main.float()).item()
                    if main_norm_sq_layer > eps:
                        g_aux.add_(g_main, alpha=-(dot / main_norm_sq_layer))

                proj_dot = torch.sum(g_main.float() * g_aux.float()).item()
                proj_dot_total += proj_dot
                proj_aux_norm_sq += torch.sum(g_aux.float() * g_aux.float()).item()

                g_aux.add_(g_main)
                final_grads[idx] = g_aux
            elif g_main is not None:
                final_grads[idx] = g_main
            # else: final_grads[idx] = g_aux (already set, possibly None)

            main_grads[idx] = None

            if final_grads[idx] is not None:
                final_norm_sq += torch.sum(final_grads[idx].float() * final_grads[idx].float()).item()

        scaled_aux_norm = scaled_aux_norm_sq ** 0.5
        proj_aux_norm = proj_aux_norm_sq ** 0.5
        final_norm = final_norm_sq ** 0.5

        metrics = {
            "pcgrad/conflict_ratio": total_conflict / max(total_layers, 1),
            "pcgrad/conflict_count": float(total_conflict),
            "pcgrad/total_param_pairs": float(total_layers),
            "pcgrad/raw_generator_grad_norm": raw_main_norm,
            "pcgrad/raw_retriever_grad_norm": raw_aux_norm,
            "pcgrad/max_ratio": float(max_ratio),
            "pcgrad/aux_global_scale": aux_scale,
            "pcgrad/scaled_retriever_grad_norm": scaled_aux_norm,
            "pcgrad/raw_grad_dot": raw_dot_total,
            "pcgrad/raw_grad_cosine": self._safe_cosine(raw_dot_total, raw_main_norm, raw_aux_norm, eps),
            "pcgrad/projected_retriever_grad_norm": proj_aux_norm,
            "pcgrad/projected_dot_with_generator": proj_dot_total,
            "pcgrad/projected_cosine_with_generator": self._safe_cosine(
                proj_dot_total, raw_main_norm, proj_aux_norm, eps
            ),
            "pcgrad/final_grad_norm": final_norm,
            "pcgrad/final_cosine_with_generator": self._safe_cosine(
                raw_main_norm_sq + proj_dot_total, final_norm, raw_main_norm, eps
            ),
        }
        return final_grads, metrics

    def apply_projection(
        self,
        task_grad_buffers: dict,
        mode: str = "symmetric",
        normalize_task_grads: bool = False,
        main_task: str = "generator",
        aux_task: str = "retriever",
        aux_weight: float = 1.0,
        eps: float = 1e-12,
        main_task_norm_ema_decay: float = 0.95,
        main_task_norm_floor_min: float = 0.0,
        max_ratio: float = 0.3,
    ):
        """Apply task-gradient projection.

        Args:
            task_grad_buffers: Dict[task_name, List[Tensor]] — accumulated grads per task.
        Returns:
            (final_grads, metrics): projected gradient list and conflict stats.
        """
        task_names = list(task_grad_buffers.keys())
        num_tasks = len(task_names)

        if num_tasks <= 1:
            if num_tasks == 0:
                return [None] * len(self.params), {}
            only_task = task_names[0]
            return list(task_grad_buffers[only_task]), {}

        if mode == "norm_equalized":
            return self._apply_norm_equalized_projection(
                task_grad_buffers=task_grad_buffers,
                main_task=main_task,
                aux_task=aux_task,
                max_ratio=max_ratio,
                eps=eps,
            )

        if mode == "generator_dominant":
            return self._apply_generator_dominant_projection(
                task_grad_buffers=task_grad_buffers,
                main_task=main_task,
                aux_task=aux_task,
                aux_weight=aux_weight,
                eps=eps,
                main_task_norm_ema_decay=main_task_norm_ema_decay,
                main_task_norm_floor_min=main_task_norm_floor_min,
            )
        if mode != "symmetric":
            raise ValueError(f"Unsupported PCGrad mode: {mode}")

        if num_tasks == 2:
            return self._apply_two_task_symmetric_projection(
                task_grad_buffers=task_grad_buffers,
                task_names=task_names,
                normalize_task_grads=normalize_task_grads,
                eps=eps,
            )

        # Keep the randomized sequential PCGrad fallback for 3+ tasks.
        final_grads = []
        metrics = {}
        total_conflict = 0
        total_pairs = 0

        for p_idx, _ in enumerate(self.named_params):
            original_grads = {}
            for t in task_names:
                g = task_grad_buffers[t][p_idx]
                original_grads[t] = g

            sorted_names = sorted(task_names)
            for i in range(len(sorted_names)):
                for j in range(i + 1, len(sorted_names)):
                    total_pairs += 1
                    g_i = original_grads[sorted_names[i]]
                    g_j = original_grads[sorted_names[j]]
                    if g_i is None or g_j is None:
                        continue
                    dot = torch.sum(g_i * g_j)
                    if dot.item() < 0:
                        total_conflict += 1

            projected = {t: (original_grads[t].clone() if original_grads[t] is not None else None) for t in task_names}
            task_indices = torch.randperm(num_tasks).tolist()

            for i in task_indices:
                task_i = task_names[i]
                g_i = projected[task_i]
                if g_i is None:
                    continue
                others = [x for x in task_indices if x != i]
                random.shuffle(others)
                for j in others:
                    task_j = task_names[j]
                    g_j = original_grads[task_j]
                    if g_j is None:
                        continue
                    dot = torch.sum(g_i * g_j)
                    if dot < 0:
                        norm_sq = torch.sum(g_j * g_j)
                        if norm_sq > 1e-12:
                            g_i.sub_((dot / norm_sq) * g_j)

            g_sum = None
            for t in task_names:
                g_t = projected[t]
                if g_t is None:
                    continue
                if g_sum is None:
                    g_sum = g_t
                else:
                    g_sum.add_(g_t)
            final_grads.append(g_sum)

        metrics["pcgrad/conflict_ratio"] = total_conflict / max(total_pairs, 1)
        metrics["pcgrad/conflict_count"] = total_conflict
        metrics["pcgrad/total_param_pairs"] = total_pairs
        return final_grads, metrics


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # Initialize PCGrad utility for joint training (Actor mode only)
        if self.actor_optimizer is not None:
            self.pcgrad_util = LayerWisePCGradUtility(self.actor_module)
        self._optimizer_diag_printed = set()
        self._dual_lora_optimizer_isolation_checked = False

    def _get_compute_device(self):
        try:
            return next(self.actor_module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _require_dual_lora_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        required = {"retriever", "generator"}
        if not isinstance(self.actor_optimizer, dict):
            raise RuntimeError(
                "Dual-LoRA update requires isolated optimizers for retriever and generator; "
                f"got {type(self.actor_optimizer).__name__}."
            )
        optimizer_keys = set(self.actor_optimizer.keys())
        if optimizer_keys != required:
            raise RuntimeError(
                "Dual-LoRA optimizer keys must be exactly "
                f"{sorted(required)}, got {sorted(optimizer_keys)}."
            )
        if not self._dual_lora_optimizer_isolation_checked:
            param_ids_by_adapter = {}
            for adapter_name, optimizer in self.actor_optimizer.items():
                param_ids = set()
                param_count = 0
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        param_ids.add(id(param))
                        param_count += int(param.numel())
                        if param.dtype != torch.float32:
                            raise RuntimeError(
                                "Dual-LoRA optimizer parameters must stay fp32 so AdamW state stays fp32; "
                                f"adapter={adapter_name}, dtype={param.dtype}."
                            )
                if param_count <= 0:
                    raise RuntimeError(f"Dual-LoRA optimizer for {adapter_name} has no parameters.")
                param_ids_by_adapter[adapter_name] = param_ids
            overlap = param_ids_by_adapter["retriever"] & param_ids_by_adapter["generator"]
            if overlap:
                raise RuntimeError(
                    "Dual-LoRA optimizer isolation failed: retriever and generator optimizers "
                    f"share {len(overlap)} parameter object(s)."
                )
            self._dual_lora_optimizer_isolation_checked = True
        return self.actor_optimizer

    def _dual_lora_use_no_sync(self) -> bool:
        return bool(
            self.config.get("dual_lora_use_no_sync", True)
            and hasattr(self.actor_module, "no_sync")
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _maybe_release_dual_lora_cache(self):
        if not self.config.get("dual_lora_empty_cache_between_passes", False):
            return
        import gc
        from verl.utils.device import get_torch_device

        gc.collect()
        get_torch_device().empty_cache()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self, optimizer: torch.optim.Optimizer | None = None):
        assert self.config.grad_clip is not None
        optimizer = optimizer if optimizer is not None else self.actor_optimizer
        if isinstance(optimizer, dict):
            raise RuntimeError(
                "A dict of Dual-LoRA optimizers cannot be stepped by the generic optimizer path. "
                "Use update_policy_dual_lora or update_policy_single_lora instead."
            )

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
        return grad_norm

    def _trace_dual_lora_actor_params(self, event: str, adapter_name: str, global_steps=None, ppo_epoch=None):
        if os.getenv("ACE_SQL_LORA_TRACE", "0").lower() not in {"1", "true", "yes", "on"}:
            return

        peft_model = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        adapter_tensors = {}
        adapter_param_numel = 0
        for name, param in peft_model.named_parameters():
            if "lora_" not in name or adapter_name not in name:
                continue
            adapter_tensors[name] = param.detach()
            adapter_param_numel += int(param.numel())

        trace_lora_path(
            "[DataParallelPPOActor.dual_lora_params] "
            f"event={event} adapter_name={adapter_name} global_steps={global_steps} "
            f"ppo_epoch={ppo_epoch} params={len(adapter_tensors)} numel={adapter_param_numel} "
            f"tensor_fp={lora_tensors_fingerprint(adapter_tensors)}"
        )

    def _collect_optimizer_diagnostics(self, optimizer: torch.optim.Optimizer, prefix: str) -> dict:
        """Collect first-step optimizer dtype diagnostics for Dual LoRA debugging."""

        def _numel_by_dtype(tensors):
            counts = {}
            total = 0
            for tensor in tensors:
                if tensor is None:
                    continue
                numel = tensor.numel()
                key = str(tensor.dtype).replace("torch.", "")
                counts[key] = counts.get(key, 0) + numel
                total += numel
            return counts, total

        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        param_norm_sq = 0.0
        grad_norm_sq = 0.0
        adam_update_norm_sq = 0.0
        adam_update_numel = 0
        adam_eps = None
        for group in optimizer.param_groups:
            group_lr = group.get("lr", 0.0)
            group_eps = group.get("eps", 1e-8)
            if adam_eps is None:
                adam_eps = group_eps
            for param in group["params"]:
                params.append(param)
                grads.append(param.grad)
                param_norm_sq += torch.sum(param.detach().float() * param.detach().float()).item()
                if param.grad is not None:
                    grad = param.grad.detach().float()
                    grad_norm_sq += torch.sum(grad * grad).item()
                state = optimizer.state.get(param, {})
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)
                if isinstance(exp_avg, torch.Tensor):
                    exp_avgs.append(exp_avg)
                if isinstance(exp_avg_sq, torch.Tensor):
                    exp_avg_sqs.append(exp_avg_sq)
                step = state.get("step", None)
                if isinstance(exp_avg, torch.Tensor) and isinstance(exp_avg_sq, torch.Tensor) and step is not None:
                    step_value = step.item() if isinstance(step, torch.Tensor) else step
                    if step_value:
                        beta1, beta2 = group.get("betas", (0.9, 0.999))
                        bias_correction1 = 1 - beta1**step_value
                        bias_correction2 = 1 - beta2**step_value
                        denom = exp_avg_sq.detach().float().sqrt() / (bias_correction2**0.5)
                        denom = denom.add(group_eps)
                        update = (exp_avg.detach().float() / bias_correction1).div(denom).mul(group_lr)
                        adam_update_norm_sq += torch.sum(update * update).item()
                        adam_update_numel += update.numel()

        param_counts, param_total = _numel_by_dtype(params)
        grad_counts, grad_total = _numel_by_dtype(grads)
        exp_avg_counts, exp_avg_total = _numel_by_dtype(exp_avgs)
        exp_avg_sq_counts, exp_avg_sq_total = _numel_by_dtype(exp_avg_sqs)
        param_norm = param_norm_sq**0.5
        grad_norm = grad_norm_sq**0.5
        adam_update_norm = adam_update_norm_sq**0.5
        adam_update_rms = (adam_update_norm_sq / adam_update_numel) ** 0.5 if adam_update_numel else 0.0

        def _frac(counts, total, dtype_name):
            return float(counts.get(dtype_name, 0) / total) if total else 0.0

        metrics = {
            f"{prefix}/diag/param_numel": float(param_total),
            f"{prefix}/diag/param_fp32_frac": _frac(param_counts, param_total, "float32"),
            f"{prefix}/diag/param_bf16_frac": _frac(param_counts, param_total, "bfloat16"),
            f"{prefix}/diag/grad_numel": float(grad_total),
            f"{prefix}/diag/grad_fp32_frac": _frac(grad_counts, grad_total, "float32"),
            f"{prefix}/diag/grad_bf16_frac": _frac(grad_counts, grad_total, "bfloat16"),
            f"{prefix}/diag/exp_avg_numel": float(exp_avg_total),
            f"{prefix}/diag/exp_avg_fp32_frac": _frac(exp_avg_counts, exp_avg_total, "float32"),
            f"{prefix}/diag/exp_avg_bf16_frac": _frac(exp_avg_counts, exp_avg_total, "bfloat16"),
            f"{prefix}/diag/exp_avg_sq_numel": float(exp_avg_sq_total),
            f"{prefix}/diag/exp_avg_sq_fp32_frac": _frac(exp_avg_sq_counts, exp_avg_sq_total, "float32"),
            f"{prefix}/diag/exp_avg_sq_bf16_frac": _frac(exp_avg_sq_counts, exp_avg_sq_total, "bfloat16"),
            f"{prefix}/diag/optimizer_state_initialized": float(exp_avg_total > 0 and exp_avg_sq_total > 0),
            f"{prefix}/diag/adam_eps": float(adam_eps if adam_eps is not None else 0.0),
            f"{prefix}/diag/param_norm": float(param_norm),
            f"{prefix}/diag/grad_norm_from_params": float(grad_norm),
            f"{prefix}/diag/adam_update_norm_est": float(adam_update_norm),
            f"{prefix}/diag/adam_update_rms_est": float(adam_update_rms),
            f"{prefix}/diag/adam_update_to_param_norm_est": float(adam_update_norm / param_norm) if param_norm else 0.0,
        }

        if prefix not in self._optimizer_diag_printed and torch.distributed.get_rank() == 0:
            print(
                f"[DualLoRA][diagnostic][after_first_step][{prefix}] "
                f"param_dtypes={param_counts} grad_dtypes={grad_counts} "
                f"exp_avg_dtypes={exp_avg_counts} exp_avg_sq_dtypes={exp_avg_sq_counts} "
                f"adam_eps={adam_eps} adam_update_norm_est={adam_update_norm:.6g} "
                f"adam_update_rms_est={adam_update_rms:.6g}",
                flush=True,
            )
            self._optimizer_diag_printed.add(prefix)

        return metrics

    def _apply_pcgrad_and_step(
        self,
        task_grad_buffers,
        pcgrad_mode: str = "symmetric",
        normalize_task_grads: bool = False,
        pcgrad_main_task: str = "generator",
        pcgrad_aux_task: str = "retriever",
        pcgrad_aux_weight: float = 1.0,
        pcgrad_eps: float = 1e-12,
        pcgrad_main_grad_norm_ema_decay: float = 0.95,
        pcgrad_main_grad_norm_floor_min: float = 0.0,
        pcgrad_max_ratio: float = 0.3,
    ):
        """Apply task-gradient projection if multiple tasks, then optimizer step.

        Args:
            task_grad_buffers: Dict[task_name, List[Tensor]]
        Returns:
            (grad_norm, metrics)
        """
        if len(task_grad_buffers) <= 1:
            for task_grads in task_grad_buffers.values():
                _set_grad_tensors(self.pcgrad_util.params, task_grads)
            return self._optimizer_step(), {}

        final_grad_tensors, pcgrad_metrics = self.pcgrad_util.apply_projection(
            task_grad_buffers,
            mode=pcgrad_mode,
            normalize_task_grads=normalize_task_grads,
            main_task=pcgrad_main_task,
            aux_task=pcgrad_aux_task,
            aux_weight=pcgrad_aux_weight,
            eps=pcgrad_eps,
            main_task_norm_ema_decay=pcgrad_main_grad_norm_ema_decay,
            main_task_norm_floor_min=pcgrad_main_grad_norm_floor_min,
            max_ratio=pcgrad_max_ratio,
        )
        _set_grad_tensors(self.pcgrad_util.params, final_grad_tensors)
        for task_name, task_grads in task_grad_buffers.items():
            for idx in range(len(task_grads)):
                task_grads[idx] = None
        del final_grad_tensors
        grad_norm = self._optimizer_step()
        return grad_norm, pcgrad_metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        if isinstance(self.actor_optimizer, dict):
            raise RuntimeError(
                "Generic update_policy was called with Dual-LoRA optimizers. "
                "Use update_policy_dual_lora or update_policy_single_lora so the target adapter is explicit."
            )
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        ppo_mini_batch_size = int(self.config.ppo_mini_batch_size)
        effective_repeat_n = data.meta_info.get("ppo_effective_repeat_n", None)
        base_repeat_n = data.meta_info.get("ppo_base_repeat_n", None)
        if effective_repeat_n is not None or base_repeat_n is not None:
            if effective_repeat_n is None or base_repeat_n is None:
                raise RuntimeError(
                    "single-pass PPO repeat scaling requires both ppo_effective_repeat_n "
                    "and ppo_base_repeat_n in DataProto.meta_info."
                )
            effective_repeat_n = int(effective_repeat_n)
            base_repeat_n = int(base_repeat_n)
            if effective_repeat_n <= 0 or base_repeat_n <= 0:
                raise RuntimeError(
                    "single-pass PPO repeat scaling expects positive repeat counts, "
                    f"got effective={effective_repeat_n}, base={base_repeat_n}."
                )
            scaled_mini = ppo_mini_batch_size * effective_repeat_n
            if scaled_mini % base_repeat_n != 0:
                raise RuntimeError(
                    "single-pass PPO mini-batch scaling is not integral: "
                    f"base_mini={ppo_mini_batch_size}, effective_repeat_n={effective_repeat_n}, "
                    f"base_repeat_n={base_repeat_n}. Adjust ppo_mini_batch_size or repeat counts."
                )
            ppo_mini_batch_size = scaled_mini // base_repeat_n
        if ppo_mini_batch_size <= 0:
            raise RuntimeError(f"ppo_mini_batch_size must be positive, got {ppo_mini_batch_size}.")
        mini_batches = data.split(ppo_mini_batch_size)

        metrics = {
            "actor/ppo_mini_batch_size": [float(ppo_mini_batch_size)],
            "actor/ppo_mini_batches_per_epoch": [float(len(mini_batches))],
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad(set_to_none=True)

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics

    def _compute_single_pass_grads(
        self, data: DataProto, temperature: float, scale: float, prefix: str, sync_state: dict | None = None
    ):
        """Compute and accumulate gradients for one pass (retriever or generator).

        This is a helper for update_policy_joint(). It runs the same forward/loss logic
        as update_policy() but scales the loss by `scale` before backward, and does NOT
        call optimizer.step() — the caller is responsible for that.

        Returns per-micro-batch metrics keyed with the given prefix.
        """
        select_keys = [
            "responses", "response_mask", "input_ids",
            "attention_mask", "position_ids", "old_log_probs", "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        # update_policy_joint/PCGrad need full-task gradients and therefore
        # step once after this helper returns. Average over PPO mini-batches so
        # this path does not scale gradients by len(mini_batches).
        num_mini_batches = max(len(mini_batches), 1)
        metrics = {}
        compute_device = self._get_compute_device()

        for mini_batch in mini_batches:
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(compute_device)
                micro_batch_metrics = {}
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                response_mask = model_inputs["response_mask"]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = (
                    self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                )
                clip_ratio_high = (
                    self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                )
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                calculate_entropy = entropy_coeff != 0
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )

                if self.config.policy_loss.loss_mode == "vanilla":
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )
                else:
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                if entropy_coeff != 0:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    kld = kl_penalty(
                        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                    )
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    micro_batch_metrics[f"{prefix}/kl_loss"] = kl_loss.detach().item()

                # Scale loss by alpha/beta and gradient accumulation
                if self.config.use_dynamic_bsz:
                    loss = (
                        policy_loss
                        * scale
                        * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                        / num_mini_batches
                    )
                else:
                    loss = policy_loss * scale / gradient_accumulation / num_mini_batches
                if sync_state is None:
                    loss.backward()
                else:
                    sync_state["idx"] += 1
                    is_last_backward = sync_state["idx"] >= sync_state["total"]
                    sync_ctx = nullcontext() if is_last_backward else self.actor_module.no_sync()
                    with sync_ctx:
                        loss.backward()

                micro_batch_metrics.update({
                    f"{prefix}/pg_loss": pg_loss.detach().item(),
                    f"{prefix}/pg_clipfrac": pg_clipfrac.detach().item(),
                    f"{prefix}/ppo_kl": ppo_kl.detach().item(),
                    f"{prefix}/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                })
                append_to_dict(metrics, micro_batch_metrics)

                del model_inputs, response_mask, old_log_prob, advantages, entropy, log_prob
                del pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, policy_loss, loss
                if self.config.use_kl_loss:
                    del ref_log_prob, kld, kl_loss
                if entropy_coeff != 0:
                    del entropy_loss

            if micro_batches:
                del micro_batch
            del micro_batches
            mini_batch = mini_batch.to("cpu")
            del mini_batch
            if self.config.get("empty_cache_per_mini_batch", False):
                torch.cuda.empty_cache()

        return metrics

    def _update_single_pass_policy(
        self,
        data: DataProto,
        temperature: float,
        scale: float,
        prefix: str,
        optimizer: torch.optim.Optimizer,
    ):
        """Update one dual-LoRA adapter with the same mini-batch semantics as VERL update_policy()."""
        select_keys = [
            "responses", "response_mask", "input_ids",
            "attention_mask", "position_ids", "old_log_probs", "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        ppo_mini_batch_size = int(self.config.ppo_mini_batch_size)
        effective_repeat_n = data.meta_info.get("ppo_effective_repeat_n", None)
        base_repeat_n = data.meta_info.get("ppo_base_repeat_n", None)
        if effective_repeat_n is not None or base_repeat_n is not None:
            if effective_repeat_n is None or base_repeat_n is None:
                raise RuntimeError(
                    "single-pass PPO repeat scaling requires both ppo_effective_repeat_n "
                    "and ppo_base_repeat_n in DataProto.meta_info."
                )
            effective_repeat_n = int(effective_repeat_n)
            base_repeat_n = int(base_repeat_n)
            if effective_repeat_n <= 0 or base_repeat_n <= 0:
                raise RuntimeError(
                    "single-pass PPO repeat scaling expects positive repeat counts, "
                    f"got effective={effective_repeat_n}, base={base_repeat_n}."
                )
            scaled_mini = ppo_mini_batch_size * effective_repeat_n
            if scaled_mini % base_repeat_n != 0:
                raise RuntimeError(
                    "single-pass PPO mini-batch scaling is not integral: "
                    f"base_mini={ppo_mini_batch_size}, effective_repeat_n={effective_repeat_n}, "
                    f"base_repeat_n={base_repeat_n}. Adjust ppo_mini_batch_size or repeat counts."
                )
            ppo_mini_batch_size = scaled_mini // base_repeat_n
        if ppo_mini_batch_size <= 0:
            raise RuntimeError(f"ppo_mini_batch_size must be positive, got {ppo_mini_batch_size}.")

        mini_batches = data.split(ppo_mini_batch_size)
        metrics = {}
        append_to_dict(
            metrics,
            {
                f"{prefix}/ppo_mini_batch_size": float(ppo_mini_batch_size),
                f"{prefix}/ppo_mini_batches_per_epoch": float(len(mini_batches)),
            },
        )
        compute_device = self._get_compute_device()

        for mini_batch in mini_batches:
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                if ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu != 0:
                    raise RuntimeError(
                        "ppo_mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu "
                        "when dynamic batching is disabled, got "
                        f"{ppo_mini_batch_size} and {self.config.ppo_micro_batch_size_per_gpu}."
                    )
                gradient_accumulation = (
                    ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            optimizer.zero_grad(set_to_none=True)
            can_no_sync = self._dual_lora_use_no_sync() and len(micro_batches) > 1
            append_to_dict(
                metrics,
                {
                    f"{prefix}/dual_lora_no_sync_enabled": float(can_no_sync),
                    f"{prefix}/dual_lora_micro_batches_per_mini_batch": float(len(micro_batches)),
                },
            )

            for micro_idx, micro_batch in enumerate(micro_batches):
                micro_batch = micro_batch.to(compute_device)
                micro_batch_metrics = {}
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                response_mask = model_inputs["response_mask"]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = (
                    self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                )
                clip_ratio_high = (
                    self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                )
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                calculate_entropy = entropy_coeff != 0
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                if self.config.policy_loss.loss_mode == "vanilla":
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )
                else:
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                if entropy_coeff != 0:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    kld = kl_penalty(
                        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                    )
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    micro_batch_metrics[f"{prefix}/kl_loss"] = kl_loss.detach().item()
                    micro_batch_metrics[f"{prefix}/kl_coef"] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    loss = policy_loss * scale * (response_mask.shape[0] / ppo_mini_batch_size)
                else:
                    loss = policy_loss * scale / gradient_accumulation
                sync_ctx = nullcontext()
                if can_no_sync and micro_idx < len(micro_batches) - 1:
                    sync_ctx = self.actor_module.no_sync()
                with sync_ctx:
                    loss.backward()

                micro_batch_metrics.update({
                    f"{prefix}/pg_loss": pg_loss.detach().item(),
                    f"{prefix}/pg_clipfrac": pg_clipfrac.detach().item(),
                    f"{prefix}/ppo_kl": ppo_kl.detach().item(),
                    f"{prefix}/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                })
                append_to_dict(metrics, micro_batch_metrics)

                del model_inputs, response_mask, old_log_prob, advantages, entropy, log_prob
                del pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, policy_loss, loss
                if self.config.use_kl_loss:
                    del ref_log_prob, kld, kl_loss
                if entropy_coeff != 0:
                    del entropy_loss

            grad_norm = self._optimizer_step(optimizer)
            append_to_dict(metrics, {f"{prefix}/grad_norm": grad_norm.detach().item()})
            append_to_dict(metrics, self._collect_optimizer_diagnostics(optimizer, prefix))

            if micro_batches:
                del micro_batch
            del micro_batches
            mini_batch = mini_batch.to("cpu")
            del mini_batch
            if self.config.get("empty_cache_per_mini_batch", False):
                torch.cuda.empty_cache()

        optimizer.zero_grad(set_to_none=True)
        return metrics

    def _count_single_pass_micro_batches(self, data: DataProto) -> int:
        """Count how many backward calls one pass will trigger."""
        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        count = 0
        for mini_batch in mini_batches:
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
            count += len(micro_batches)
        return count

    @GPUMemoryLogger(role="dp actor joint", logger=logger)
    def update_policy_joint(
        self,
        data_p1: DataProto,
        data_p2: DataProto,
        alpha: float,
        beta: float,
        use_pcgrad: bool = False,
        pcgrad_mode: str = "symmetric",
        normalize_task_grads: bool = False,
        pcgrad_main_task: str = "generator",
        pcgrad_aux_task: str = "retriever",
        pcgrad_aux_weight: float = 1.0,
        pcgrad_eps: float = 1e-12,
        pcgrad_main_grad_norm_ema_decay: float = 0.95,
        pcgrad_main_grad_norm_floor_min: float = 0.0,
        pcgrad_pre_boost_generator: bool = False,
        pcgrad_pre_boost_target_ratio: float = 1.0,
        pcgrad_pre_boost_max_scale: float = 10.0,
        pcgrad_max_ratio: float = 0.3,
    ):
        """Joint policy update: accumulate gradients from two passes then step once.

        Args:
            data_p1: Retriever pass data (with old_log_probs, advantages, etc.)
            data_p2: Generator pass data
            alpha: Loss scaling for retriever pass
            beta: Loss scaling for generator pass
            use_pcgrad: If True, apply task-gradient projection to resolve
                gradient conflicts between retriever and generator passes.
            pcgrad_max_ratio: For norm_equalized mode, the maximum ratio of
                aux (retriever) gradient norm to main (generator) gradient norm.
        """
        if isinstance(self.actor_optimizer, dict):
            raise RuntimeError(
                "Shared update_policy_joint was called with Dual-LoRA optimizers. "
                "Use update_policy_dual_lora so retriever and generator adapters are updated independently."
            )
        self.actor_module.train()
        ret_temperature = float(data_p1.meta_info["temperature"])
        gen_temperature = float(data_p2.meta_info["temperature"])

        can_no_sync = (
            hasattr(self.actor_module, "no_sync")
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        p1_sync_total = 0
        p2_sync_total = 0
        sync_total = 0
        if can_no_sync:
            p1_sync_total = self._count_single_pass_micro_batches(data_p1)
            p2_sync_total = self._count_single_pass_micro_batches(data_p2)
            sync_total = p1_sync_total + p2_sync_total
            if sync_total <= 1:
                can_no_sync = False

        is_norm_equalized = pcgrad_mode == "norm_equalized"

        metrics = {
            "retriever/update_temperature": ret_temperature,
            "generator/update_temperature": gen_temperature,
        }
        for _ in range(self.config.ppo_epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)

            if use_pcgrad:
                sync_state_p1 = {"idx": 0, "total": p1_sync_total} if can_no_sync and p1_sync_total > 1 else None
                sync_state_p2 = {"idx": 0, "total": p2_sync_total} if can_no_sync and p2_sync_total > 1 else None

                # norm_equalized: use scale=1.0 so both tasks produce full-strength
                # gradients; the norm equalization step handles the balance.
                # Other modes: use alpha/beta from the SERL schedule.
                ret_scale = 1.0 if is_norm_equalized else alpha
                gen_scale = 1.0 if is_norm_equalized else beta

                # === Store grads per task, project, then step ===
                # Pass 1 (retriever): backward and capture gradients
                p1_metrics = self._compute_single_pass_grads(
                    data_p1, ret_temperature, ret_scale, prefix="retriever", sync_state=sync_state_p1
                )
                append_to_dict(metrics, p1_metrics)

                ret_grads = []
                for p in self.pcgrad_util.params:
                    ret_grads.append(p.grad.detach().clone() if p.grad is not None else None)
                self.actor_optimizer.zero_grad(set_to_none=True)

                # Pass 2 (generator): backward and capture gradients
                p2_metrics = self._compute_single_pass_grads(
                    data_p2, gen_temperature, gen_scale, prefix="generator", sync_state=sync_state_p2
                )
                append_to_dict(metrics, p2_metrics)

                gen_grads = []
                for p in self.pcgrad_util.params:
                    gen_grads.append(p.grad.detach().clone() if p.grad is not None else None)
                self.actor_optimizer.zero_grad(set_to_none=True)

                # Pre-boost is only for legacy PCGrad modes; norm_equalized
                # handles magnitude balance via global scaling instead.
                pre_boost_metrics = {}
                if pcgrad_pre_boost_generator and not is_norm_equalized:
                    raw_ret_norm_sq = 0.0
                    raw_gen_norm_sq = 0.0
                    for ret_g, gen_g in zip(ret_grads, gen_grads):
                        if ret_g is not None:
                            raw_ret_norm_sq += torch.sum(ret_g.float() * ret_g.float()).item()
                        if gen_g is not None:
                            raw_gen_norm_sq += torch.sum(gen_g.float() * gen_g.float()).item()
                    raw_ret_norm = raw_ret_norm_sq ** 0.5
                    raw_gen_norm = raw_gen_norm_sq ** 0.5
                    target_gen_norm = pcgrad_pre_boost_target_ratio * raw_ret_norm
                    if raw_gen_norm > pcgrad_eps and raw_gen_norm < target_gen_norm:
                        generator_scale = min(pcgrad_pre_boost_max_scale, target_gen_norm / raw_gen_norm)
                    else:
                        generator_scale = 1.0
                    if generator_scale > 1.0:
                        for idx, gen_g in enumerate(gen_grads):
                            if gen_g is not None:
                                gen_grads[idx] = gen_g.mul(generator_scale)
                    boosted_gen_norm = raw_gen_norm * generator_scale
                    pre_boost_metrics = {
                        "pcgrad/pre_boost_enabled": 1.0,
                        "pcgrad/pre_boost_target_ratio": float(pcgrad_pre_boost_target_ratio),
                        "pcgrad/pre_boost_max_scale": float(pcgrad_pre_boost_max_scale),
                        "pcgrad/pre_boost_raw_retriever_grad_norm": raw_ret_norm,
                        "pcgrad/pre_boost_raw_generator_grad_norm": raw_gen_norm,
                        "pcgrad/pre_boost_target_generator_grad_norm": target_gen_norm,
                        "pcgrad/pre_boost_generator_scale": float(generator_scale),
                        "pcgrad/pre_boost_boosted_generator_grad_norm": boosted_gen_norm,
                    }

                # Apply gradient projection and optimizer step
                task_grad_buffer = {"retriever": ret_grads, "generator": gen_grads}
                grad_norm, pcgrad_metrics = self._apply_pcgrad_and_step(
                    task_grad_buffer,
                    pcgrad_mode=pcgrad_mode,
                    normalize_task_grads=normalize_task_grads,
                    pcgrad_main_task=pcgrad_main_task,
                    pcgrad_aux_task=pcgrad_aux_task,
                    pcgrad_aux_weight=pcgrad_aux_weight,
                    pcgrad_eps=pcgrad_eps,
                    pcgrad_main_grad_norm_ema_decay=pcgrad_main_grad_norm_ema_decay,
                    pcgrad_main_grad_norm_floor_min=pcgrad_main_grad_norm_floor_min,
                    pcgrad_max_ratio=pcgrad_max_ratio,
                )
                append_to_dict(metrics, pre_boost_metrics)
                append_to_dict(metrics, pcgrad_metrics)
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

                del task_grad_buffer, ret_grads, gen_grads
            else:
                # === Original mode: accumulate gradients and step ===
                sync_state = {"idx": 0, "total": sync_total} if can_no_sync else None

                p1_metrics = self._compute_single_pass_grads(
                    data_p1, ret_temperature, alpha, prefix="retriever", sync_state=sync_state
                )
                append_to_dict(metrics, p1_metrics)

                p2_metrics = self._compute_single_pass_grads(
                    data_p2, gen_temperature, beta, prefix="generator", sync_state=sync_state
                )
                append_to_dict(metrics, p2_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics

    @GPUMemoryLogger(role="dp actor dual_lora", logger=logger)
    def update_policy_dual_lora(self, data_p1: DataProto, data_p2: DataProto):
        """Dual LoRA policy update: each adapter updated independently with zero conflict.

        Each adapter follows the original VERL update_policy() semantics:
        split into PPO mini-batches, accumulate only micro-batches inside one
        mini-batch, then optimizer.step() once per mini-batch.
        """
        self.actor_module.train()
        ret_temperature = float(data_p1.meta_info["temperature"])
        gen_temperature = float(data_p2.meta_info["temperature"])
        ret_scale = data_p1.meta_info.get("loss_scale", 1.0)
        gen_scale = data_p2.meta_info.get("loss_scale", 1.0)
        peft_model = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        metrics = {
            "retriever/loss_scale": float(ret_scale),
            "generator/loss_scale": float(gen_scale),
            "retriever/update_temperature": ret_temperature,
            "generator/update_temperature": gen_temperature,
        }
        optimizers = self._require_dual_lora_optimizers()
        ret_optimizer = optimizers["retriever"]
        gen_optimizer = optimizers["generator"]
        global_steps = data_p1.meta_info.get("global_steps", "unknown")

        def _set_adapter_keep_grad(name):
            """Switch active adapter but keep all LoRA params grad-enabled for FSDP."""
            peft_model.set_adapter(name)
            for n, p in peft_model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        def _zero_all_lora_optimizers():
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)

        for ppo_epoch in range(self.config.ppo_epochs):
            # --- Pass 1: Update retriever adapter ---
            _set_adapter_keep_grad("retriever")
            _zero_all_lora_optimizers()
            self._trace_dual_lora_actor_params(
                event="before_update",
                adapter_name="retriever",
                global_steps=global_steps,
                ppo_epoch=ppo_epoch,
            )
            p1_metrics = self._update_single_pass_policy(
                data_p1,
                temperature=ret_temperature,
                scale=ret_scale,
                prefix="retriever",
                optimizer=ret_optimizer,
            )
            self._trace_dual_lora_actor_params(
                event="after_update",
                adapter_name="retriever",
                global_steps=global_steps,
                ppo_epoch=ppo_epoch,
            )
            append_to_dict(metrics, p1_metrics)

            # Free activation memory before generator pass
            _zero_all_lora_optimizers()
            self._maybe_release_dual_lora_cache()

            # --- Pass 2: Update generator adapter ---
            _set_adapter_keep_grad("generator")
            _zero_all_lora_optimizers()
            self._trace_dual_lora_actor_params(
                event="before_update",
                adapter_name="generator",
                global_steps=global_steps,
                ppo_epoch=ppo_epoch,
            )
            p2_metrics = self._update_single_pass_policy(
                data_p2,
                temperature=gen_temperature,
                scale=gen_scale,
                prefix="generator",
                optimizer=gen_optimizer,
            )
            self._trace_dual_lora_actor_params(
                event="after_update",
                adapter_name="generator",
                global_steps=global_steps,
                ppo_epoch=ppo_epoch,
            )
            append_to_dict(metrics, p2_metrics)

        _zero_all_lora_optimizers()
        return metrics

    @GPUMemoryLogger(role="dp actor single_lora", logger=logger)
    def update_policy_single_lora(self, data: DataProto, adapter_name: str):
        """Update exactly one Dual-LoRA adapter for single-phase training."""
        if adapter_name not in {"retriever", "generator"}:
            raise ValueError(f"Unsupported single LoRA adapter_name={adapter_name!r}")
        optimizers = self._require_dual_lora_optimizers()

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        loss_scale = data.meta_info.get("loss_scale", 1.0)
        optimizer = optimizers[adapter_name]
        peft_model = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        global_steps = data.meta_info.get("global_steps", "unknown")

        def _set_adapter_keep_grad(name):
            peft_model.set_adapter(name)
            for n, p in peft_model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        def _zero_all_lora_optimizers():
            for optimizer_obj in optimizers.values():
                optimizer_obj.zero_grad(set_to_none=True)

        metrics = {f"{adapter_name}/loss_scale": float(loss_scale)}

        for ppo_epoch in range(self.config.ppo_epochs):
            _set_adapter_keep_grad(adapter_name)
            _zero_all_lora_optimizers()
            self._trace_dual_lora_actor_params(
                event="before_update",
                adapter_name=adapter_name,
                global_steps=global_steps,
                ppo_epoch=ppo_epoch,
            )
            adapter_metrics = self._update_single_pass_policy(
                data,
                temperature=temperature,
                scale=loss_scale,
                prefix=adapter_name,
                optimizer=optimizer,
            )
            self._trace_dual_lora_actor_params(
                event="after_update",
                adapter_name=adapter_name,
                global_steps=global_steps,
                ppo_epoch=ppo_epoch,
            )
            append_to_dict(metrics, adapter_metrics)

            _zero_all_lora_optimizers()
            self._maybe_release_dual_lora_cache()

        _zero_all_lora_optimizers()
        return metrics
