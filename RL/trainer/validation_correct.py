"""Fixed validation for Two-Pass GRPO using the training rollout/reward path."""

import json
import time
import gc
import ctypes

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import compute_response_mask

from rewards.joint_reward import aggregate_reward_details


def _trim_process_memory():
    try:
        import sys
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
    except Exception:
        pass


def validate_training_rollout(trainer_self):
    """Validation on a fixed parquet using the training rollout/reward path.

    The validation dataset must be prepared before training. This function does
    not sample from the train set at runtime. It uses the same N_ret/N_gen,
    validation temperature, dual-LoRA routing, generator-prompt construction, and reward
    manager logic as the training step, with update_pool=False.
    """
    t0 = time.time()

    tokenizer = trainer_self.tokenizer
    N_gen = int(getattr(trainer_self, "gen_n", trainer_self.config.actor_rollout_ref.rollout.n))
    N_ret = int(getattr(trainer_self, "ret_n", N_gen))
    if hasattr(trainer_self, "_rollout_temperature"):
        retriever_temperature = trainer_self._rollout_temperature("validation", "retriever", validation=True)
        generator_temperature = trainer_self._rollout_temperature("validation", "generator", validation=True)
    else:
        retriever_temperature = trainer_self.config.actor_rollout_ref.rollout.temperature
        generator_temperature = retriever_temperature

    # size_divisor for padding (same logic as training)
    if trainer_self.async_rollout_mode:
        size_divisor = len(trainer_self.async_rollout_manager.agent_loop_workers)
    else:
        size_divisor = trainer_self.actor_rollout_wg.world_size

    all_ret_rewards = []
    all_gen_rewards = []
    all_details = []
    all_ret_reward_mapped = []
    total_samples = 0

    # Use the same rollout context as training
    trainer_self._enter_rollout_context()

    try:
        for val_data in trainer_self.val_dataloader:
            val_batch = DataProto.from_single_dict(val_data)
            bs = int(val_batch.batch.batch_size[0])
            total_samples += bs

            if "uid" not in val_batch.non_tensor_batch:
                val_batch.non_tensor_batch["uid"] = np.array(
                    [f"fixed-val-{trainer_self.global_steps}-{len(all_details)}-{i}" for i in range(bs)],
                    dtype=object,
                )

            # ==========================================================
            # Phase 1: Retriever rollout, same as training
            # ==========================================================
            ret_input = trainer_self._prepare_rollout_batch(val_batch)
            ret_input.meta_info["temperature"] = retriever_temperature
            ret_input.meta_info["global_steps"] = trainer_self.global_steps
            ret_input.meta_info["validate"] = True
            if hasattr(trainer_self, "_set_rollout_response_length"):
                trainer_self._set_rollout_response_length(ret_input, "retriever")
            if getattr(trainer_self, "dual_lora", False):
                ret_input.meta_info["_lora_int_id"] = 1
            ret_input = ret_input.repeat(repeat_times=N_ret, interleave=True)

            ret_input_padded, ret_pad = pad_dataproto_to_divisor(ret_input, size_divisor)
            ret_output_padded = trainer_self._generate_in_context(ret_input_padded)
            ret_output = unpad_dataproto(ret_output_padded, ret_pad)

            ret_batch = val_batch.repeat(repeat_times=N_ret, interleave=True)
            for key, val in ret_batch.non_tensor_batch.items():
                if key not in ret_output.non_tensor_batch:
                    ret_output.non_tensor_batch[key] = val
            ret_batch = ret_output
            if "response_mask" not in ret_batch.batch.keys():
                ret_batch.batch["response_mask"] = compute_response_mask(ret_batch)
            ret_batch.meta_info["temperature"] = retriever_temperature

            # ==========================================================
            # Phase 2: Build generator prompts, same as training
            # ==========================================================
            gen_input = trainer_self._build_generator_batch(ret_batch)

            # ==========================================================
            # Phase 3: Generator rollout, same as training
            # ==========================================================
            gen_input.meta_info["temperature"] = generator_temperature
            gen_input.meta_info["global_steps"] = trainer_self.global_steps
            gen_input.meta_info["validate"] = True
            if hasattr(trainer_self, "_set_rollout_response_length"):
                trainer_self._set_rollout_response_length(gen_input, "generator")
            if getattr(trainer_self, "dual_lora", False):
                gen_input.meta_info["_lora_int_id"] = 2

            gen_input_padded, gen_pad = pad_dataproto_to_divisor(gen_input, size_divisor)
            gen_output_padded = trainer_self._generate_in_context(gen_input_padded)
            gen_output = unpad_dataproto(gen_output_padded, gen_pad)

            gen_batch = gen_output
            for key, val in gen_input.non_tensor_batch.items():
                if key not in gen_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch[key] = val
            if "response_mask" not in gen_output.batch.keys():
                gen_batch.batch["response_mask"] = compute_response_mask(gen_batch)
            gen_batch.meta_info["temperature"] = generator_temperature

            gen_texts = tokenizer.batch_decode(
                gen_batch.batch["responses"], skip_special_tokens=True,
            )
            ret_texts = list(gen_batch.non_tensor_batch["ret_texts"])
            ret_resp_lens = list(gen_batch.non_tensor_batch["ret_resp_lens"])
            gen_resp_lens = torch.sum(gen_batch.batch["response_mask"], dim=-1).tolist()

            ground_truths = []
            extra_infos = []
            for i in range(len(ret_texts)):
                rm_obj = gen_batch.non_tensor_batch["reward_model"][i]
                rm = json.loads(rm_obj) if isinstance(rm_obj, str) else rm_obj
                ground_truths.append(rm.get("ground_truth", {}))

                ex_obj = gen_batch.non_tensor_batch["extra_info"][i]
                extra_infos.append(json.loads(ex_obj) if isinstance(ex_obj, str) else ex_obj)

            # ==========================================================
            # Phase 4: Compute training rewards, without updating the pool
            # ==========================================================
            ret_rewards, gen_rewards, details = trainer_self.reward_manager.compute_rewards(
                ret_texts=ret_texts,
                gen_texts=gen_texts,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                group_ids=list(gen_batch.non_tensor_batch["uid"]),
                update_pool=False,
                ret_n=N_ret,
                gen_resp_lens=gen_resp_lens,
                ret_resp_lens=ret_resp_lens,
                ret_response_length=int(ret_batch.batch["responses"].shape[1]),
            )

            if N_ret == N_gen:
                ret_rewards_mapped = ret_rewards
            elif N_ret < N_gen:
                num_q = len(ret_rewards) // N_gen
                ret_rewards_mapped = []
                for q in range(num_q):
                    for i in range(N_ret):
                        ret_rewards_mapped.append(ret_rewards[q * N_gen + i])
            else:
                # In merged-union alternating configs the generator group can be
                # smaller than the retriever group. Validation still reports the
                # generator-aligned retriever rewards from the actual SQL prompts.
                ret_rewards_mapped = ret_rewards

            all_ret_rewards.extend(ret_rewards)
            all_ret_reward_mapped.extend(ret_rewards_mapped)
            all_gen_rewards.extend(gen_rewards)
            all_details.extend(details)

            del gen_output_padded, gen_output, gen_input
            del ret_output_padded, ret_output, ret_input
            del ret_texts, gen_texts, ret_rewards, gen_rewards, details
            del ground_truths, extra_infos, ret_rewards_mapped, val_batch, ret_batch, gen_batch
            gc.collect()
            _trim_process_memory()

    finally:
        trainer_self._exit_rollout_context()

    # ==========================================================
    # Aggregate metrics
    # ==========================================================
    val_time = time.time() - t0
    metrics = {}
    if not all_details:
        return metrics

    exec_statuses = [d.get("exec_status", "Unexecutable") for d in all_details if d is not None]
    n_total = len(exec_statuses)
    n_match = sum(1 for status in exec_statuses if status == "Match")

    metrics["val/exec_match_rate"] = n_match / n_total if n_total > 0 else 0.0
    metrics["val/ret_reward_mean"] = float(np.mean(all_ret_reward_mapped))
    metrics["val/ret_reward_all_gen_aligned_mean"] = float(np.mean(all_ret_rewards))
    metrics["val/gen_reward_mean"] = float(np.mean(all_gen_rewards))
    metrics["val/ret_reward_std"] = float(np.std(all_ret_reward_mapped))
    metrics["val/gen_reward_std"] = float(np.std(all_gen_rewards))
    metrics["val/time_s"] = val_time
    metrics["val/fixed_sample_count"] = total_samples
    metrics["val/rollout_n"] = N_gen
    metrics["val/ret_n"] = N_ret
    metrics["val/temperature_retriever"] = retriever_temperature
    metrics["val/temperature_generator"] = generator_temperature

    # Detailed metrics from training reward decomposition
    agg_metrics = aggregate_reward_details(all_details, N=N_gen, prefix="val")
    metrics.update(agg_metrics)

    # Coverage
    coverages = [d.get("coverage_rate", 0.0) for d in all_details if d is not None]
    metrics["val/coverage_rate_mean"] = float(np.mean(coverages)) if coverages else 0.0

    print(
        f"[Validation] step={trainer_self.global_steps} time={val_time:.1f}s "
        f"exec_match={metrics['val/exec_match_rate']:.3f} ({n_match}/{n_total}) "
        f"coverage={metrics.get('val/coverage_rate_mean', 0):.3f} "
        f"ret_r={metrics['val/ret_reward_mean']:.2f} "
        f"gen_r={metrics['val/gen_reward_mean']:.2f} "
        f"temp_ret={retriever_temperature:.2f} temp_gen={generator_temperature:.2f} "
        f"N_ret={N_ret} N_gen={N_gen} samples={total_samples} "
        f"[fixed-val+training-rollout-reward]",
        flush=True,
    )

    return metrics


def validate_greedy(trainer_self):
    """Backward-compatible alias for older imports."""
    return validate_training_rollout(trainer_self)
