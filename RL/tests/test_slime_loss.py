"""CPU checks for role normalization and the weighted, unprojected objective.

Set SLIME_SOURCE_DIR to the pinned checkout to also exercise the actual upstream
policy_loss_function without importing CUDA/Megatron during this CPU test.
"""

import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace_sql_slime.loss import policy_loss, reward_postprocess, validate_loss_configuration, weighted_sample_reducer


def configuration(**overrides):
    values = dict(
        advantage_estimator="grpo", context_parallel_size=1, calculate_per_token_loss=False,
        allgather_cp=False, normalize_advantages=False, grpo_std_normalization=True,
        kl_coef=0.0, ref_update_interval=None, ace_group_size=2, use_opsm=False,
        use_tis=False, get_mismatch_metrics=False, custom_pg_loss_reducer_function_path=None,
        use_rollout_logprobs=False, use_kl_loss=True, use_unbiased_kl=False,
        eps_clip=0.2, eps_clip_high=0.2, entropy_coef=0.0, kl_loss_coef=0.03,
        kl_loss_type="low_var_kl",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class Sample:
    def __init__(self, question, role, reward, weight=0.5):
        self.group_index = question
        self.train_metadata = {"role": role, "loss_weight": weight}
        self.metadata = {}
        self.reward = reward

    def get_reward_value(self, _args):
        return self.reward


def make_batch(ret_weight, gen_weight):
    return {
        "response_lengths": [2, 1, 3, 2],
        "loss_masks": [torch.tensor(mask, dtype=torch.float64) for mask in ([1, 1], [1], [1, 0, 1], [1, 1])],
        "metadata": [{"role": role, "loss_weight": weight} for role, weight in (
            ("retriever", ret_weight), ("generator", gen_weight),
            ("retriever", ret_weight), ("generator", gen_weight),
        )],
    }


def reducer_for(batch):
    def reduce(values):
        return sum((part * mask).sum() / mask.sum().clamp_min(1) for part, mask in zip(
            values.split(batch["response_lengths"]), batch["loss_masks"], strict=True))
    return reduce


def oracle_token_objective(log_probs, old, reference, advantages):
    ratio = torch.exp(log_probs - old)
    pg = torch.maximum(-advantages * ratio, -advantages * torch.clamp(ratio, 0.8, 1.2))
    log_ratio = reference - log_probs
    kl = torch.clamp(log_ratio.exp() - 1 - log_ratio, min=-10, max=10)
    return pg + 0.03 * kl


class WeightedLossTests(unittest.TestCase):
    def test_analytic_loss_and_gradients_with_unequal_lengths(self):
        for wr, wg in ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5)):
            with self.subTest(weights=(wr, wg)):
                batch = make_batch(wr, wg)
                current = torch.tensor([0.02, -0.03, 0.05, 0.5, -0.5, 0.02, -0.04, 0.1], dtype=torch.float64, requires_grad=True)
                old = torch.zeros_like(current)
                reference = torch.full_like(current, 0.08)
                advantage = torch.tensor([1, 1, -1, 0.5, 0.5, 0.5, 0.7, 0.7], dtype=torch.float64)
                values = oracle_token_objective(current, old, reference, advantage)
                actual = weighted_sample_reducer(batch, reducer_for(batch))(values) / 4
                individual = [(x * m).sum() / m.sum() for x, m in zip(values.split(batch["response_lengths"]), batch["loss_masks"], strict=True)]
                expected = wr * (individual[0] + individual[2]) / 2 + wg * (individual[1] + individual[3]) / 2
                torch.testing.assert_close(actual, expected)
                actual_gradient = torch.autograd.grad(actual, current, retain_graph=True)[0]
                expected_gradient = torch.autograd.grad(expected, current)[0]
                torch.testing.assert_close(actual_gradient, expected_gradient)
                self.assertEqual(actual_gradient[4], 0)  # masked token
                if wg == 0:
                    torch.testing.assert_close(actual_gradient[[2, 6, 7]], torch.zeros(3, dtype=current.dtype))

    def test_one_role_microbatch_preserves_global_denominator(self):
        batch = make_batch(0.75, 0.25)
        values = torch.arange(8, dtype=torch.float64, requires_grad=True)
        whole = weighted_sample_reducer(batch, reducer_for(batch))(values)
        parts = values.split(batch["response_lengths"])
        accumulated = values.sum() * 0
        for indices in ([0, 2], [1, 3]):
            local = {key: [batch[key][i] for i in indices] for key in batch}
            accumulated = accumulated + weighted_sample_reducer(local, reducer_for(local))(torch.cat([parts[i] for i in indices]))
        torch.testing.assert_close(whole, accumulated)

    def test_whole_objective_including_kl_is_weighted(self):
        batch = make_batch(1.0, 0.0)
        logits = torch.arange(8, dtype=torch.float64, requires_grad=True)
        module = ModuleType("slime.backends.megatron_utils.loss")
        # Three upstream reductions model PG, entropy, and reference KL.
        def upstream(args, data, values, reducer):
            loss = reducer(values) - 0.2 * reducer(values.square()) + 0.03 * reducer(values.exp())
            return loss, {"loss": loss.detach()}
        module.policy_loss_function = upstream
        with patch.dict(sys.modules, {module.__name__: module}):
            value, _ = policy_loss(configuration(), batch, logits, reducer_for(batch))
        gradient = torch.autograd.grad(value, logits)[0]
        torch.testing.assert_close(gradient[[2, 6, 7]], torch.zeros(3, dtype=logits.dtype))

    def test_missing_metadata_and_unsupported_settings_fail(self):
        batch = make_batch(0.5, 0.5)
        del batch["metadata"]
        with self.assertRaisesRegex(ValueError, "metadata"):
            weighted_sample_reducer(batch, reducer_for(batch))
        cases = [dict(calculate_per_token_loss=True), dict(context_parallel_size=2),
                 dict(normalize_advantages=True), dict(use_tis=True), dict(get_mismatch_metrics=True),
                 dict(use_opsm=True), dict(kl_coef=0.001), dict(ref_update_interval=10),
                 dict(advantage_estimator="gspo"), dict(custom_pg_loss_reducer_function_path="other")]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_loss_configuration(configuration(**values))


class RewardNormalizationTests(unittest.TestCase):
    def test_interleaved_roles_and_questions_use_unbiased_std(self):
        samples = [Sample(0, "retriever", 0), Sample(1, "generator", 7),
                   Sample(0, "generator", 10), Sample(1, "retriever", 3),
                   Sample(0, "retriever", 1), Sample(1, "generator", 7),
                   Sample(0, "generator", 20), Sample(1, "retriever", 3)]
        raw, normalized = reward_postprocess(configuration(), samples)
        self.assertEqual(raw, [0, 7, 10, 3, 1, 7, 20, 3])
        expected = [-0.5 / (math.sqrt(0.5) + 1e-6), 0, -5 / (math.sqrt(50) + 1e-6),
                    0, 0.5 / (math.sqrt(0.5) + 1e-6), 0, 5 / (math.sqrt(50) + 1e-6), 0]
        torch.testing.assert_close(torch.tensor(normalized), torch.tensor(expected))

    def test_row_key_fallback_and_zero_generator_weight_still_normalizes(self):
        samples = [Sample(None, role, reward, weight) for role, weight in (("retriever", 1), ("generator", 0)) for reward in (0, 1)]
        for sample in samples:
            sample.train_metadata["row_key"] = "db||question"
        _, values = reward_postprocess(configuration(n_samples_per_prompt=4), samples)
        self.assertEqual(values[:2], values[2:])
        self.assertLess(values[0], 0)
        self.assertGreater(values[1], 0)

    def test_incomplete_group_and_old_noncomplementary_weights_fail(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            reward_postprocess(configuration(), [Sample(0, "retriever", 0)])
        samples = [Sample(0, role, reward, 1) for role in ("retriever", "generator") for reward in (0, 1)]
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            reward_postprocess(configuration(), samples)


@unittest.skipUnless(os.environ.get("SLIME_SOURCE_DIR"), "Set SLIME_SOURCE_DIR for pinned upstream integration")
class UpstreamIntegrationTests(unittest.TestCase):
    def test_patched_metadata_survives_partition_and_microbatch_and_logging(self):
        source = Path(os.environ["SLIME_SOURCE_DIR"])
        script = Path(__file__).resolve().parents[1] / "scripts/apply_slime_metadata_patch.py"
        spec = importlib.util.spec_from_file_location("metadata_patch", script)
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)
        namespace = {"torch": torch, "Sequence": list,
                     "ray": SimpleNamespace(put=lambda value: value), "Box": lambda value: value}
        selected = []
        for relative in ("slime/ray/rollout.py", "slime/backends/megatron_utils/data.py"):
            text = (source / relative).read_text()
            before, after = patch_module.FILES[relative]
            text = text.replace(before, after)
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
                    "_split_train_data_by_dp", "DataIterator", "log_rollout_data"
                }:
                    selected.append(node)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "patched_metadata_transport", "exec"), namespace)
        data = {"tokens": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "metadata": [{"role": "retriever" if i % 2 == 0 else "generator", "loss_weight": 0.5, "row": i} for i in range(4)],
                "response_lengths": [1, 1, 1, 1], "rewards": [0, 0.5, 1, 0.5],
                "loss_masks": [torch.ones(1)] * 4}
        partitions = namespace["_split_train_data_by_dp"](SimpleNamespace(args=SimpleNamespace(balance_data=False)), data, 2)
        self.assertEqual([m["row"] for m in partitions[0]["metadata"]], [0, 2])
        self.assertEqual([m["row"] for m in partitions[1]["metadata"]], [1, 3])
        iterator = namespace["DataIterator"](partitions[0], micro_batch_size=None, micro_batch_indices=[[1, 0]])
        batch = iterator.get_next(["tokens", "metadata"])
        self.assertEqual([m["row"] for m in batch["metadata"]], [2, 0])
        self.assertEqual(batch["tokens"], [[1, 2, 3], [1]])
        logged = {}
        namespace["mpu"] = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0,
                                            is_pipeline_last_stage=lambda: True,
                                            get_context_parallel_world_size=lambda: 1)
        namespace["gather_log_data"] = lambda _tag, _args, _id, metrics: logged.update(metrics)
        rank_data = partitions[0]
        rank_data["total_lengths"] = [rank_data["total_lengths"][i] for i in rank_data.pop("partition")]
        namespace["log_rollout_data"](0, SimpleNamespace(ci_test=False, log_multi_turn=False,
                                                         log_passrate=False, log_correct_samples=False), rank_data)
        self.assertNotIn("metadata", logged)
        self.assertEqual(logged["rewards"], 0.5)

    def test_actual_upstream_policy_and_kl_function(self):
        """Execute upstream source with only the CUDA logits extraction replaced."""
        source = Path(os.environ["SLIME_SOURCE_DIR"])
        namespace = {"torch": torch}
        for relative, names in (("slime/utils/ppo_utils.py", {"compute_policy_loss", "compute_approx_kl"}),
                                ("slime/backends/megatron_utils/loss.py", {"policy_loss_function"})):
            tree = ast.parse((source / relative).read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            self.assertEqual(len(functions), len(names))
            for node in functions:
                node.decorator_list = []  # avoid compiling in the CPU test
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(source / relative), "exec"), namespace)
        def logprobs(logits, **kwargs):
            parts = list(logits.split(kwargs["response_lengths"]))
            return None, {"log_probs": parts, "entropy": [part * 0 for part in parts]}
        namespace["get_log_probs_and_entropy"] = logprobs
        upstream = ModuleType("slime.backends.megatron_utils.loss")
        upstream.policy_loss_function = namespace["policy_loss_function"]
        for wr, wg in ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5)):
            batch = make_batch(wr, wg)
            logits = torch.tensor([0.02, -0.03, 0.05, 0.5, -0.5, 0.02, -0.04, 0.1], requires_grad=True)
            sizes = batch["response_lengths"]
            old = torch.zeros_like(logits)
            reference = torch.full_like(logits, 0.08)
            advantages = torch.tensor([1, 1, -1, 0.5, 0.5, 0.5, 0.7, 0.7])
            batch.update(log_probs=list(old.split(sizes)), ref_log_probs=list(reference.split(sizes)),
                         advantages=list(advantages.split(sizes)), total_lengths=[length + 1 for length in sizes],
                         unconcat_tokens=[None] * 4)
            with patch.dict(sys.modules, {upstream.__name__: upstream}):
                actual, metrics = policy_loss(configuration(), batch, logits, reducer_for(batch))
            actual = actual / 4
            token_values = oracle_token_objective(logits, old, reference, advantages)
            means = [(part * mask).sum() / mask.sum() for part, mask in zip(token_values.split(sizes), batch["loss_masks"], strict=True)]
            expected = wr * (means[0] + means[2]) / 2 + wg * (means[1] + means[3]) / 2
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0])
            self.assertIn("kl_loss", metrics)


if __name__ == "__main__":
    unittest.main()
