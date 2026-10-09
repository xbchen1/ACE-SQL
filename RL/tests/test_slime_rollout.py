"""CPU rollout contracts with real SQL grading and mocked inference transport.

Set ACE_SQL_TEST_SLIME_DIR to a pinned checkout to use its actual dataclasses
and validate the launcher against its argument declarations. Without a checkout,
only these small transport dataclasses are substituted; no GPU library is used.
"""

import ast
import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from ace_sql_slime import launch, rollout
from ace_sql_slime.data import row_to_record


SLIME_ROOT = Path(os.environ.get("ACE_SQL_TEST_SLIME_DIR", "/tmp/ace-sql-slime-v023"))


def _dataclasses():
    """Load the real upstream contract when installed, else a narrow CPU stand-in."""
    if (SLIME_ROOT / "slime/utils/types.py").is_file():
        spec = importlib.util.spec_from_file_location("_ace_test_upstream_types", SLIME_ROOT / "slime/utils/types.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.Sample

    @dataclass
    class Sample:
        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"

        group_index: int = 0
        index: int = 0
        prompt: object = ""
        label: object = None
        metadata: dict = field(default_factory=dict)
        tokens: list = field(default_factory=list)
        response: str = ""
        response_length: int = 0
        rollout_log_probs: list = field(default_factory=list)
        loss_mask: list = field(default_factory=list)
        status: object = Status.PENDING
        reward: object = None
        train_metadata: object = None

    return Sample


Sample = _dataclasses()


class Tokenizer:
    eos_token_id = 999

    def __init__(self):
        self.prompts = {}

    def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return json.dumps(prompt, ensure_ascii=False) + "\nassistant\n<think>\n"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        index = len(self.prompts) + 1000
        self.prompts[index] = text
        return [7, 8, index]


class Server:
    """Returns sampled token IDs unrelated to encode(response), as SGLang does."""

    def __init__(self, tokenizer, ret_responses=None):
        self.tokenizer = tokenizer
        self.ret_responses = ret_responses or ["choose a</think>\n```\n[t.a]\n```"]
        self.calls = []
        self.counts = {"retriever": 0, "generator": 0}

    async def post(self, url, payload):
        text = self.tokenizer.prompts[payload["input_ids"][-1]]
        role = "retriever" if "RETRIEVE::" in text else "generator"
        count = self.counts[role]
        self.counts[role] += 1
        self.calls.append({"url": url, "payload": deepcopy(payload), "prompt": text, "role": role})
        response = (self.ret_responses[count % len(self.ret_responses)] if role == "retriever"
                    else "query</think>\n```sql\nSELECT b FROM t\n```")
        return {"text": response, "meta_info": {
            "output_token_logprobs": [[-0.2, 101, None], [-0.3, 102, None], [-0.1, 999, None]],
            "completion_tokens": 3, "finish_reason": {"type": "stop"}}}


class RolloutTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "toy.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE t(a INTEGER PRIMARY KEY, b INTEGER)")
            connection.executemany("INSERT INTO t VALUES (?, ?)", [(1, 1), (2, 2)])
        self.records = [self.record(f"question {i}") for i in range(5)]
        self.validation = [self.record("held out")]
        for filename, rows in (("train.jsonl", self.records), ("validation.jsonl", self.validation)):
            (self.root / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.pool = {row["metadata"]["row_key"]: {"1": 2.0} for row in self.records + self.validation}
        (self.root / "initial_pool.json").write_text(json.dumps(self.pool))
        self.args = SimpleNamespace(
            prompt_data=str(self.root / "train.jsonl"), ace_validation_data=str(self.root / "validation.jsonl"),
            ace_initial_pool=str(self.root / "initial_pool.json"), ace_group_size=8,
            n_samples_per_prompt=16, rollout_batch_size=2, global_batch_size=32,
            ace_pool_gamma=0.5, ace_ramp_ratio=0.25, num_rollout=20, rollout_seed=42,
            save=str(self.root / "save"), load=str(self.root / "save"),
            rollout_max_prompt_len=4096, rollout_max_response_len=2048, rollout_temperature=1.0,
            rollout_top_p=1.0, rollout_top_k=-1, sglang_router_ip="127.0.0.1", sglang_router_port=30000,
            ace_concurrency=4, hf_checkpoint="unused-tokenizer", reward_key=None,
        )
        self.tokenizer = Tokenizer()
        self.server = Server(self.tokenizer)
        type_module = ModuleType("slime.utils.types")
        type_module.Sample = Sample
        base_module = ModuleType("slime.rollout.base_types")
        base_module.RolloutFnTrainOutput = lambda **kw: SimpleNamespace(**kw)
        base_module.RolloutFnEvalOutput = lambda **kw: SimpleNamespace(**kw)
        http_module = ModuleType("slime.utils.http_utils")

        async def post(url, payload):
            return await self.server.post(url, payload)

        http_module.post = post
        tokenizer_module = ModuleType("transformers")
        tokenizer_module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: self.tokenizer)
        self.modules = patch.dict(sys.modules, {
            "slime.utils.types": type_module, "slime.rollout.base_types": base_module,
            "slime.utils.http_utils": http_module, "transformers": tokenizer_module,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def record(self, question):
        schema = {"db_id": "toy", "table_names_original": ["t"],
                  "column_names_original": [[-1, "*"], [0, "a"], [0, "b"]],
                  "column_names": [[-1, "*"], [0, "a"], [0, "b"]],
                  "column_types": ["text", "integer", "integer"], "primary_keys": [1], "foreign_keys": []}
        gt = {"db_id": "toy", "db_path": str(self.database), "question": question,
              "sql": "SELECT a FROM t", "gold_column_indices": [1]}
        extra = {"db_id": "toy", "question": question, "evidence": "", "db_info": json.dumps(schema),
                 "sampled_values": "{}", "relavant_values": "{}"}
        return row_to_record({"prompt": [{"role": "user", "content": "RETRIEVE::" + question}],
                              "reward_model": {"ground_truth": gt}, "extra_info": extra})

    def source(self, args=None):
        source = rollout.ACEDataSource(args or self.args)
        self.addCleanup(source.reward_engine.close)
        source.tokenizer = self.tokenizer
        return source

    def run_rollout(self, source, step=0, evaluation=False):
        return asyncio.run(rollout._rollout(source.args, step, source, evaluation))

    def state(self, source):
        return deepcopy((source.epoch, source.offset, source.group_index, source.sample_index,
                         source.last_rollout_id, source.order, source.reward_engine.state_dict(),
                         source.provenance, source.records, source.validation))

    def test_two_pass_uses_eight_plus_eight_with_role_weights(self):
        source = self.source()
        output = self.run_rollout(source)
        self.assertEqual(self.server.counts, {"retriever": 16, "generator": 16})
        self.assertEqual([len(group) for group in output.samples], [8, 8, 8, 8])
        self.assertEqual(output.metrics["ace/generator_ex"], 1.0)
        for ret, gen in zip(output.samples[::2], output.samples[1::2]):
            self.assertEqual({s.train_metadata["role"] for s in ret}, {"retriever"})
            self.assertEqual({s.train_metadata["role"] for s in gen}, {"generator"})
            self.assertEqual({s.train_metadata["loss_weight"] for s in ret}, {1.0})
            self.assertEqual({s.train_metadata["loss_weight"] for s in gen}, {0.0})
            self.assertEqual({s.reward for s in ret}, {0.0})
            self.assertEqual({s.reward for s in gen}, {1.0})
            self.assertTrue(all(s.prompt == gen[0].prompt for s in gen))
            self.assertEqual(len({s.group_index for s in ret + gen}), 1)
            self.assertEqual(source.provenance[ret[0].metadata["row_key"]]["SELECT b FROM t"], 8)
        all_samples = [s for group in output.samples for s in group]
        self.assertEqual(len({s.index for s in all_samples}), 32)
        self.assertEqual(source.sample_index, 32)
        self.assertEqual(sorted(call["payload"]["sampling_params"]["sampling_seed"] for call in self.server.calls),
                         list(range(42, 74)))
        self.assertTrue((Path(self.args.save) / "ace_audit/rollout_0.json").is_file())
        output = self.run_rollout(source, 1)
        self.assertEqual(output.metrics["ace/retriever_weight"], 0.9)
        self.assertEqual(output.metrics["ace/generator_weight"], 0.1)

    def test_empty_vote_has_no_schema_fallback(self):
        self.server = Server(self.tokenizer, ["no list</think>"])
        output = self.run_rollout(self.source())
        for group in output.samples[1::2]:
            self.assertNotIn("CREATE TABLE", group[0].prompt[0]["content"])
        self.assertEqual(self.server.counts["generator"], 16)

    def test_think_prefix_is_only_in_parser_view(self):
        sample = self.source().get_samples(2)[0][0]
        asyncio.run(rollout._generate(self.args, sample, self.tokenizer, asyncio.Semaphore(1)))
        before = deepcopy((sample.tokens, sample.response, sample.response_length,
                           sample.rollout_log_probs, sample.loss_mask))
        self.assertTrue(rollout.parser_response(sample).startswith("<think>"))
        self.assertFalse(sample.response.startswith("<think>"))
        self.assertEqual(sample.response_length, 3)
        self.assertEqual(sample.tokens[-3:], [101, 102, 999])
        self.assertEqual(sample.rollout_log_probs, [-0.2, -0.3, -0.1])
        self.assertEqual(sample.loss_mask, [1, 1, 1])
        self.assertEqual(before, (sample.tokens, sample.response, sample.response_length,
                                  sample.rollout_log_probs, sample.loss_mask))
        sample.response = "<think>already explicit</think>[t.a]"
        self.assertEqual(rollout.parser_response(sample), sample.response)

    def test_generation_rejects_missing_or_invalid_token_logprobs(self):
        valid = {"text": "response", "meta_info": {"output_token_logprobs": [[-0.1, 20, None]],
                 "completion_tokens": 1, "finish_reason": {"type": "stop"}}}
        variants = []
        for logprobs in (None, [], [[float("nan"), 20, None]], [[float("inf"), 20, None]]):
            variant = deepcopy(valid)
            variant["meta_info"]["output_token_logprobs"] = logprobs
            variants.append(variant)
        variant = deepcopy(valid)
        variant["meta_info"]["completion_tokens"] = 2
        variants.append(variant)
        variant = deepcopy(valid)
        variant["meta_info"]["finish_reason"] = {"type": "abort"}
        variants.append(variant)
        for response in variants:
            async def post(url, payload):
                return response
            self.server.post = post
            sample = Sample(index=0, prompt=self.records[0]["prompt"], metadata={})
            with self.subTest(response=response), self.assertRaises((RuntimeError, ValueError)):
                asyncio.run(rollout._generate(self.args, sample, self.tokenizer, asyncio.Semaphore(1)))

    def test_overlong_retriever_fails_and_generator_left_truncates(self):
        class LongTokenizer(Tokenizer):
            def encode(inner, text, add_special_tokens=False):
                inner.last_full_ids = [42] * 5000 + super().encode(text, add_special_tokens=add_special_tokens)
                return inner.last_full_ids

        self.tokenizer = LongTokenizer()
        self.server = Server(self.tokenizer)
        sample = Sample(index=0, prompt=self.records[0]["prompt"], metadata={})
        with self.assertRaisesRegex(ValueError, "Retriever prompt exceeds"):
            asyncio.run(rollout._generate(self.args, sample, self.tokenizer, asyncio.Semaphore(1)))
        self.assertEqual(self.server.calls, [])
        self.assertEqual(sample.tokens, [])
        asyncio.run(rollout._generate(self.args, sample, self.tokenizer, asyncio.Semaphore(1), role="generator"))
        expected_ids = self.tokenizer.last_full_ids[-4096:]
        self.assertEqual(self.server.calls[0]["payload"]["input_ids"], expected_ids)
        self.assertEqual(sample.tokens[:-sample.response_length], expected_ids)

    def test_partial_reward_failure_rolls_back_and_retry_preserves_seeds(self):
        source = self.source()
        before = self.state(source)
        real_compute = source.reward_engine.compute_group
        observed = []

        def fail_second(*args, **kwargs):
            observed.append(self.state(source))
            if len(observed) == 2:
                raise RuntimeError("injected failure after first reward group")
            return real_compute(*args, **kwargs)

        with patch.object(source.reward_engine, "compute_group", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.run_rollout(source)
        self.assertEqual(len(observed), 2)
        self.assertNotEqual(observed[0][6], observed[1][6])  # first group's pool did update
        self.assertNotEqual(observed[0][7], observed[1][7])  # and its SQL provenance did update
        self.assertEqual(self.state(source), before)
        self.assertFalse((Path(self.args.save) / "ace_audit/rollout_0.json").exists())
        first_seeds = [call["payload"]["sampling_params"]["sampling_seed"] for call in self.server.calls]
        self.server.calls.clear()
        retried = self.run_rollout(source)
        retry_seeds = [call["payload"]["sampling_params"]["sampling_seed"] for call in self.server.calls]
        self.assertEqual(first_seeds, retry_seeds)
        self.assertEqual(source.last_rollout_id, 0)
        self.assertEqual(source.offset, 2)
        for group in retried.samples[::2]:
            expected = 0.0 if group[0].train_metadata["role"] == "retriever" else 1.0
            self.assertEqual([sample.reward for sample in group], [expected] * 8)
            self.assertEqual(source.provenance[group[0].metadata["row_key"]]["SELECT b FROM t"], 8)

    def test_failed_generation_gather_cancels_other_requests(self):
        events = []

        async def run():
            pending_started = asyncio.Event()

            async def fail():
                await pending_started.wait()
                raise RuntimeError("transport failed")

            async def pending():
                pending_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    events.append("cancelled")

            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                await rollout._gather([fail(), pending()])

        asyncio.run(run())
        self.assertEqual(events, ["cancelled"])

    def test_validation_is_greedy_one_route_and_immutable(self):
        source = self.source()
        self.run_rollout(source)
        before = self.state(source)
        self.server.calls.clear()
        self.server.counts = {"retriever": 0, "generator": 0}
        result = self.run_rollout(source, 0, evaluation=True)
        self.assertEqual(before, self.state(source))
        self.assertEqual(self.server.counts, {"retriever": 1, "generator": 1})
        self.assertEqual(result.data["ace_sql_greedy_ex"]["rewards"], [1.0])
        for call in self.server.calls:
            self.assertEqual(call["payload"]["sampling_params"]["temperature"], 0.0)
            self.assertNotIn("sampling_seed", call["payload"]["sampling_params"])

    def test_checkpoint_replay_restores_pool_cursor_and_drop_last_order(self):
        source = self.source()
        self.run_rollout(source, 0)
        self.run_rollout(source, 1)
        source.save(1)
        saved = self.state(source)
        restored = self.source()
        restored.load(1)
        self.assertEqual(saved, self.state(restored))
        expected = self.run_rollout(source, 2)
        replay = self.run_rollout(restored, 2)
        self.assertEqual(source.epoch, 1)  # fifth row was dropped before the next full batch
        self.assertEqual(self.state(source), self.state(restored))
        expected_samples = [(s.metadata["row_key"], s.index, s.group_index, s.reward, s.train_metadata)
                            for group in expected.samples for s in group]
        replay_samples = [(s.metadata["row_key"], s.index, s.group_index, s.reward, s.train_metadata)
                          for group in replay.samples for s in group]
        self.assertEqual(expected_samples, replay_samples)

    def test_checkpoint_missing_mismatch_and_nonsequential_are_rejected(self):
        source = self.source()
        with self.assertRaises(FileNotFoundError):
            source.load(0)
        with self.assertRaises(ValueError):
            source.save(0)
        with self.assertRaises(ValueError):
            self.run_rollout(source, 1)
        self.run_rollout(source, 0)
        source.save(0)
        incompatible_args = deepcopy(self.args)
        incompatible_args.rollout_seed += 1
        with self.assertRaises(ValueError):
            self.source(incompatible_args).load(0)
        path = Path(self.args.save) / "rollout/ace_state_0.json"
        state = json.loads(path.read_text())
        state["rollout_id"] = 1
        path.write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            self.source().load(0)


class LauncherTest(unittest.TestCase):
    def test_fresh_run_keeps_reference_separate_from_resume_load(self):
        kwargs = dict(slime=Path("slime"), model=Path("model"), reference=Path("reference"),
                      data=Path("data"), output=Path("output"), steps=910)
        fresh = launch.build_command(**kwargs)
        self.assertNotIn("--load", fresh)
        self.assertEqual(fresh[fresh.index("--ref-load") + 1], "reference")
        resumed = launch.build_command(**kwargs, resume=Path("previous-run"))
        self.assertEqual(resumed[resumed.index("--load") + 1], "previous-run")
        self.assertEqual(resumed[resumed.index("--ref-load") + 1], "reference")

    def test_default_paper_plan_has_910_steps_and_full_role_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {key: value for key, value in os.environ.items() if not key.startswith("ACE_SQL_")}
            env["ACE_SQL_SLIME_DATA"] = str(Path(directory) / "missing-data")
            stream = io.StringIO()
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", ["launch", "--dry-run"]), redirect_stdout(stream):
                launch.main()
            plan, _ = json.JSONDecoder().raw_decode(stream.getvalue())
        self.assertEqual(plan["steps"], 910)
        self.assertEqual(plan["drop_last_per_epoch"], 1)
        self.assertEqual(plan["trajectories_per_step"], 256)
        command = plan["command"]
        self.assertEqual(command[command.index("--n-samples-per-prompt") + 1], "16")
        self.assertEqual(command[command.index("--global-batch-size") + 1], "256")
        self.assertEqual(command[command.index("--num-rollout") + 1], "910")
        self.assertNotIn("--disable-rollout-global-dataset", command)
        self.assertFalse(any("pcgrad" in word.lower() for word in command))

    @unittest.skipUnless((SLIME_ROOT / "slime/utils/arguments.py").is_file(), "requires pinned slime checkout")
    def test_slime_flags_and_evaluation_configuration_match_upstream(self):
        command = launch.build_command(SLIME_ROOT, Path("missing-model"), Path("reference"), Path("data"),
                                       Path("output"), steps=910)
        tree = ast.parse((SLIME_ROOT / "slime/utils/arguments.py").read_text())
        declared = {node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")}
        owned_prefixes = ("--actor-", "--rollout-", "--custom-", "--sglang-", "--n-samples-", "--ace-")
        # SGLang flags are generated with an upstream prefix from ServerArgs.
        generated_sglang = {"--sglang-mem-fraction-static", "--sglang-attention-backend"}
        explicit = {"--hf-checkpoint", "--ref-load", "--prompt-data", "--input-key", "--label-key", "--metadata-key",
                    "--data-source-path", "--eval-function-path", "--loss-type", "--num-rollout", "--save-hf",
                    "--balance-data", "--colocate", "--advantage-estimator", "--use-kl-loss"}
        for flag in (item for item in command if item.startswith("--")):
            if flag.startswith(owned_prefixes) or flag in explicit:
                self.assertTrue(flag in declared or flag in generated_sglang, f"Unknown pinned slime option {flag}")
        self.assertTrue("--eval-prompt-data" in command or "--eval-config" in command,
                        "slime_validate_args requires an eval dataset when eval-interval is enabled")

    @unittest.skipUnless((SLIME_ROOT / "slime/utils/arguments.py").is_file()
                         and Path(os.environ.get("ACE_SQL_TEST_MEGATRON_ARGS", "/tmp/ace-sql-megatron-arguments.py")).is_file(),
                         "requires pinned slime and Megatron argument sources")
    def test_all_launcher_options_are_declared_by_pinned_upstreams(self):
        paths = [SLIME_ROOT / "slime/utils/arguments.py",
                 Path(os.environ.get("ACE_SQL_TEST_MEGATRON_ARGS", "/tmp/ace-sql-megatron-arguments.py"))]
        declared = set()
        for path in paths:
            tree = ast.parse(path.read_text())
            declared.update(node.value for node in ast.walk(tree)
                            if isinstance(node, ast.Constant) and isinstance(node.value, str)
                            and node.value.startswith("--"))
        # Slime forwards these registered SGLang ServerArgs under its own prefix.
        declared.update({"--sglang-mem-fraction-static", "--sglang-attention-backend"})
        command = launch.build_command(SLIME_ROOT, Path("missing-model"), Path("reference"), Path("data"),
                                       Path("output"), steps=910, resume=Path("checkpoint"))
        unknown = [item for item in command if item.startswith("--") and item not in declared]
        self.assertEqual(unknown, [])


if __name__ == "__main__":
    unittest.main()
