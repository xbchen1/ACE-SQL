"""CPU regression checks for migration semantics; no training framework needed."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ace_sql.rewards.engine import RewardEngine
from ace_sql.schema.rollout import generator_prompt, union_columns, role_group_id
from ace_sql.training.schedule import role_weights
from ace_sql.data.rl import convert, filter_overlong_prompts, question_key, row_to_record, validate_splits
from ace_sql.rewards.sql import compute_sql_reward
from ace_sql.schema.retrieval import build_generator_prompt


def retrieval(*columns):
    return "<think>choose columns</think>\n```\n[" + ", ".join(f"t.{c}" for c in columns) + "]\n```"


def generation(sql):
    return f"<think>query the table</think>\n```sql\n{sql}\n```"


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        database = self.directory / "toy.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b INTEGER, c TEXT)")
            connection.executemany("INSERT INTO t VALUES (?, ?, ?)", [(1, 1, "x"), (2, 2, "y")])
        self.schema = {
            "db_id": "toy", "table_names_original": ["t"],
            "column_names_original": [[-1, "*"], [0, "a"], [0, "b"], [0, "c"]],
            "column_names": [[-1, "*"], [0, "a"], [0, "b"], [0, "c"]],
            "column_types": ["text", "integer", "integer", "text"],
            "primary_keys": [1], "foreign_keys": [],
        }
        self.gt = {"db_id": "toy", "db_path": str(database), "sql": "SELECT a FROM t",
                   "question": "List the values", "gold_column_indices": [1]}
        self.extra = {"db_id": "toy", "question": self.gt["question"], "evidence": "The values are integers.",
                      "db_info": json.dumps(self.schema), "sampled_values": '{"t.b": [1, 2]}', "relavant_values": "{}"}
        self.row = {"prompt": [{"role": "system", "content": " Preserve spacing \n"},
                               {"role": "user", "content": "Original retriever prompt.\n"}],
                    "reward_model": {"style": "rule", "ground_truth": self.gt},
                    "extra_info": self.extra, "data_source": "toy_source"}
        self.record = row_to_record(self.row)
        self.key = question_key(self.gt)

    def engine(self, entries=None, **kwargs):
        engine = RewardEngine(max_workers=2, **kwargs)
        self.addCleanup(engine.close)
        engine.pool = {self.key: entries or {frozenset({1}): 2.0}}
        return engine

    def test_original_messages_and_metadata_are_preserved(self):
        self.assertEqual(self.record["prompt"], self.row["prompt"])
        self.assertEqual(self.record["label"], self.gt)
        self.assertEqual(self.record["metadata"]["ground_truth"], self.gt)
        self.assertEqual(self.record["metadata"]["extra_info"], self.extra)
        self.assertEqual(self.record["metadata"]["row_key"], self.key)
        self.assertEqual(row_to_record(json.loads(json.dumps(self.row))), self.record)
        null_evidence = deepcopy(self.row)
        null_evidence["extra_info"]["evidence"] = None
        converted = row_to_record(null_evidence)
        self.assertIsNone(converted["metadata"]["extra_info"]["evidence"])
        generator_prompt([retrieval("a")] * 8, converted["metadata"])

    def test_union_keeps_one_vote_without_primary_key_injection(self):
        responses = [retrieval("b", "b")] * 6 + [retrieval("c"), "invalid"]
        self.assertEqual(union_columns(responses, self.schema), {2, 3})
        messages, selected = generator_prompt(responses, self.record["metadata"])
        expected = build_generator_prompt(
            self.gt["question"], self.extra["evidence"], self.schema, {2, 3}, {"t.b": [1, 2]}, {})
        self.assertEqual(selected, {2, 3})
        self.assertEqual(messages, [{"role": "user", "content": expected}])
        self.assertNotIn("PRIMARY KEY", messages[0]["content"])
        empty, selected = generator_prompt(["invalid"] * 8, self.record["metadata"])
        self.assertEqual(selected, set())
        self.assertNotIn("CREATE TABLE", empty[0]["content"])

    def test_current_sql_updates_target_before_retriever_reward(self):
        engine = self.engine()
        ret, gen, details = engine.compute_group(
            [retrieval("a")] * 4 + [retrieval("b")] * 4,
            [generation("SELECT b FROM t")] * 8, self.record["metadata"],
            ret_resp_lens=[40] * 8, gen_resp_lens=[50] * 8)
        self.assertEqual(ret, [0.0] * 4 + [1.0] * 4)
        self.assertEqual(gen, [1.0] * 8)
        self.assertTrue(all(detail["exec_status"] == "Match" for detail in details))
        self.assertEqual(engine.pool[self.key], {frozenset({1}): 1.0, frozenset({2}): 8.0})
        ret, _, _ = engine.compute_group([retrieval("b")] * 8, [generation("SELECT b FROM t")] * 8,
                                          self.record["metadata"], update_pool=False)
        self.assertEqual(ret, [1.0] * 8)

    def test_tied_targets_and_validation_do_not_mutate_pool(self):
        engine = self.engine({frozenset({1}): 2.0, frozenset({2}): 2.0})
        before = engine.state_dict()
        ret, gen, _ = engine.compute_group(
            [retrieval("a"), retrieval("b")] * 4,
            [generation("SELECT b FROM t")] * 8, self.record["metadata"], update_pool=False)
        self.assertEqual(ret, [1.0] * 8)
        self.assertEqual(gen, [1.0] * 8)
        self.assertEqual(engine.state_dict(), before)

    def test_sql_execution_and_length_gate(self):
        score, base, detail = compute_sql_reward(generation("SELECT a FROM t"), self.gt, 2048)
        self.assertEqual((score, base, detail["exec_status"]), (0.5, 1.0, "Match"))
        for sql in ("SELECT a FROM t WHERE a=1", "SELECT missing FROM t"):
            self.assertEqual(compute_sql_reward(generation(sql), self.gt, 2048)[0], 0.0)
        self.assertEqual(compute_sql_reward("```sql SELECT a FROM t```", self.gt)[0], 0.0)
        engine = self.engine()
        ret, gen, details = engine.compute_group(
            [retrieval("a")] * 8, [generation("SELECT a FROM t")] * 8,
            self.record["metadata"], ret_resp_lens=[512, 1280, 2047, 2048] * 2,
            gen_resp_lens=[2048] * 8, update_pool=False)
        self.assertEqual(ret[0], 1.0)
        self.assertEqual(ret[1], 0.75)
        self.assertGreater(ret[2], 0.5)
        self.assertEqual(ret[3], 0.0)
        self.assertEqual(details[3]["ret_case"], "TRUNCATED")
        self.assertEqual(gen, [0.5] * 8)

    def test_group_boundaries_and_complementary_weight_schedule(self):
        self.assertEqual(role_weights(0, 100), (1.0, 0.0))
        self.assertEqual(role_weights(10, 100), (0.8, 0.2))
        self.assertEqual(role_weights(25, 100), (0.5, 0.5))
        self.assertEqual(role_weights(100, 100), (0.5, 0.5))
        self.assertNotEqual(role_group_id(self.key, "retriever"), role_group_id(self.key, "generator"))
        with self.assertRaises(ValueError):
            union_columns([retrieval("a")] * 7, self.schema)
        with self.assertRaises(ValueError):
            self.engine().compute_group([retrieval("a")] * 8, [generation("SELECT a FROM t")] * 7,
                                        self.record["metadata"])
        with self.assertRaises(ValueError):
            role_weights(-1, 100)

    def test_checkpoint_roundtrip(self):
        engine = self.engine()
        state = json.loads(json.dumps(engine.state_dict()))
        restored = self.engine({frozenset({2}): 1.0})
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        pool_path = self.directory / "checkpoint" / "pool.json"
        restored.save_pool(pool_path)
        self.assertEqual(json.loads(pool_path.read_text()), state["pool"])
        bad = {**state, "pool_gamma": 1.0}
        with self.assertRaises(ValueError):
            restored.load_state_dict(bad)

    def test_zero_decay_uses_current_matches_and_validation_remains_read_only(self):
        engine = self.engine(pool_gamma=0)
        ret, _, _ = engine.compute_group(
            [retrieval("b")] * 8, [generation("SELECT b FROM t")] * 8,
            self.record["metadata"])
        self.assertEqual(ret, [1.0] * 8)
        self.assertEqual(engine.pool[self.key], {frozenset({2}): 8.0})
        before = engine.state_dict()
        ret, _, _ = engine.compute_group(
            [retrieval("b")], [generation("SELECT b FROM t")],
            self.record["metadata"], update_pool=False)
        self.assertEqual(ret, [1.0])
        self.assertEqual(engine.state_dict(), before)
        engine.compute_group([retrieval("b")] * 8, [generation("SELECT c FROM t")] * 8,
                             self.record["metadata"])
        self.assertEqual(engine.pool[self.key], {})
        # An empty gamma=0 pool is legal; a later correct rollout repopulates it.
        ret, _, _ = engine.compute_group([retrieval("b")] * 8, [generation("SELECT b FROM t")] * 8,
                                         self.record["metadata"])
        self.assertEqual(ret, [1.0] * 8)

    def test_frozen_and_gold_controls_do_not_update_pool(self):
        for mode, expected in (("frozen", [0.0] * 4 + [1.0] * 4),
                               ("gold", [1.0] * 4 + [0.0] * 4)):
            with self.subTest(mode=mode):
                engine = self.engine({frozenset({2}): 2.0}, target_mode=mode)
                before = engine.state_dict()
                ret, _, _ = engine.compute_group(
                    [retrieval("a")] * 4 + [retrieval("b")] * 4,
                    [generation("SELECT a FROM t")] * 8, self.record["metadata"])
                self.assertEqual(ret, expected)
                self.assertEqual(engine.state_dict(), before)

    def test_dense_reward_matches_paper_coverage_and_noise(self):
        engine = self.engine({frozenset({1, 2}): 2.0}, target_mode="dense")
        ret, _, _ = engine.compute_group(
            [retrieval("a"), retrieval("a", "b"), retrieval("a", "b", "c"), retrieval("c")] * 2,
            [generation("SELECT a FROM t")] * 8, self.record["metadata"], update_pool=False)
        self.assertEqual(ret, [0.5, 1.0, 1.0 - 0.5 / 3, -0.5] * 2)

    def test_no_length_penalty_and_no_schedule_controls(self):
        engine = self.engine(length_penalty=False)
        ret, gen, _ = engine.compute_group(
            [retrieval("a")] * 8, [generation("SELECT a FROM t")] * 8,
            self.record["metadata"], ret_resp_lens=[1500] * 8, gen_resp_lens=[2048] * 8)
        self.assertEqual(ret, [1.0] * 8)
        self.assertEqual(gen, [1.0] * 8)
        self.assertEqual(role_weights(0, 100, ramp_ratio=0), (0.5, 0.5))

    def test_training_execution_preserves_row_order_and_duplicates(self):
        for sql in ("SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t UNION ALL SELECT a FROM t"):
            with self.subTest(sql=sql):
                self.assertEqual(compute_sql_reward(generation(sql), self.gt)[2]["exec_status"], "Mismatch")

    def test_old_or_different_method_checkpoints_are_rejected(self):
        engine = self.engine()
        state = engine.state_dict()
        for bad in ({**state, "version": 1}, {**state, "method_version": "old"},
                    {**state, "target_mode": "frozen"}, {**state, "length_penalty": False}):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                engine.load_state_dict(bad)

    def validation_row(self):
        row = deepcopy(self.row)
        row["reward_model"]["ground_truth"]["question"] = "Another question"
        row["extra_info"]["question"] = "Another question"
        return row

    def test_split_and_pool_preflight(self):
        validation = row_to_record(self.validation_row())
        pool = {self.key: {"1": 2.0}, validation["metadata"]["row_key"]: {"2": 3.0}}
        result = validate_splits([self.record], [validation], pool)
        self.assertEqual(result["train_rows"], 1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits([self.record], [self.record], pool)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_splits([self.record], [validation], {self.key: {"1": 2.0}})
        with self.assertRaisesRegex(ValueError, "out-of-schema"):
            validate_splits([self.record], [validation], {**pool, self.key: {"99": 1.0}})
        broken = deepcopy(self.row)
        del broken["extra_info"]["db_info"]
        with self.assertRaisesRegex(ValueError, "db_info"):
            row_to_record(broken)

    def test_parquet_to_jsonl_roundtrip(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        train_path, val_path = self.directory / "train.parquet", self.directory / "val.parquet"
        parquet.write_table(arrow.Table.from_pylist([self.row]), train_path)
        val_row = self.validation_row()
        parquet.write_table(arrow.Table.from_pylist([val_row]), val_path)
        pool_path = self.directory / "pool.json"
        pool_path.write_text(json.dumps({self.key: {"1": 2}, question_key(val_row["reward_model"]["ground_truth"]): {"2": 2}}))
        output = self.directory / "output"
        manifest = convert(train_path, val_path, pool_path, output)
        self.assertEqual(json.loads((output / "train.jsonl").read_text()), self.record)
        self.assertEqual(manifest["validation_rows"], 1)
        self.assertEqual(len(manifest["source_sha256"]["train"]), 64)
        self.assertEqual(json.loads((output / "initial_pool.json").read_text()), json.loads(pool_path.read_text()))

    def test_filter_uses_template_and_exact_boundary_without_changing_rows(self):
        kept = deepcopy(self.record)
        kept["prompt"] = [{"role": "user", "content": "boundary"}]
        dropped = row_to_record(self.validation_row())
        dropped["prompt"] = [{"role": "user", "content": "overlong"}]
        records = [kept, dropped]
        before = deepcopy(records)

        class FakeTokenizer:
            def apply_chat_template(inner, messages, *, tokenize, add_generation_prompt):
                self.assertFalse(tokenize)
                self.assertTrue(add_generation_prompt)
                return messages[0]["content"] + "::assistant<think>"

            def encode(inner, prompt, *, add_special_tokens):
                self.assertFalse(add_special_tokens)
                self.assertTrue(prompt.endswith("::assistant<think>"))
                return [1] * (4096 if prompt.startswith("boundary") else 4097)

        filtered, report = filter_overlong_prompts(records, FakeTokenizer())
        self.assertEqual(records, before)
        self.assertEqual(filtered, [kept])
        self.assertIs(filtered[0], kept)
        self.assertEqual(report, {"input_rows": 2, "kept_rows": 1, "dropped_rows": 1,
                                  "kept_keys": [kept["metadata"]["row_key"]],
                                  "dropped_keys": [dropped["metadata"]["row_key"]]})

    def test_filtered_conversion_reports_both_splits_and_rejects_empty_result(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        def make_row(question, marker):
            row = deepcopy(self.row)
            row["reward_model"]["ground_truth"]["question"] = question
            row["extra_info"]["question"] = question
            row["prompt"] = [{"role": "user", "content": marker}]
            return row

        train_rows = [make_row("train kept", "short"), make_row("train dropped", "long")]
        val_rows = [make_row("validation kept", "short"), make_row("validation dropped", "long")]
        train_path, val_path = self.directory / "filter_train.parquet", self.directory / "filter_val.parquet"
        parquet.write_table(arrow.Table.from_pylist(train_rows), train_path)
        parquet.write_table(arrow.Table.from_pylist(val_rows), val_path)
        pool = {question_key(row["reward_model"]["ground_truth"]): {"1": 2.0} for row in train_rows + val_rows}
        pool_path = self.directory / "filter_pool.json"
        pool_path.write_text(json.dumps(pool))

        class FakeTokenizer:
            name_or_path = "fake-checkpoint"

            def apply_chat_template(inner, messages, **kwargs):
                return messages[0]["content"]

            def encode(inner, text, **kwargs):
                return [1] * (3 if text == "short" else 4097)

        output = self.directory / "filtered"
        manifest = convert(train_path, val_path, pool_path, output, tokenizer=FakeTokenizer())
        self.assertEqual((manifest["train_rows"], manifest["validation_rows"], manifest["pool_keys"]), (1, 1, 4))
        self.assertEqual(manifest["prompt_filter"]["tokenizer"], "fake-checkpoint")
        self.assertEqual(manifest["prompt_filter"]["max_prompt_len"], 4096)
        for split, rows in (("train", train_rows), ("validation", val_rows)):
            report = manifest["prompt_filter"][split]
            self.assertEqual((report["input_rows"], report["kept_rows"], report["dropped_rows"]), (2, 1, 1))
            self.assertEqual(report["kept_keys"], [question_key(rows[0]["reward_model"]["ground_truth"])])
            self.assertEqual(report["dropped_keys"], [question_key(rows[1]["reward_model"]["ground_truth"])])
            self.assertEqual(json.loads((output / f"{split}.jsonl").read_text()), row_to_record(rows[0]))
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        self.assertEqual(json.loads((output / "initial_pool.json").read_text()), pool)
        empty_output = self.directory / "empty-filter"
        with self.assertRaisesRegex(ValueError, "removed all.*train"):
            convert(train_path, val_path, pool_path, empty_output, tokenizer=FakeTokenizer(), max_prompt_len=2)
        self.assertFalse(empty_output.exists())
        parquet.write_table(arrow.Table.from_pylist([val_rows[1]]), val_path)
        with self.assertRaisesRegex(ValueError, "removed all.*validation"):
            convert(train_path, val_path, pool_path, empty_output, tokenizer=FakeTokenizer())
        self.assertFalse(empty_output.exists())


if __name__ == "__main__":
    unittest.main()
