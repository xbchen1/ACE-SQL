"""Lossless conversion of the published ACE-SQL parquet data for slime.

Run ``ace-sql data --help`` after installing the package.  The JSONL
columns map to slime's prompt, label, and metadata keys without importing a
training framework. Nested JSON strings in extra_info are deliberately kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def as_object(value: Any, field: str) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _plain(value: Any) -> Any:
    """Handle Arrow/pandas scalar containers without depending on numpy."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    return value


def question_key(ground_truth: dict) -> str:
    for field in ("db_id", "question"):
        if not isinstance(ground_truth.get(field), str) or not ground_truth[field].strip():
            raise ValueError(f"ground_truth.{field} must be a nonempty string")
    return f"{ground_truth['db_id']}||{ground_truth['question']}"


def validate_metadata(metadata: dict) -> tuple[dict, dict]:
    """Validate fields used by both rollouts and reward computation."""
    metadata = as_object(metadata, "metadata")
    gt = as_object(metadata.get("ground_truth"), "ground_truth")
    extra = as_object(metadata.get("extra_info"), "extra_info")
    key = question_key(gt)
    if metadata.get("row_key", key) != key:
        raise ValueError("metadata.row_key does not match db_id and question")
    for field in ("sql", "db_path"):
        if not isinstance(gt.get(field), str) or not gt[field].strip():
            raise ValueError(f"ground_truth.{field} must be a nonempty string")
    gold = gt.get("gold_column_indices")
    if not isinstance(gold, list) or any(type(col) is not int for col in gold):
        raise ValueError("ground_truth.gold_column_indices must be a list of integers")
    for field in ("db_id", "question"):
        if extra.get(field) != gt[field]:
            raise ValueError(f"extra_info.{field} must match ground_truth.{field}")
    if extra.get("evidence") is not None and not isinstance(extra["evidence"], str):
        raise ValueError("extra_info.evidence must be a string or null")
    schema = as_object(extra.get("db_info"), "extra_info.db_info")
    for field in ("table_names_original", "column_names_original", "column_names", "column_types", "primary_keys", "foreign_keys"):
        if not isinstance(schema.get(field), list):
            raise ValueError(f"extra_info.db_info.{field} must be a list")
    columns = schema["column_names_original"]
    if not schema["table_names_original"] or not columns:
        raise ValueError("database schema must contain tables and columns")
    if len(columns) != len(schema["column_names"]) or len(columns) != len(schema["column_types"]):
        raise ValueError("database schema column arrays have different lengths")
    for column in columns:
        if (not isinstance(column, list) or len(column) != 2
                or type(column[0]) is not int or not isinstance(column[1], str)
                or not -1 <= column[0] < len(schema["table_names_original"])):
            raise ValueError("invalid database schema column")
    if any(col < 0 or col >= len(columns) for col in gold):
        raise ValueError("gold_column_indices contain out-of-schema columns")
    for field in ("sampled_values", "relavant_values"):
        as_object(extra.get(field), f"extra_info.{field}")
    return gt, extra


def row_to_record(row: Mapping[str, Any]) -> dict:
    """Return a JSONL record directly consumable by slime's Dataset/Sample."""
    row = _plain(row)
    reward_model = as_object(row.get("reward_model"), "reward_model")
    gt = as_object(reward_model.get("ground_truth"), "reward_model.ground_truth")
    extra = as_object(row.get("extra_info"), "extra_info")
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise ValueError("prompt must contain the original chat message list") from exc
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a nonempty chat message list")
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("each prompt message must contain string content")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid prompt message role")
    metadata = {"row_key": question_key(gt), "ground_truth": gt, "extra_info": extra,
                "reward_model": reward_model}
    for field in ("data_source", "ability"):
        if field in row:
            metadata[field] = row[field]
    validate_metadata(metadata)
    record = {"prompt": prompt, "label": gt, "metadata": metadata}
    # Fail here, rather than emitting JSON NaN or nonportable scalar values.
    json.dumps(record, ensure_ascii=False, allow_nan=False)
    return record


def read_parquet_records(path: str | Path) -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Parquet conversion requires pyarrow; install the slime requirements") from exc
    records = []
    for index, row in enumerate(parquet.read_table(path).to_pylist()):
        try:
            records.append(row_to_record(row))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: row {index}: {exc}") from exc
    return records


def validate_splits(train: list[dict], validation: list[dict], pool: dict) -> dict:
    """Check disjoint question keys and complete, valid initial-pool coverage."""
    pool = as_object(pool, "initial_pool")
    split_keys = []
    for name, records in (("train", train), ("validation", validation)):
        if not records:
            raise ValueError(f"{name} must not be empty")
        keys = []
        for record in records:
            gt, extra = validate_metadata(record["metadata"])
            key = question_key(gt)
            keys.append(key)
            if key not in pool:
                raise ValueError(f"{name} question missing from initial_pool: {key}")
            entries = as_object(pool[key], f"initial_pool[{key}]")
            if not entries:
                raise ValueError(f"initial_pool has no target for {key}")
            schema = as_object(extra["db_info"], "db_info")
            allowed = {idx for idx, (table, _) in enumerate(schema["column_names_original"]) if table >= 0}
            positive = False
            for columns, frequency in entries.items():
                try:
                    indices = {int(index) for index in columns.split(",") if index}
                    freq = float(frequency)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid initial_pool entry for {key}") from exc
                if not indices or not indices <= allowed:
                    raise ValueError(f"initial_pool contains empty or out-of-schema columns for {key}")
                if not math.isfinite(freq) or freq < 0:
                    raise ValueError(f"initial_pool frequency must be finite and nonnegative for {key}")
                positive |= freq > 0
            if not positive:
                raise ValueError(f"initial_pool has no positive target for {key}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate question keys in {name}")
        split_keys.append(set(keys))
    overlap = split_keys[0] & split_keys[1]
    if overlap:
        raise ValueError(f"train/validation question overlap: {sorted(overlap)[0]}")
    return {"train_rows": len(train), "validation_rows": len(validation), "pool_keys": len(pool)}


def atomic_write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def filter_overlong_prompts(records: list[dict], tokenizer: Any,
                            max_prompt_len: int = 4096) -> tuple[list[dict], dict]:
    """Filter complete retriever chats using the exact rollout tokenization.

    The released data recipe applies its prompt-length filter to train and validation
    datasets. Keep the original row keys and messages; never truncate a schema.
    """
    if type(max_prompt_len) is not int or max_prompt_len <= 0:
        raise ValueError("max_prompt_len must be a positive integer")
    retained, kept_keys, dropped_keys = [], [], []
    for record in records:
        prompt = tokenizer.apply_chat_template(
            record["prompt"], tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        key = record["metadata"]["row_key"]
        if len(token_ids) <= max_prompt_len:
            retained.append(record)
            kept_keys.append(key)
        else:
            dropped_keys.append(key)
    return retained, {"input_rows": len(records), "kept_rows": len(retained),
                      "dropped_rows": len(dropped_keys),
                      "kept_keys": kept_keys, "dropped_keys": dropped_keys}


def convert(train_path: str | Path, validation_path: str | Path, initial_pool_path: str | Path,
            output_dir: str | Path, *, tokenizer: Any = None,
            max_prompt_len: int = 4096) -> dict:
    train = read_parquet_records(train_path)
    validation = read_parquet_records(validation_path)
    pool = json.loads(Path(initial_pool_path).read_text(encoding="utf-8"))
    # Validate source provenance before filtering so malformed rows or split
    # leakage cannot be hidden by dropping an overlong question.
    manifest = validate_splits(train, validation, pool)
    filter_report = {"enabled": tokenizer is not None}
    if tokenizer is not None:
        if isinstance(tokenizer, (str, Path)):
            tokenizer_name = str(tokenizer)
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("--tokenizer requires transformers") from exc
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
        else:
            tokenizer_name = str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__))
        train, train_report = filter_overlong_prompts(train, tokenizer, max_prompt_len)
        validation, validation_report = filter_overlong_prompts(validation, tokenizer, max_prompt_len)
        for split, report in (("train", train_report), ("validation", validation_report)):
            if report["kept_rows"] == 0:
                raise ValueError(f"Prompt length filtering removed all {report['input_rows']} {split} rows; no output written")
        manifest = validate_splits(train, validation, pool)
        filter_report.update({"tokenizer": tokenizer_name, "max_prompt_len": max_prompt_len,
                              "train": train_report, "validation": validation_report})
    manifest["prompt_filter"] = filter_report
    manifest["format_version"] = 1
    manifest["source_sha256"] = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (("train", train_path), ("validation", validation_path), ("initial_pool", initial_pool_path))
    }
    output_dir = Path(output_dir)
    for name, records in (("train", train), ("validation", validation)):
        atomic_write_text(output_dir / f"{name}.jsonl", "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records))
    atomic_write_text(output_dir / "initial_pool.json", json.dumps(pool, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--initial-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", help="Optional HF tokenizer path/name; filters overlong retriever prompts in both splits")
    parser.add_argument("--max-prompt-len", type=int, default=4096,
                        help="Keep tokenized retriever prompts of at most this length (default: 4096)")
    args = parser.parse_args()
    manifest = convert(args.train, args.validation, args.initial_pool, args.output_dir,
                       tokenizer=args.tokenizer, max_prompt_len=args.max_prompt_len)
    # Full kept/dropped row keys are recorded in manifest.json, not the console.
    summary = {**manifest, "prompt_filter": dict(manifest["prompt_filter"])}
    for split in ("train", "validation"):
        if split in summary["prompt_filter"]:
            summary["prompt_filter"][split] = {
                key: value for key, value in summary["prompt_filter"][split].items()
                if key not in {"kept_keys", "dropped_keys"}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
