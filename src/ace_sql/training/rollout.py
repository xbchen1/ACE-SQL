"""Synchronous two-pass rollout and durable empirical-pool data source.

The returned 16 trajectories are two independent groups of eight, not a
16-sample GRPO group. All inference uses the same synchronized SGLang policy.
"""

import asyncio
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

from ace_sql.rewards.engine import METHOD_VERSION, RewardEngine
from ace_sql.schema.rollout import generator_prompt
from ace_sql.training.schedule import role_weights


def read_records(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ACEDataSource:
    """Checkpoint pool, cursor, ordering, and SQL provenance in one file.

    slime calls save after the corresponding optimizer checkpoint. Resume
    selects the exact rollout id reported by the loaded model, never the newest
    pool file independently. A missing or mismatched state is a hard error.
    """

    def __init__(self, args):
        self.args = args
        self.records = read_records(args.prompt_data)
        self.validation = read_records(args.ace_validation_data)
        if not self.validation:
            raise ValueError("Validation data must not be empty")
        self.group_size = args.ace_group_size
        if self.group_size != 8 or args.n_samples_per_prompt != 16:
            raise ValueError("ACE requires 8 samples per role, 16 total trajectories per question")
        if args.global_batch_size != args.rollout_batch_size * 16:
            raise ValueError("ACE requires one optimizer update per complete two-pass rollout")
        if len(self.records) < args.rollout_batch_size:
            raise ValueError("Training data is smaller than one question batch")
        self.reward_engine = RewardEngine(args.ace_initial_pool, group_size=8,
                                         pool_gamma=args.ace_pool_gamma,
                                         target_mode=getattr(args, "ace_target_mode", "online"),
                                         length_penalty=getattr(args, "ace_length_penalty", True))
        self.epoch = self.offset = self.group_index = self.sample_index = 0
        self.last_rollout_id = -1
        self.provenance = {}
        self.order = self._order(0)
        self.fingerprint = self._fingerprint()
        self.tokenizer = None

    def _fingerprint(self):
        payload = {"method_version": METHOD_VERSION, "train": self.records, "validation": self.validation,
                   "target_mode": self.reward_engine.target_mode,
                   "length_penalty": self.reward_engine.length_penalty,
                   "initial_pool": json.loads(Path(self.args.ace_initial_pool).read_text()),
                   "batch": self.args.rollout_batch_size, "seed": self.args.rollout_seed,
                   "group_size": self.group_size, "num_rollout": self.args.num_rollout,
                   "pool_gamma": self.args.ace_pool_gamma, "ramp": self.args.ace_ramp_ratio}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _order(self, epoch):
        order = list(range(len(self.records)))
        random.Random(self.args.rollout_seed + epoch).shuffle(order)
        return order

    def __len__(self):
        return len(self.records)

    def get_samples(self, num_samples):
        from slime.utils.types import Sample

        if num_samples != self.args.rollout_batch_size:
            raise ValueError("ACE does not support oversampling or partial question batches")
        # Match the released train loader's drop_last semantics, reshuffle each epoch.
        if self.offset + num_samples > len(self.records):
            self.epoch += 1
            self.offset = 0
            self.order = self._order(self.epoch)
        groups = []
        for row_id in self.order[self.offset:self.offset + num_samples]:
            record = self.records[row_id]
            group = []
            for _ in range(self.group_size):
                group.append(Sample(group_index=self.group_index, index=self.sample_index,
                                    prompt=copy.deepcopy(record["prompt"]),
                                    label=record["label"], metadata=copy.deepcopy(record["metadata"])))
                self.sample_index += 1
            self.group_index += 1
            groups.append(group)
        self.offset += num_samples
        return groups

    def add_samples(self, samples):
        raise RuntimeError("ACE uses synchronous complete groups; rollout buffering is disabled")

    def save(self, rollout_id):
        if rollout_id != self.last_rollout_id:
            raise ValueError("Pool checkpoint id does not match the completed rollout")
        atomic_json(Path(self.args.save) / "rollout" / f"ace_state_{rollout_id}.json", {
            "version": 1, "rollout_id": rollout_id, "fingerprint": self.fingerprint,
            "epoch": self.epoch, "offset": self.offset, "group_index": self.group_index,
            "sample_index": self.sample_index, "pool": self.reward_engine.state_dict(),
            "provenance": self.provenance,
        })

    def load(self, rollout_id=None):
        if rollout_id is None or rollout_id < 0:
            return
        if not self.args.load:
            raise ValueError("Resume requires a model checkpoint directory")
        path = Path(self.args.load) / "rollout" / f"ace_state_{rollout_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Refusing model-only resume: missing empirical pool/cursor {path}")
        state = json.loads(path.read_text())
        if state["version"] != 1 or state["rollout_id"] != rollout_id or state["fingerprint"] != self.fingerprint:
            raise ValueError("ACE checkpoint differs in data, schedule, batch size, seed, or rollout id")
        self.reward_engine.load_state_dict(state["pool"])
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.group_index, self.sample_index = state["group_index"], state["sample_index"]
        self.provenance = state["provenance"]
        self.order = self._order(self.epoch)
        self.last_rollout_id = rollout_id


def parser_response(sample):
    # Qwen3 normally places the opening think marker in the prompt. Repair only
    # the parser view; response_length, tokens, and trainable logprobs stay exact.
    if sample.metadata.get("ace_prompt_open_think") and not sample.response.lstrip().startswith("<think>"):
        return "<think>" + sample.response
    return sample.response


async def _generate(args, sample, tokenizer, semaphore, evaluation=False, role="retriever"):
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    prompt = tokenizer.apply_chat_template(sample.prompt, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if role == "retriever" and len(ids) > args.rollout_max_prompt_len:
        raise ValueError(
            f"Retriever prompt exceeds {args.rollout_max_prompt_len} tokens for "
            f"{sample.metadata.get('row_key')}; filter with the actual checkpoint tokenizer "
            "before training. Silent retriever truncation changes the released method."
        )
    ids = ids[-args.rollout_max_prompt_len:]
    # Inspect only the assistant prefix, not think tags occurring in few-shot examples.
    sample.metadata["ace_prompt_open_think"] = prompt.rstrip().endswith("<think>")
    params = {"temperature": 0.0 if evaluation else args.rollout_temperature,
              "top_p": 1.0 if evaluation else args.rollout_top_p,
              "top_k": -1 if evaluation else args.rollout_top_k,
              "max_new_tokens": args.rollout_max_response_len,
              "skip_special_tokens": True, "no_stop_trim": True}
    if not evaluation:
        params["sampling_seed"] = (args.rollout_seed + sample.index) % (2 ** 31)
    if tokenizer.eos_token_id is not None:
        params["stop_token_ids"] = [tokenizer.eos_token_id]
    payload = {"input_ids": ids, "sampling_params": params, "return_logprob": True}
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with semaphore:
        output = await post(url, payload)
    meta = output["meta_info"]
    logprobs = meta.get("output_token_logprobs")
    if not logprobs:
        raise RuntimeError("SGLang returned no sampled token IDs/logprobs; refusing retokenization")
    generated = [int(item[1]) for item in logprobs]
    probabilities = [float(item[0]) for item in logprobs]
    if not all(math.isfinite(value) for value in probabilities):
        raise ValueError("SGLang returned non-finite rollout logprobs")
    if meta.get("completion_tokens", len(generated)) != len(generated):
        raise ValueError("SGLang completion token count disagrees with returned logprobs")
    sample.tokens = ids + generated
    sample.response = output["text"]
    sample.response_length = len(generated)
    sample.rollout_log_probs = probabilities
    sample.loss_mask = [1] * len(generated)
    finish = meta.get("finish_reason", {})
    reason = finish.get("type") if isinstance(finish, dict) else finish
    if reason not in {"stop", "length"}:
        raise RuntimeError(f"SGLang generation did not complete: {finish}")
    sample.status = Sample.Status.TRUNCATED if reason == "length" else Sample.Status.COMPLETED
    return sample


async def _gather(coroutines):
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _rollout(args, rollout_id, source, evaluation):
    if evaluation:
        return await _rollout_impl(args, rollout_id, source, evaluation)
    cursor_fields = ("epoch", "offset", "group_index", "sample_index", "last_rollout_id", "order")
    cursor = {key: copy.deepcopy(getattr(source, key)) for key in cursor_fields}
    pool, provenance = source.reward_engine.state_dict(), copy.deepcopy(source.provenance)
    try:
        return await _rollout_impl(args, rollout_id, source, evaluation)
    except BaseException:
        for key, value in cursor.items():
            setattr(source, key, value)
        source.reward_engine.load_state_dict(pool)
        source.provenance = provenance
        raise


async def _rollout_impl(args, rollout_id, source, evaluation):
    from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
    from slime.utils.types import Sample
    from transformers import AutoTokenizer

    if source.tokenizer is None:
        source.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=False)
    semaphore = asyncio.Semaphore(args.ace_concurrency)
    if evaluation:
        groups = [[Sample(group_index=i, index=i, prompt=copy.deepcopy(row["prompt"]),
                          label=row["label"], metadata=copy.deepcopy(row["metadata"]))]
                  for i, row in enumerate(source.validation)]
    else:
        if rollout_id != source.last_rollout_id + 1:
            raise ValueError("Non-sequential rollout id; load the matching pool state before resuming")
        groups = source.get_samples(args.rollout_batch_size)
    flat_ret = [sample for group in groups for sample in group]
    await _gather([_generate(args, sample, source.tokenizer, semaphore, evaluation) for sample in flat_ret])
    generator_groups = []
    for group in groups:
        messages, _ = generator_prompt([parser_response(s) for s in group], group[0].metadata,
                                       group_size=len(group))
        gen_group = []
        for i in range(len(group)):
            index = i if evaluation else source.sample_index
            if not evaluation:
                source.sample_index += 1
            gen_group.append(Sample(group_index=group[0].group_index, index=index,
                                    prompt=copy.deepcopy(messages), label=group[0].label,
                                    metadata=copy.deepcopy(group[0].metadata)))
        generator_groups.append(gen_group)
    await _gather([_generate(args, s, source.tokenizer, semaphore, evaluation, role="generator")
                           for group in generator_groups for s in group])
    weights = role_weights(rollout_id, args.num_rollout, args.ace_ramp_ratio)
    train_groups, exec_scores, details_log = [], [], []
    for ret, gen in zip(groups, generator_groups, strict=True):
        # Validation is one greedy route per question, pool strictly read-only.
        engine = source.reward_engine
        rewards = engine.compute_group(
            [parser_response(s) for s in ret], [parser_response(s) for s in gen], ret[0].metadata,
            ret_resp_lens=[s.response_length for s in ret], gen_resp_lens=[s.response_length for s in gen],
            update_pool=not evaluation,
        )
        rr, gr, details = rewards
        for role, samples, values, weight in [("retriever", ret, rr, weights[0]), ("generator", gen, gr, weights[1])]:
            for sample, value in zip(samples, values, strict=True):
                sample.reward = float(value)
                sample.train_metadata = {"role": role, "loss_weight": weight,
                                         "question_key": sample.metadata["row_key"]}
            train_groups.append(samples)
        exec_scores.extend(float(d["exec_status"] == "Match") for d in details)
        key = ret[0].metadata["row_key"]
        for detail in details:
            details_log.append({"question_key": key, **detail})
            if not evaluation and detail["exec_status"] == "Match":
                sql = detail["pred_sql"]
                source.provenance.setdefault(key, {})[sql] = source.provenance.get(key, {}).get(sql, 0) + 1
    if evaluation:
        return RolloutFnEvalOutput(data={"ace_sql_greedy_ex": {"rewards": exec_scores,
                                    "samples": [s for group in generator_groups for s in group]}})
    source.last_rollout_id = rollout_id
    atomic_json(Path(args.save) / "ace_audit" / f"rollout_{rollout_id}.json", details_log)
    return RolloutFnTrainOutput(samples=train_groups, metrics={"ace/generator_ex": sum(exec_scores)/len(exec_scores),
                                "ace/retriever_weight": weights[0], "ace/generator_weight": weights[1]})


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run
    return run(_rollout(args, rollout_id, data_source, evaluation))
