"""Build and run the pinned slime command; --dry-run requires no GPU packages."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from dataclasses import asdict
from .config import TrainConfig, load_config


def build_command(slime, model, reference, data, output, *, steps, questions=16, gpus=8, resume=None, settings=None):
    settings = settings or TrainConfig()
    config_path = model / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {
        "model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096 if settings.model_size == "8B" else 2560,
        "intermediate_size": 12288 if settings.model_size == "8B" else 9728, "num_attention_heads": 32, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000, "vocab_size": 151936, "head_dim": 128,
    }
    if config["model_type"] != "qwen3":
        raise ValueError("This recipe supports dense Qwen3-4B/8B checkpoints")
    command = [sys.executable, str(slime / "train.py")]
    values = {
        "actor-num-nodes": 1, "actor-num-gpus-per-node": gpus, "num-gpus-per-node": gpus,
        "hf-checkpoint": model, "ref-load": reference,
        "save": output, "save-interval": settings.save_interval, "eval-interval": settings.eval_interval,
        "save-hf": output / "hf" / "rollout_{rollout_id}",
        "prompt-data": data / "train.jsonl", "input-key": "prompt", "label-key": "label",
        "metadata-key": "metadata", "data-source-path": "ace_sql.training.rollout.ACEDataSource",
        "rollout-function-path": "ace_sql.training.rollout.generate_rollout",
        "eval-function-path": "ace_sql.training.rollout.generate_rollout",
        "custom-reward-post-process-path": "ace_sql.training.loss.reward_postprocess",
        "loss-type": "custom_loss", "custom-loss-function-path": "ace_sql.training.loss.policy_loss",
        "custom-config-path": output / "ace_config.json",
        "num-rollout": steps, "rollout-batch-size": questions, "n-samples-per-prompt": 16,
        "global-batch-size": questions * 16, "micro-batch-size": 1,
        "rollout-max-prompt-len": 4096, "rollout-max-response-len": 2048,
        "rollout-max-context-len": 6144, "seq-length": 6144, "rollout-temperature": settings.temperature,
        "rollout-top-p": 1, "rollout-top-k": -1, "rollout-seed": settings.seed,
        "seed": settings.seed,
        "n-samples-per-eval-prompt": 1,
        "tensor-model-parallel-size": 2, "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1, "rollout-num-gpus-per-engine": 2,
        "sglang-mem-fraction-static": 0.65, "sglang-attention-backend": "flashinfer",
        "advantage-estimator": "grpo", "kl-coef": 0, "kl-loss-coef": settings.kl_coefficient,
        "kl-loss-type": "low_var_kl", "entropy-coef": 0,
        "eps-clip": settings.clip, "eps-clip-high": settings.clip,
        "optimizer": "adam", "lr": settings.learning_rate, "lr-decay-style": "constant",
        "lr-warmup-fraction": settings.warmup_fraction, "weight-decay": 0, "clip-grad": 1,
        "adam-beta1": 0.9, "adam-beta2": 0.999,
        "recompute-granularity": "full", "recompute-method": "uniform", "recompute-num-layers": 1,
        "attention-dropout": 0, "hidden-dropout": 0, "attention-backend": "flash",
        "num-layers": config["num_hidden_layers"], "hidden-size": config["hidden_size"],
        "ffn-hidden-size": config["intermediate_size"], "num-attention-heads": config["num_attention_heads"],
        "num-query-groups": config["num_key_value_heads"], "normalization": "RMSNorm",
        "norm-epsilon": config["rms_norm_eps"], "rotary-base": config["rope_theta"],
        "vocab-size": config["vocab_size"], "kv-channels": config.get("head_dim", 128),
    }
    if resume is not None:
        values["load"] = resume
    # With no --load, pinned slime initializes from --ref-load and sets
    # finetune/no-load-optim/no-load-rng/start-rollout-id=0. An explicit fresh
    # --load would mistake the conversion's release checkpoint for RL resume.
    for name, value in values.items():
        command.extend(["--" + name, str(value)])
    command.extend(["--eval-prompt-data", "ace_sql_greedy_ex", str(data / "validation.jsonl")])
    command += ["--" + flag for flag in [
        "colocate", "bf16", "use-kl-loss", "balance-data", "sequence-parallel",
        "use-distributed-optimizer", "accumulate-allreduce-grads-in-fp32", "attention-softmax-in-fp32",
        "swiglu", "group-query-attention", "use-rotary-position-embeddings", "disable-bias-linear",
        "qk-layernorm", "untie-embeddings-and-output-weights",
    ]]
    return command


def build_plan(settings):
    settings.validate()
    slime, model, reference, data, output = (Path(getattr(settings, k)) for k in ('slime', 'model', 'reference', 'data', 'output'))
    train = data / 'train.jsonl'
    rows = sum(1 for line in train.open() if line.strip()) if train.is_file() else 2913
    steps = (rows // settings.questions) * settings.epochs
    if steps < 1:
        raise ValueError('No complete training batches')
    custom = {'ace_initial_pool': str(data / 'initial_pool.json'),
              'ace_validation_data': str(data / 'validation.jsonl'), 'ace_group_size': 8,
              'ace_pool_gamma': settings.pool_gamma, 'ace_ramp_ratio': settings.ramp_ratio,
              'ace_concurrency': settings.concurrency, 'ace_target_mode': settings.target_mode,
              'ace_length_penalty': settings.length_penalty}
    command = build_command(slime, model, reference, data, output, steps=steps,
                            questions=settings.questions, gpus=settings.gpus,
                            resume=Path(settings.resume) if settings.resume else None, settings=settings)
    return {'configuration': asdict(settings), 'training_rows': rows, 'steps': steps,
            'drop_last_per_epoch': rows % settings.questions,
            'questions_per_step': settings.questions, 'trajectories_per_step': settings.questions * 16,
            'custom_config': custom, 'command': command,
            'validation_status': 'CPU tests only; distributed H20 execution not established'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=os.environ.get('ACE_SQL_CONFIG'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without files, GPU load, or training')
    options = parser.parse_args(argv)
    settings = load_config(options.config, overrides={k: getattr(options, k) for k in ('seed', 'output', 'resume')})
    plan = build_plan(settings)
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        print(shlex.join(plan['command']))
        return
    for path in (Path(settings.slime) / 'train.py', Path(settings.model) / 'config.json',
                 Path(settings.data) / 'train.jsonl', Path(settings.data) / 'manifest.json',
                 Path(settings.data) / 'validation.jsonl', Path(settings.data) / 'initial_pool.json',
                 Path(settings.reference) / 'latest_checkpointed_iteration.txt'):
        if not path.is_file():
            raise FileNotFoundError(f'Required asset missing: {path}')
    output = Path(settings.output)
    if not settings.resume and (output / 'latest_checkpointed_iteration.txt').exists():
        raise FileExistsError('Output already contains a model checkpoint; use --resume explicitly')
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'ace_sql.runtime.environment', '--slime-dir', settings.slime,
                    '--expected-cuda', settings.expected_cuda, '--expected-gpus', str(settings.gpus),
                    '--report', str(output / 'environment.json'), '--gpu-smoke'], check=True)
    subprocess.run([sys.executable, '-m', 'ace_sql.runtime.metadata_patch', settings.slime], check=True)
    (output / 'ace_config.json').write_text(json.dumps(plan['custom_config'], indent=2) + '\n')
    (output / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    os.execv(sys.executable, plan['command'])


if __name__ == '__main__':
    main()
