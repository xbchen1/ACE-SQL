"""Build and run the pinned slime command; --dry-run requires no GPU packages."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def env(name, default):
    return os.environ.get("ACE_SQL_" + name, str(default))


def build_command(slime, model, reference, data, output, *, steps, questions=16, gpus=8, resume=None):
    config_path = model / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {
        "model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
        "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000, "vocab_size": 151936, "head_dim": 128,
    }
    if config["model_type"] != "qwen3":
        raise ValueError("This recipe supports dense Qwen3-4B/8B checkpoints")
    command = [sys.executable, str(slime / "train.py")]
    values = {
        "actor-num-nodes": 1, "actor-num-gpus-per-node": gpus, "num-gpus-per-node": gpus,
        "hf-checkpoint": model, "ref-load": reference,
        "save": output, "save-interval": 80, "eval-interval": 40,
        "save-hf": output / "hf" / "rollout_{rollout_id}",
        "prompt-data": data / "train.jsonl", "input-key": "prompt", "label-key": "label",
        "metadata-key": "metadata", "data-source-path": "ace_sql_slime.rollout.ACEDataSource",
        "rollout-function-path": "ace_sql_slime.rollout.generate_rollout",
        "eval-function-path": "ace_sql_slime.rollout.generate_rollout",
        "custom-reward-post-process-path": "ace_sql_slime.loss.reward_postprocess",
        "loss-type": "custom_loss", "custom-loss-function-path": "ace_sql_slime.loss.policy_loss",
        "custom-config-path": output / "ace_config.json",
        "num-rollout": steps, "rollout-batch-size": questions, "n-samples-per-prompt": 16,
        "global-batch-size": questions * 16, "micro-batch-size": 1,
        "rollout-max-prompt-len": 4096, "rollout-max-response-len": 2048,
        "rollout-max-context-len": 6144, "seq-length": 6144, "rollout-temperature": 1,
        "rollout-top-p": 1, "rollout-top-k": -1, "rollout-seed": env("SEED", 42),
        "seed": env("SEED", 42),
        "n-samples-per-eval-prompt": 1,
        "tensor-model-parallel-size": 2, "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1, "rollout-num-gpus-per-engine": 2,
        "sglang-mem-fraction-static": 0.65, "sglang-attention-backend": "flashinfer",
        "advantage-estimator": "grpo", "kl-coef": 0, "kl-loss-coef": 0.001,
        "kl-loss-type": "low_var_kl", "entropy-coef": 0,
        "eps-clip": 0.2, "eps-clip-high": 0.2,
        "optimizer": "adam", "lr": "1e-6", "lr-decay-style": "constant",
        "lr-warmup-fraction": 0.05, "weight-decay": 0, "clip-grad": 1,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan; no files/GPU startup")
    options = parser.parse_args()
    slime = Path(env("SLIME_DIR", ROOT / "external/slime")).expanduser().resolve()
    model = Path(env("MODEL_PATH", ROOT / "models/sft_checkpoint")).expanduser().resolve()
    reference = Path(env("REFERENCE_PATH", ROOT / "models/sft_torch_dist")).expanduser().resolve()
    data = Path(env("SLIME_DATA", ROOT / ".run/slime_data")).expanduser().resolve()
    output = Path(env("SLIME_OUTPUT", ROOT / ".run/slime_h20")).expanduser().resolve()
    resume = os.environ.get("ACE_SQL_SLIME_RESUME")
    resume = Path(resume).expanduser().resolve() if resume else None
    questions, gpus, epochs = int(env("TRAIN_BATCH_SIZE", 16)), int(env("NUM_GPUS", 8)), int(env("TOTAL_EPOCHS", 5))
    if questions < 1 or epochs < 1 or gpus < 2 or gpus % 2 or (questions * 16) % (gpus // 2):
        raise ValueError("Positive epochs/batch and even GPU count with divisible DP batch are required")
    manifest = data / "manifest.json"
    rows = sum(1 for line in (data / "train.jsonl").open() if line.strip()) if (data / "train.jsonl").exists() else 2913
    steps = (rows // questions) * epochs
    if steps < 1:
        raise ValueError("No complete training batches")
    gamma, ramp = float(env("POOL_GAMMA", 0.5)), float(env("RAMP_RATIO", 0.25))
    target = env("TARGET_MODE", "online")
    length_penalty = env("LENGTH_PENALTY", "true").lower()
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (gamma, ramp)):
        raise ValueError("POOL_GAMMA and RAMP_RATIO must be finite and in [0, 1]")
    if target not in {"online", "frozen", "gold", "dense"}:
        raise ValueError("TARGET_MODE must be online, frozen, gold, or dense")
    if length_penalty not in {"true", "false", "1", "0"}:
        raise ValueError("LENGTH_PENALTY must be true or false")
    custom = {"ace_initial_pool": str(data / "initial_pool.json"),
              "ace_validation_data": str(data / "validation.jsonl"), "ace_group_size": 8,
              "ace_pool_gamma": gamma, "ace_ramp_ratio": ramp, "ace_concurrency": 64,
              "ace_target_mode": target, "ace_length_penalty": length_penalty in {"true", "1"}}
    command = build_command(slime, model, reference, data, output, steps=steps,
                            questions=questions, gpus=gpus, resume=resume)
    plan = {"cuda_required": env("EXPECTED_CUDA", "12.9"), "epochs": epochs,
            "training_rows": rows, "steps": steps, "drop_last_per_epoch": rows % questions,
            "questions_per_step": questions, "trajectories_per_step": questions * 16,
            "custom_config": custom, "command": command,
            "validation_status": "CPU contract tests only; H20 execution not established"}
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        print(shlex.join(command))
        return
    if os.environ.get("ACE_SQL_GRAD_PROJ_ENABLED", "false").lower() not in {"false", "0", ""}:
        raise ValueError("Gradient projection is not supported by this recipe")
    for path in [slime / "train.py", model / "config.json", manifest, data / "validation.jsonl",
                 data / "initial_pool.json", reference / "latest_checkpointed_iteration.txt"]:
        if not path.is_file():
            raise FileNotFoundError(f"Required asset missing: {path}")
    if not resume and (output / "latest_checkpointed_iteration.txt").exists():
        raise FileExistsError("Output already contains a model checkpoint; set ACE_SQL_SLIME_RESUME explicitly")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/check_slime_env.py"), "--slime-dir", str(slime),
                    "--expected-cuda", env("EXPECTED_CUDA", "12.9"), "--expected-gpus", str(gpus),
                    "--report", str(output / "environment.json"), "--gpu-smoke"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/apply_slime_metadata_patch.py"), str(slime)], check=True)
    # Explicitly persist all method-specific args before Ray serializes them.
    (output / "ace_config.json").write_text(json.dumps(custom, indent=2) + "\n")
    (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
