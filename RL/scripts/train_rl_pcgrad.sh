#!/usr/bin/env bash
set -euo pipefail

# Joint two-pass GRPO training with PCGrad enabled for shared-policy updates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ "${ACE_SQL_DEBUG_SHELL:-0}" = "1" ]; then
    set -x
fi

if [ -n "${ACE_SQL_CONDA_SH:-}" ] && [ -f "${ACE_SQL_CONDA_SH}" ]; then
    source "${ACE_SQL_CONDA_SH}"
elif [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_ENV_NAME="${ACE_SQL_CONDA_ENV:-ace-sql}"
    conda activate "${CONDA_ENV_NAME}" || {
        echo "Failed to activate conda environment: ${CONDA_ENV_NAME}" >&2
        exit 1
    }
fi

export PYTHONPATH=".:./third_party:${PYTHONPATH:-}"
export WANDB_MODE="${ACE_SQL_WANDB_MODE:-offline}"
export WANDB_DIR="${ACE_SQL_WANDB_DIR:-.run/wandb}"
export VLLM_USE_V1=1

export ACE_SQL_DUAL_LORA_ENABLED=False
export ACE_SQL_LORA_RANK=0
export ACE_SQL_LORA_ALPHA=0

export ACE_SQL_EXPERIMENT_NAME="${ACE_SQL_EXPERIMENT_NAME:-rl_pcgrad}"
export ACE_SQL_MODEL_PATH="${ACE_SQL_MODEL_PATH:-models/sft_checkpoint}"
export ACE_SQL_TRAIN_FILE="${ACE_SQL_TRAIN_FILE:-data/train.parquet}"
export ACE_SQL_VAL_FILE="${ACE_SQL_VAL_FILE:-data/validation.parquet}"
export ACE_SQL_INITIAL_POOL_PATH="${ACE_SQL_INITIAL_POOL_PATH:-data/initial_pool.json}"

export ACE_SQL_TRAIN_DB_ROOT="${ACE_SQL_TRAIN_DB_ROOT:-external/train_databases}"
export ACE_SQL_DEV_DB_ROOT="${ACE_SQL_DEV_DB_ROOT:-external/dev_databases}"
export ACE_SQL_LOOSE_DB_ROOTS="${ACE_SQL_LOOSE_DB_ROOTS:-external/databases}"

export ACE_SQL_CKPT_ROOT="${ACE_SQL_CKPT_ROOT:-.run}"
export ACE_SQL_LOCAL_TMP_BASE="${ACE_SQL_LOCAL_TMP_BASE:-.run/local_tmp}"
export ACE_SQL_SYSTEM_TMP_BASE="${ACE_SQL_SYSTEM_TMP_BASE:-.run/system_tmp}"
export ACE_SQL_TMP_ROOT="${ACE_SQL_TMP_ROOT:-${ACE_SQL_LOCAL_TMP_BASE}/tmp}"
export ACE_SQL_RAY_TMPDIR="${ACE_SQL_RAY_TMPDIR:-${ACE_SQL_LOCAL_TMP_BASE}/ray}"
export ACE_SQL_RAY_SPILL_DIR_PRIMARY="${ACE_SQL_RAY_SPILL_DIR_PRIMARY:-${ACE_SQL_LOCAL_TMP_BASE}/ray_spill}"
export ACE_SQL_RAY_SPILL_DIR_FALLBACK="${ACE_SQL_RAY_SPILL_DIR_FALLBACK:-${ACE_SQL_SYSTEM_TMP_BASE}/ray_spill}"

export ACE_SQL_ACTOR_LR="${ACE_SQL_ACTOR_LR:-1e-6}"
export ACE_SQL_ACTOR_ADAM_EPS="${ACE_SQL_ACTOR_ADAM_EPS:-1e-8}"
export ACE_SQL_ACTOR_WEIGHT_DECAY="${ACE_SQL_ACTOR_WEIGHT_DECAY:-0.0}"
export ACE_SQL_ACTOR_GRAD_CLIP="${ACE_SQL_ACTOR_GRAD_CLIP:-1.0}"
export ACE_SQL_ACTOR_CLIP_RATIO="${ACE_SQL_ACTOR_CLIP_RATIO:-0.2}"
export ACE_SQL_ACTOR_ENTROPY_COEFF="${ACE_SQL_ACTOR_ENTROPY_COEFF:-0.0}"
export ACE_SQL_ACTOR_WARMUP_STYLE="${ACE_SQL_ACTOR_WARMUP_STYLE:-constant}"
export ACE_SQL_ACTOR_LR_WARMUP_STEPS_RATIO="${ACE_SQL_ACTOR_LR_WARMUP_STEPS_RATIO:-0.05}"
export ACE_SQL_ACTOR_MIN_LR_RATIO="${ACE_SQL_ACTOR_MIN_LR_RATIO:-0.2}"

export ACE_SQL_TRAIN_BATCH_SIZE="${ACE_SQL_TRAIN_BATCH_SIZE:-16}"
export ACE_SQL_VAL_BATCH_SIZE="${ACE_SQL_VAL_BATCH_SIZE:-16}"
export ACE_SQL_PPO_MINI_BATCH_SIZE="${ACE_SQL_PPO_MINI_BATCH_SIZE:-${ACE_SQL_TRAIN_BATCH_SIZE}}"
export ACE_SQL_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACE_SQL_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ACE_SQL_ACTOR_PPO_EPOCHS="${ACE_SQL_ACTOR_PPO_EPOCHS:-1}"

export ACE_SQL_ROLLOUT_N="${ACE_SQL_ROLLOUT_N:-8}"
export ACE_SQL_RET_N="${ACE_SQL_RET_N:-8}"
export ACE_SQL_RETRIEVER_ONLY_WARMUP_EPOCHS=0
export ACE_SQL_RETRIEVER_ONLY_RET_N="${ACE_SQL_RETRIEVER_ONLY_RET_N:-8}"
export ACE_SQL_TRAINING_PHASE_SCHEDULE=joint
export ACE_SQL_GENERATOR_PROMPT_MODE="${ACE_SQL_GENERATOR_PROMPT_MODE:-majority_vote}"
export ACE_SQL_GENERATOR_PROMPT_VOTE_THRESHOLD="${ACE_SQL_GENERATOR_PROMPT_VOTE_THRESHOLD:-0.5}"
export ACE_SQL_RETRIEVER_REWARD_MODE="${ACE_SQL_RETRIEVER_REWARD_MODE:-pool_exact}"
export ACE_SQL_POOL_EXACT_REWARD="${ACE_SQL_POOL_EXACT_REWARD:-1.0}"
export ACE_SQL_POOL_GAMMA="${ACE_SQL_POOL_GAMMA:-0.5}"
export ACE_SQL_CONSTANT_LOSS_SCALE=False
export ACE_SQL_LOSS_WEIGHT_SCHEDULE=linear_joint_ramp
export ACE_SQL_LOSS_WEIGHT_RAMP_RATIO="${ACE_SQL_LOSS_WEIGHT_RAMP_RATIO:-0.25}"
export ACE_SQL_RETRIEVER_LOSS_WEIGHT_START="${ACE_SQL_RETRIEVER_LOSS_WEIGHT_START:-1.0}"
export ACE_SQL_RETRIEVER_LOSS_WEIGHT_END="${ACE_SQL_RETRIEVER_LOSS_WEIGHT_END:-1.0}"
export ACE_SQL_GENERATOR_LOSS_WEIGHT_START="${ACE_SQL_GENERATOR_LOSS_WEIGHT_START:-0.0}"
export ACE_SQL_GENERATOR_LOSS_WEIGHT_END="${ACE_SQL_GENERATOR_LOSS_WEIGHT_END:-1.0}"

export ACE_SQL_GRAD_PROJ_ENABLED="${ACE_SQL_GRAD_PROJ_ENABLED:-True}"
export ACE_SQL_GRAD_PROJ_MODE="${ACE_SQL_GRAD_PROJ_MODE:-symmetric}"
export ACE_SQL_GRAD_PROJ_NORMALIZE_TASK_GRADS="${ACE_SQL_GRAD_PROJ_NORMALIZE_TASK_GRADS:-False}"
export ACE_SQL_GRAD_PROJ_MAIN_TASK="${ACE_SQL_GRAD_PROJ_MAIN_TASK:-generator}"
export ACE_SQL_GRAD_PROJ_AUX_TASK="${ACE_SQL_GRAD_PROJ_AUX_TASK:-retriever}"
export ACE_SQL_GRAD_PROJ_AUX_WEIGHT="${ACE_SQL_GRAD_PROJ_AUX_WEIGHT:-1.0}"
export ACE_SQL_GRAD_PROJ_EPS="${ACE_SQL_GRAD_PROJ_EPS:-1e-12}"
export ACE_SQL_GRAD_PROJ_MAIN_GRAD_NORM_EMA_DECAY="${ACE_SQL_GRAD_PROJ_MAIN_GRAD_NORM_EMA_DECAY:-0.95}"
export ACE_SQL_GRAD_PROJ_MAIN_GRAD_NORM_FLOOR_MIN="${ACE_SQL_GRAD_PROJ_MAIN_GRAD_NORM_FLOOR_MIN:-0.0}"
export ACE_SQL_GRAD_PROJ_PRE_BOOST_GENERATOR="${ACE_SQL_GRAD_PROJ_PRE_BOOST_GENERATOR:-False}"
export ACE_SQL_GRAD_PROJ_PRE_BOOST_TARGET_RATIO="${ACE_SQL_GRAD_PROJ_PRE_BOOST_TARGET_RATIO:-1.0}"
export ACE_SQL_GRAD_PROJ_PRE_BOOST_MAX_SCALE="${ACE_SQL_GRAD_PROJ_PRE_BOOST_MAX_SCALE:-10.0}"
export ACE_SQL_GRAD_PROJ_MAX_RATIO="${ACE_SQL_GRAD_PROJ_MAX_RATIO:-0.3}"

export ACE_SQL_ROLLOUT_TEMPERATURE="${ACE_SQL_ROLLOUT_TEMPERATURE:-1.0}"
export ACE_SQL_VALIDATION_RETRIEVER_TEMPERATURE="${ACE_SQL_VALIDATION_RETRIEVER_TEMPERATURE:-0.8}"
export ACE_SQL_VALIDATION_GENERATOR_TEMPERATURE="${ACE_SQL_VALIDATION_GENERATOR_TEMPERATURE:-0.8}"

export ACE_SQL_RETRIEVER_RESPONSE_LENGTH="${ACE_SQL_RETRIEVER_RESPONSE_LENGTH:-2048}"
export ACE_SQL_GENERATOR_RESPONSE_LENGTH="${ACE_SQL_GENERATOR_RESPONSE_LENGTH:-2048}"
if [ -z "${ACE_SQL_MAX_RESPONSE_LENGTH:-}" ]; then
    ACE_SQL_MAX_RESPONSE_LENGTH="$(python3 - <<'PY'
import os

ret = int(os.environ["ACE_SQL_RETRIEVER_RESPONSE_LENGTH"])
gen = int(os.environ["ACE_SQL_GENERATOR_RESPONSE_LENGTH"])
print(max(ret, gen))
PY
)"
    export ACE_SQL_MAX_RESPONSE_LENGTH
fi
export ACE_SQL_ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACE_SQL_ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-12000}"
export ACE_SQL_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ACE_SQL_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-12000}"
export ACE_SQL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ACE_SQL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ACE_SQL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ACE_SQL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${ACE_SQL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}}"
export ACE_SQL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ACE_SQL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${ACE_SQL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}}"
export ACE_SQL_ACTOR_USE_DYNAMIC_BSZ="${ACE_SQL_ACTOR_USE_DYNAMIC_BSZ:-True}"
export ACE_SQL_LOG_PROB_USE_DYNAMIC_BSZ="${ACE_SQL_LOG_PROB_USE_DYNAMIC_BSZ:-True}"
export ACE_SQL_USE_REMOVE_PADDING="${ACE_SQL_USE_REMOVE_PADDING:-True}"
export ACE_SQL_ACTOR_FSDP_SIZE="${ACE_SQL_ACTOR_FSDP_SIZE:--1}"
export ACE_SQL_ACTOR_ULYSSES_SP_SIZE="${ACE_SQL_ACTOR_ULYSSES_SP_SIZE:-1}"

export ACE_SQL_ACTOR_MODEL_DTYPE="${ACE_SQL_ACTOR_MODEL_DTYPE:-fp32}"
export ACE_SQL_ACTOR_MP_PARAM_DTYPE="${ACE_SQL_ACTOR_MP_PARAM_DTYPE:-bf16}"
export ACE_SQL_ACTOR_MP_REDUCE_DTYPE="${ACE_SQL_ACTOR_MP_REDUCE_DTYPE:-fp32}"
export ACE_SQL_ACTOR_MP_BUFFER_DTYPE="${ACE_SQL_ACTOR_MP_BUFFER_DTYPE:-fp32}"
export ACE_SQL_ACTOR_PARAM_OFFLOAD="${ACE_SQL_ACTOR_PARAM_OFFLOAD:-False}"
export ACE_SQL_ACTOR_OPTIMIZER_OFFLOAD="${ACE_SQL_ACTOR_OPTIMIZER_OFFLOAD:-False}"
export ACE_SQL_REF_PARAM_OFFLOAD="${ACE_SQL_REF_PARAM_OFFLOAD:-False}"
export ACE_SQL_ENABLE_ACTIVATION_OFFLOAD="${ACE_SQL_ENABLE_ACTIVATION_OFFLOAD:-False}"
export ACE_SQL_ACTOR_EMPTY_CACHE_AROUND_UPDATE="${ACE_SQL_ACTOR_EMPTY_CACHE_AROUND_UPDATE:-True}"
export ACE_SQL_ACTOR_EMPTY_CACHE_PER_MINI_BATCH="${ACE_SQL_ACTOR_EMPTY_CACHE_PER_MINI_BATCH:-False}"

export ACE_SQL_ACTOR_CHECKPOINT_SAVE_CONTENTS="${ACE_SQL_ACTOR_CHECKPOINT_SAVE_CONTENTS:-[\"hf_model\"]}"
export ACE_SQL_MAX_ACTOR_CKPT_TO_KEEP="${ACE_SQL_MAX_ACTOR_CKPT_TO_KEEP:-2}"

export ACE_SQL_ROLLOUT_GPU_MEMORY_UTILIZATION="${ACE_SQL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
export ACE_SQL_ROLLOUT_MAX_MODEL_LEN="${ACE_SQL_ROLLOUT_MAX_MODEL_LEN:-6144}"
export ACE_SQL_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ACE_SQL_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
export ACE_SQL_ROLLOUT_MAX_NUM_SEQS="${ACE_SQL_ROLLOUT_MAX_NUM_SEQS:-128}"
export ACE_SQL_ROLLOUT_ENABLE_CHUNKED_PREFILL="${ACE_SQL_ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
export ACE_SQL_AGENT_NUM_WORKERS="${ACE_SQL_AGENT_NUM_WORKERS:-8}"
export ACE_SQL_REWARD_NUM_WORKERS="${ACE_SQL_REWARD_NUM_WORKERS:-8}"
export ACE_SQL_DATALOADER_WORKERS="${ACE_SQL_DATALOADER_WORKERS:-8}"

export ACE_SQL_TOTAL_EPOCHS="${ACE_SQL_TOTAL_EPOCHS:-4}"
export ACE_SQL_TEST_FREQ="${ACE_SQL_TEST_FREQ:-40}"
export ACE_SQL_SAVE_FREQ="${ACE_SQL_SAVE_FREQ:-80}"
export ACE_SQL_VAL_BEFORE_TRAIN="${ACE_SQL_VAL_BEFORE_TRAIN:-True}"

sanitize_thread_env() {
    local name="$1"
    local current="${!name:-}"
    if [[ ! "${current}" =~ ^[1-9][0-9]*$ ]]; then
        export "${name}"=1
    fi
}

detect_available_cpus() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        echo 1
    fi
}

detect_memory_limit_bytes() {
    local mem_kb
    mem_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
    if [ -n "${mem_kb}" ] && [ "${mem_kb}" -gt 0 ]; then
        echo "$(( mem_kb * 1024 ))"
    else
        echo 0
    fi
}

require_value() {
    local name="$1"
    local expected="$2"
    local actual="${!name:-}"
    if [ "${actual}" != "${expected}" ]; then
        echo "Invalid full-PCGrad config: ${name}=${actual}, expected ${expected}" >&2
        exit 1
    fi
}

require_true() {
    local name="$1"
    local actual="${!name:-}"
    case "${actual}" in
        True|true|1|yes|on) ;;
        *)
            echo "Invalid full-PCGrad config: ${name}=${actual}, expected true" >&2
            exit 1
            ;;
    esac
}

require_file() {
    local path="$1"
    local label="$2"
    if [ ! -f "${path}" ]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

require_min_free_gb() {
    local path="$1"
    local min_gb="$2"
    mkdir -p "${path}"
    local avail_kb
    avail_kb="$(df -Pk "${path}" | awk 'NR==2 {print $4}')"
    local avail_gb=$(( avail_kb / 1024 / 1024 ))
    if [ "${avail_gb}" -lt "${min_gb}" ]; then
        echo "Insufficient free space: ${path} has ${avail_gb}GB, need ${min_gb}GB." >&2
        exit 1
    fi
}

sanitize_thread_env OMP_NUM_THREADS
sanitize_thread_env MKL_NUM_THREADS
sanitize_thread_env OPENBLAS_NUM_THREADS
sanitize_thread_env NUMEXPR_NUM_THREADS

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)"
    if [ "${GPU_COUNT:-0}" -lt 4 ]; then
        echo "Full-parameter joint training is configured for 4 GPUs; found ${GPU_COUNT:-0} visible GPUs." >&2
        echo "Set CUDA_VISIBLE_DEVICES explicitly for a smaller dry run." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"
if [ "${TRAINER_N_GPUS_PER_NODE}" -lt 4 ]; then
    echo "Full-parameter joint training expects 4 visible GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 1
fi

AVAILABLE_CPU_COUNT="$(detect_available_cpus)"
MEMORY_LIMIT_BYTES="$(detect_memory_limit_bytes)"
AVAILABLE_MEM_GB="$(( MEMORY_LIMIT_BYTES / 1024 / 1024 / 1024 ))"
export ACE_SQL_AVAILABLE_CPU_COUNT="${ACE_SQL_AVAILABLE_CPU_COUNT:-${AVAILABLE_CPU_COUNT}}"
export ACE_SQL_AVAILABLE_MEM_GB="${ACE_SQL_AVAILABLE_MEM_GB:-${AVAILABLE_MEM_GB}}"
export ACE_SQL_RAY_NUM_CPUS="${ACE_SQL_RAY_NUM_CPUS:-${ACE_SQL_AVAILABLE_CPU_COUNT}}"
export ACE_SQL_RAY_OBJECT_STORE_MEMORY_GB="${ACE_SQL_RAY_OBJECT_STORE_MEMORY_GB:-8}"

require_value ACE_SQL_DUAL_LORA_ENABLED False
require_value ACE_SQL_LORA_RANK 0
require_value ACE_SQL_LORA_ALPHA 0
require_value ACE_SQL_TRAINING_PHASE_SCHEDULE joint
require_true ACE_SQL_GRAD_PROJ_ENABLED
require_file "${ACE_SQL_MODEL_PATH}/config.json" "model config"
require_file "${ACE_SQL_TRAIN_FILE}" "train parquet"
require_file "${ACE_SQL_VAL_FILE}" "validation parquet"
require_file "${ACE_SQL_INITIAL_POOL_PATH}" "initial pool"

require_min_free_gb "${ACE_SQL_TMP_ROOT}" 20
require_min_free_gb "${ACE_SQL_RAY_TMPDIR}" 20
require_min_free_gb "${ACE_SQL_CKPT_ROOT}" 40

mkdir -p \
    "${ACE_SQL_CKPT_ROOT}" \
    "${ACE_SQL_TMP_ROOT}" \
    "${ACE_SQL_RAY_TMPDIR}" \
    "${ACE_SQL_SYSTEM_TMP_BASE}" \
    "${ACE_SQL_RAY_SPILL_DIR_PRIMARY}" \
    "${ACE_SQL_RAY_SPILL_DIR_FALLBACK}" \
    "${WANDB_DIR}"

export TMPDIR="${ACE_SQL_TMP_ROOT}"
export TEMP="${ACE_SQL_TMP_ROOT}"
export TMP="${ACE_SQL_TMP_ROOT}"
export RAY_TMPDIR="${ACE_SQL_RAY_TMPDIR}"

if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
fi
rm -rf "${ACE_SQL_RAY_TMPDIR}/session_"* "${ACE_SQL_RAY_TMPDIR}/session_latest" 2>/dev/null || true
rm -rf "${ACE_SQL_RAY_SPILL_DIR_PRIMARY}/"ray_spilled_objects_* "${ACE_SQL_RAY_SPILL_DIR_FALLBACK}/"ray_spilled_objects_* 2>/dev/null || true

if ! python3 - <<'PY'
import importlib.util
import sys

missing = [name for name in ("flash_attn", "flash_attn_2_cuda") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing required precompiled packages:", ", ".join(missing))
    sys.exit(1)
PY
then
    echo "flash-attn is not ready in the current environment." >&2
    exit 1
fi

python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

import pandas as pd

train_file = Path(os.environ["ACE_SQL_TRAIN_FILE"])
val_file = Path(os.environ["ACE_SQL_VAL_FILE"])
pool_file = Path(os.environ["ACE_SQL_INITIAL_POOL_PATH"])

def maybe_json(value):
    return json.loads(value) if isinstance(value, str) else value

def row_key(row):
    reward_model = maybe_json(row["reward_model"])
    ground_truth = reward_model.get("ground_truth", {})
    extra_info = maybe_json(row["extra_info"])
    db_id = ground_truth.get("db_id") or extra_info.get("db_id")
    question = ground_truth.get("question") or extra_info.get("question")
    return f"{db_id}||{question}"

errors = []
pool = json.loads(pool_file.read_text(encoding="utf-8"))
train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)
train_keys = {row_key(row) for _, row in train_df.iterrows()}
val_keys = {row_key(row) for _, row in val_df.iterrows()}
overlap = sorted(train_keys & val_keys)
if overlap:
    errors.append(f"train/val key overlap detected: {len(overlap)}; first={overlap[0][:160]}")

missing_train = sorted(key for key in train_keys if key not in pool)
missing_val = sorted(key for key in val_keys if key not in pool)
if missing_train:
    errors.append(f"train keys missing from pool: {len(missing_train)}; first={missing_train[0][:160]}")
if missing_val:
    errors.append(f"val keys missing from pool: {len(missing_val)}; first={missing_val[0][:160]}")

print(
    "[rl-pcgrad data preflight] "
    f"train_rows={len(train_df)} train_keys={len(train_keys)} "
    f"val_rows={len(val_df)} val_keys={len(val_keys)} "
    f"pool_keys={len(pool)}"
)
if errors:
    print("[rl-pcgrad data preflight] FAILED", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(1)
print("[rl-pcgrad data preflight] OK")
PY

echo "===== ACE-SQL RL PCGrad Training Configuration ====="
echo "Model path:           ${ACE_SQL_MODEL_PATH}"
echo "Experiment name:      ${ACE_SQL_EXPERIMENT_NAME}"
echo "Train file:           ${ACE_SQL_TRAIN_FILE}"
echo "Validation file:      ${ACE_SQL_VAL_FILE}"
echo "Initial pool:         ${ACE_SQL_INITIAL_POOL_PATH}"
echo "Batch / rollout n:    ${ACE_SQL_TRAIN_BATCH_SIZE} / ${ACE_SQL_ROLLOUT_N}"
echo "LR / grad clip:       ${ACE_SQL_ACTOR_LR} / ${ACE_SQL_ACTOR_GRAD_CLIP}"
echo "PCGrad:               enabled=${ACE_SQL_GRAD_PROJ_ENABLED}, mode=${ACE_SQL_GRAD_PROJ_MODE}, normalize=${ACE_SQL_GRAD_PROJ_NORMALIZE_TASK_GRADS}"
echo "Loss weights:         ${ACE_SQL_RETRIEVER_LOSS_WEIGHT_START}->${ACE_SQL_RETRIEVER_LOSS_WEIGHT_END}, ${ACE_SQL_GENERATOR_LOSS_WEIGHT_START}->${ACE_SQL_GENERATOR_LOSS_WEIGHT_END}"
echo "Tmp / ckpt root:      ${ACE_SQL_TMP_ROOT} / ${ACE_SQL_CKPT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

EXTRA_ARGS=()
if [ -n "${ACE_SQL_RAY_OBJECT_STORE_MEMORY_GB}" ]; then
    ACE_SQL_RAY_OBJECT_STORE_MEMORY_BYTES="$(python3 - <<PY
print(int(float("${ACE_SQL_RAY_OBJECT_STORE_MEMORY_GB}") * (1024 ** 3)))
PY
)"
    EXTRA_ARGS+=("ray_init.object_store_memory=${ACE_SQL_RAY_OBJECT_STORE_MEMORY_BYTES}")
fi
EXTRA_ARGS+=("ray_init.object_spilling_dirs=[\"${ACE_SQL_RAY_SPILL_DIR_PRIMARY}\",\"${ACE_SQL_RAY_SPILL_DIR_FALLBACK}\"]")

python3 -m trainer.main_two_pass \
    data.train_files="${ACE_SQL_TRAIN_FILE}" \
    data.val_files="${ACE_SQL_VAL_FILE}" \
    data.train_batch_size="${ACE_SQL_TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${ACE_SQL_VAL_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length="${ACE_SQL_MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.seed=1 \
    data.dataloader_num_workers="${ACE_SQL_DATALOADER_WORKERS}" \
    actor_rollout_ref.model.path="${ACE_SQL_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_activation_offload="${ACE_SQL_ENABLE_ACTIVATION_OFFLOAD}" \
    actor_rollout_ref.model.use_remove_padding="${ACE_SQL_USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.lora_rank="${ACE_SQL_LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${ACE_SQL_LORA_ALPHA}" \
    actor_rollout_ref.model.dual_lora.enabled="${ACE_SQL_DUAL_LORA_ENABLED}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ACE_SQL_PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACE_SQL_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_epochs="${ACE_SQL_ACTOR_PPO_EPOCHS}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${ACE_SQL_ACTOR_USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACE_SQL_ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
    +actor_rollout_ref.actor.empty_cache_per_mini_batch="${ACE_SQL_ACTOR_EMPTY_CACHE_PER_MINI_BATCH}" \
    +actor_rollout_ref.actor.empty_cache_around_update="${ACE_SQL_ACTOR_EMPTY_CACHE_AROUND_UPDATE}" \
    actor_rollout_ref.actor.entropy_coeff="${ACE_SQL_ACTOR_ENTROPY_COEFF}" \
    actor_rollout_ref.actor.clip_ratio="${ACE_SQL_ACTOR_CLIP_RATIO}" \
    actor_rollout_ref.actor.clip_ratio_low="${ACE_SQL_ACTOR_CLIP_RATIO}" \
    actor_rollout_ref.actor.clip_ratio_high="${ACE_SQL_ACTOR_CLIP_RATIO}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr="${ACE_SQL_ACTOR_LR}" \
    +actor_rollout_ref.actor.optim.eps="${ACE_SQL_ACTOR_ADAM_EPS}" \
    actor_rollout_ref.actor.optim.weight_decay="${ACE_SQL_ACTOR_WEIGHT_DECAY}" \
    actor_rollout_ref.actor.optim.warmup_style="${ACE_SQL_ACTOR_WARMUP_STYLE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACE_SQL_ACTOR_LR_WARMUP_STEPS_RATIO}" \
    actor_rollout_ref.actor.optim.min_lr_ratio="${ACE_SQL_ACTOR_MIN_LR_RATIO}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACE_SQL_ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACE_SQL_ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${ACE_SQL_ACTOR_FSDP_SIZE}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ACE_SQL_ACTOR_ULYSSES_SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${ACE_SQL_ACTOR_MODEL_DTYPE}" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype="${ACE_SQL_ACTOR_MP_PARAM_DTYPE}" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype="${ACE_SQL_ACTOR_MP_REDUCE_DTYPE}" \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype="${ACE_SQL_ACTOR_MP_BUFFER_DTYPE}" \
    actor_rollout_ref.actor.grad_clip="${ACE_SQL_ACTOR_GRAD_CLIP}" \
    actor_rollout_ref.actor.gradient_projection.enabled="${ACE_SQL_GRAD_PROJ_ENABLED}" \
    actor_rollout_ref.actor.gradient_projection.mode="${ACE_SQL_GRAD_PROJ_MODE}" \
    actor_rollout_ref.actor.gradient_projection.normalize_task_grads="${ACE_SQL_GRAD_PROJ_NORMALIZE_TASK_GRADS}" \
    actor_rollout_ref.actor.gradient_projection.main_task="${ACE_SQL_GRAD_PROJ_MAIN_TASK}" \
    actor_rollout_ref.actor.gradient_projection.aux_task="${ACE_SQL_GRAD_PROJ_AUX_TASK}" \
    actor_rollout_ref.actor.gradient_projection.aux_weight="${ACE_SQL_GRAD_PROJ_AUX_WEIGHT}" \
    actor_rollout_ref.actor.gradient_projection.eps="${ACE_SQL_GRAD_PROJ_EPS}" \
    actor_rollout_ref.actor.gradient_projection.main_grad_norm_ema_decay="${ACE_SQL_GRAD_PROJ_MAIN_GRAD_NORM_EMA_DECAY}" \
    actor_rollout_ref.actor.gradient_projection.main_grad_norm_floor_min="${ACE_SQL_GRAD_PROJ_MAIN_GRAD_NORM_FLOOR_MIN}" \
    actor_rollout_ref.actor.gradient_projection.pre_boost_generator="${ACE_SQL_GRAD_PROJ_PRE_BOOST_GENERATOR}" \
    actor_rollout_ref.actor.gradient_projection.pre_boost_target_ratio="${ACE_SQL_GRAD_PROJ_PRE_BOOST_TARGET_RATIO}" \
    actor_rollout_ref.actor.gradient_projection.pre_boost_max_scale="${ACE_SQL_GRAD_PROJ_PRE_BOOST_MAX_SCALE}" \
    actor_rollout_ref.actor.gradient_projection.max_ratio="${ACE_SQL_GRAD_PROJ_MAX_RATIO}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ACE_SQL_ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${ACE_SQL_ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ACE_SQL_ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ACE_SQL_ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ACE_SQL_ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.rollout.temperature="${ACE_SQL_ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.n="${ACE_SQL_ROLLOUT_N}" \
    actor_rollout_ref.rollout.agent.num_workers="${ACE_SQL_AGENT_NUM_WORKERS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ACE_SQL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${ACE_SQL_LOG_PROB_USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${ACE_SQL_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    +actor_rollout_ref.ref.fsdp_config.param_offload="${ACE_SQL_REF_PARAM_OFFLOAD}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ACE_SQL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${ACE_SQL_LOG_PROB_USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ACE_SQL_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    serl.constant_loss_scale="${ACE_SQL_CONSTANT_LOSS_SCALE}" \
    serl.loss_weight_schedule="${ACE_SQL_LOSS_WEIGHT_SCHEDULE}" \
    serl.loss_weight_ramp_ratio="${ACE_SQL_LOSS_WEIGHT_RAMP_RATIO}" \
    serl.retriever_loss_weight_start="${ACE_SQL_RETRIEVER_LOSS_WEIGHT_START}" \
    serl.retriever_loss_weight_end="${ACE_SQL_RETRIEVER_LOSS_WEIGHT_END}" \
    serl.generator_loss_weight_start="${ACE_SQL_GENERATOR_LOSS_WEIGHT_START}" \
    serl.generator_loss_weight_end="${ACE_SQL_GENERATOR_LOSS_WEIGHT_END}" \
    serl.alpha_start=1.0 \
    serl.alpha_mid=0.3 \
    serl.alpha_end=0.3 \
    serl.phase1_end=0.3 \
    serl.phase2_end=0.6 \
    serl.ret_n="${ACE_SQL_RET_N}" \
    serl.retriever_only_warmup_epochs="${ACE_SQL_RETRIEVER_ONLY_WARMUP_EPOCHS}" \
    serl.retriever_only_ret_n="${ACE_SQL_RETRIEVER_ONLY_RET_N}" \
    serl.training_phase_schedule="'${ACE_SQL_TRAINING_PHASE_SCHEDULE}'" \
    serl.gen_max_prompt_length=4096 \
    serl.generator_prompt_mode="${ACE_SQL_GENERATOR_PROMPT_MODE}" \
    serl.generator_prompt_vote_threshold="${ACE_SQL_GENERATOR_PROMPT_VOTE_THRESHOLD}" \
    serl.retriever_reward_mode="${ACE_SQL_RETRIEVER_REWARD_MODE}" \
    serl.pool_exact_reward="${ACE_SQL_POOL_EXACT_REWARD}" \
    serl.pool_gamma="${ACE_SQL_POOL_GAMMA}" \
    serl.initial_pool_path="${ACE_SQL_INITIAL_POOL_PATH}" \
    +serl.validation_retriever_temperature="${ACE_SQL_VALIDATION_RETRIEVER_TEMPERATURE}" \
    +serl.validation_generator_temperature="${ACE_SQL_VALIDATION_GENERATOR_TEMPERATURE}" \
    serl.retriever_response_length="${ACE_SQL_RETRIEVER_RESPONSE_LENGTH}" \
    serl.generator_response_length="${ACE_SQL_GENERATOR_RESPONSE_LENGTH}" \
    reward.num_workers="${ACE_SQL_REWARD_NUM_WORKERS}" \
    ray_init.num_cpus="${ACE_SQL_RAY_NUM_CPUS}" \
    ray_init.temp_dir="${ACE_SQL_RAY_TMPDIR}" \
    trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${ACE_SQL_TOTAL_EPOCHS}" \
    trainer.save_freq="${ACE_SQL_SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${ACE_SQL_MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.default_local_dir="${ACE_SQL_CKPT_ROOT}/checkpoints/${ACE_SQL_EXPERIMENT_NAME}" \
    trainer.resume_mode=disable \
    "actor_rollout_ref.actor.checkpoint.save_contents=${ACE_SQL_ACTOR_CHECKPOINT_SAVE_CONTENTS}" \
    trainer.test_freq="${ACE_SQL_TEST_FREQ}" \
    trainer.val_before_train="${ACE_SQL_VAL_BEFORE_TRAIN}" \
    trainer.logger='["console","wandb"]' \
    trainer.experiment_name="${ACE_SQL_EXPERIMENT_NAME}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
