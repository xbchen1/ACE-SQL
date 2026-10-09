#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RL_ROOT"
export PYTHONPATH="$RL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ACE_SQL_MODEL_PATH="${ACE_SQL_MODEL_PATH:-$RL_ROOT/models/sft_checkpoint}"
export ACE_SQL_REFERENCE_PATH="${ACE_SQL_REFERENCE_PATH:-$RL_ROOT/models/sft_torch_dist}"
export ACE_SQL_SLIME_DATA="${ACE_SQL_SLIME_DATA:-$RL_ROOT/.run/slime_data}"
export ACE_SQL_SLIME_OUTPUT="${ACE_SQL_SLIME_OUTPUT:-$RL_ROOT/.run/slime_h20}"
export ACE_SQL_SLIME_DIR="${ACE_SQL_SLIME_DIR:-/root/slime}"
PYTHON="${ACE_SQL_PYTHON:-python3}"
"$PYTHON" scripts/check_slime_env.py --slime-dir "$ACE_SQL_SLIME_DIR" \
  --expected-cuda "${ACE_SQL_EXPECTED_CUDA:-12.9}" --expected-gpus "${ACE_SQL_NUM_GPUS:-8}" --gpu-smoke
"$PYTHON" scripts/apply_slime_metadata_patch.py "$ACE_SQL_SLIME_DIR"
if [[ ! -f "$ACE_SQL_REFERENCE_PATH/latest_checkpointed_iteration.txt" ]]; then
  MODEL_SIZE="$($PYTHON - <<'PY'
import json, os
from pathlib import Path
c = json.loads((Path(os.environ['ACE_SQL_MODEL_PATH']) / 'config.json').read_text())
if c.get('model_type') != 'qwen3':
    raise SystemExit('Expected a dense Qwen3 SFT checkpoint')
size = {(36, 4096): '8B', (36, 2560): '4B'}.get((c['num_hidden_layers'], c['hidden_size']))
if size is None:
    raise SystemExit('Supported checkpoints: Qwen3-4B and Qwen3-8B')
target = Path(os.environ['ACE_SQL_REFERENCE_PATH'])
if target.exists() and any(target.iterdir()):
    raise SystemExit('Reference directory is nonempty but has no completed checkpoint')
print(size)
PY
)"
  source "$ACE_SQL_SLIME_DIR/scripts/models/qwen3-${MODEL_SIZE}.sh"
  torchrun --standalone --nproc_per_node="${ACE_SQL_NUM_GPUS:-8}" \
    "$ACE_SQL_SLIME_DIR/tools/convert_hf_to_torch_dist.py" "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$ACE_SQL_MODEL_PATH" --save "$ACE_SQL_REFERENCE_PATH"
fi
if [[ ! -f "$ACE_SQL_SLIME_DATA/manifest.json" ]]; then
  "$PYTHON" -m ace_sql_slime.data \
    --train "${ACE_SQL_TRAIN_FILE:-$RL_ROOT/data/train.parquet}" \
    --validation "${ACE_SQL_VAL_FILE:-$RL_ROOT/data/validation.parquet}" \
    --initial-pool "${ACE_SQL_INITIAL_POOL_PATH:-$RL_ROOT/data/initial_pool.json}" \
    --output-dir "$ACE_SQL_SLIME_DATA" --tokenizer "$ACE_SQL_MODEL_PATH"
fi
