#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${RL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE="${ACE_SQL_WANDB_MODE:-offline}"
cd "${RL_ROOT}"
exec "${ACE_SQL_PYTHON:-python3}" -m ace_sql_slime.launch "$@"
