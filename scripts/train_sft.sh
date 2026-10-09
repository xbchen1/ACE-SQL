#!/bin/bash
#SBATCH -J ace_sql_sft
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=400G

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export WANDB_MODE="${ACE_SQL_WANDB_MODE:-offline}"


for file in data/sft/domain1_alpaca_think.json data/sft/domain2_alpaca_think.json; do
  if [[ ! -f "$file" ]]; then
    echo "Missing SFT data: $file" >&2
    exit 1
  fi
done

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="${ACE_SQL_NUM_GPUS:-8}" \
  llamafactory-cli train "${ACE_SQL_SFT_CONFIG:-configs/sft/sft_qwen3_8b.yaml}" "$@"
