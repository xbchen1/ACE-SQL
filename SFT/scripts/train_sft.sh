#!/bin/bash
#SBATCH -J ace_sql_sft
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=400G

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export WANDB_MODE="${ACE_SQL_WANDB_MODE:-offline}"

echo "========== ACE-SQL SFT =========="
echo "Domain1: 7092 samples (think mode)"
echo "Domain2: 9309 samples (think mode)"
echo "Total: 16401 samples"
free -h
echo "================================================"

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4 llamafactory-cli train configs/sft_qwen3_8b.yaml
