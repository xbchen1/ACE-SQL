#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

MODEL_PATH="${MODEL_PATH:-models/retriever_and_sql_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/spider}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MERGE_MODE="${MERGE_MODE:-union}"
PROMPT_FORMAT="${PROMPT_FORMAT:-list}"
PIPELINE_MODE="${PIPELINE_MODE:-rag}"

export SPIDER_DATA_DIR="${SPIDER_DATA_DIR:-$PROJECT_DIR/data/spider}"
export OMNISQL_EVAL_DIR="${OMNISQL_EVAL_DIR:-$PROJECT_DIR/omnisql_eval}"

python3 eval_spider.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --merge_mode "$MERGE_MODE" \
  --prompt_format "$PROMPT_FORMAT" \
  --pipeline_mode "$PIPELINE_MODE"
