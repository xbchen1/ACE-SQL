#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

MODEL_PATH="${MODEL_PATH:-models/retriever_or_merged_model}"
SQL_MODEL="${SQL_MODEL:-models/sql_generator_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/bird_maj}"

NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RETRIEVER_MERGE_MODE="${RETRIEVER_MERGE_MODE:-union}"
ADD_PRIMARY_KEYS="${ADD_PRIMARY_KEYS:-false}"
RETRIEVER_PROMPT_FORMAT="${RETRIEVER_PROMPT_FORMAT:-list}"
NO_THINK="${NO_THINK:-false}"
SQL_PROMPTS_FILE="${SQL_PROMPTS_FILE:-}"

export BIRD_DEV_DATA="${BIRD_DEV_DATA:-$PROJECT_DIR/data/bird_dev/dev.json}"
export BIRD_TABLES="${BIRD_TABLES:-$PROJECT_DIR/data/bird_dev/dev_tables.json}"
export BIRD_DB_PATH="${BIRD_DB_PATH:-$PROJECT_DIR/data/bird_dev/dev_databases}"
export BIRD_DB_CONTENT_INDEX_PATH="${BIRD_DB_CONTENT_INDEX_PATH:-$PROJECT_DIR/data/bird_dev/db_contents_index}"

cmd=(
  python3 single_eval_maj.py
  --checkpoint_path "$MODEL_PATH"
  --sql_model "$SQL_MODEL"
  --skip_merge
  --num_samples "$NUM_SAMPLES"
  --temperature "$TEMPERATURE"
  --retriever_merge_mode "$RETRIEVER_MERGE_MODE"
  --retriever_prompt_format "$RETRIEVER_PROMPT_FORMAT"
  --output_dir "$OUTPUT_DIR"
)

if [[ "$ADD_PRIMARY_KEYS" == "false" ]]; then
  cmd+=(--no_add_primary_keys)
fi
if [[ "$NO_THINK" == "true" ]]; then
  cmd+=(--no_think)
fi
if [[ -n "$SQL_PROMPTS_FILE" ]]; then
  cmd+=(--sql_prompts_file "$SQL_PROMPTS_FILE")
fi

printf 'Run:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if [[ -f "$OUTPUT_DIR/sql_outputs_maj.json" ]]; then
  python3 scripts/analyze_column_coverage.py \
    --input "$OUTPUT_DIR/sql_outputs_maj.json" \
    --tables "$BIRD_TABLES" \
    --db_path "$BIRD_DB_PATH" \
    --output "$OUTPUT_DIR/column_coverage_analysis.json"
fi

if [[ -f "$OUTPUT_DIR/retriever_outputs_merged.json" ]]; then
  python3 scripts/analyze_retriever_coverage.py \
    --input "$OUTPUT_DIR/retriever_outputs_merged.json" \
    --output "$OUTPUT_DIR/retriever_coverage_analysis.json"
fi
