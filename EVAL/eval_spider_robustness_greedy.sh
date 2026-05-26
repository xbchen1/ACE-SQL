#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

OMNI_DIR="${OMNI_DIR:-$PROJECT_DIR/omnisql_eval}"
DATA_DIR="${DATA_DIR:-$OMNI_DIR/data}"
PIPELINE_PY="$PROJECT_DIR/spider_robustness_pipeline.py"

MODEL_PATH="${MODEL_PATH:-models/sql_generator_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/spider_robustness_greedy}"
DATASETS="${DATASETS:-spider_dk,spider_syn,spider_realistic}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

DOWNLOAD_TEST_SUITE="${DOWNLOAD_TEST_SUITE:-false}"
REQUIRE_TEST_SUITE="${REQUIRE_TEST_SUITE:-false}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-60}"
SQL_VALIDATION_TIMEOUT="${SQL_VALIDATION_TIMEOUT:-300}"
VALUE_LIMIT="${VALUE_LIMIT:-2}"
FORCE="${FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
RUN_EVAL="${RUN_EVAL:-true}"
BUILD_INDEXES="${BUILD_INDEXES:-true}"
RUN_PROCESS="${RUN_PROCESS:-true}"
TEST_SUITE_DB_PATH="${TEST_SUITE_DB_PATH:-}"

export OMNI_DIR DATA_DIR TEST_SUITE_DB_PATH SQL_VALIDATION_TIMEOUT
if [[ -n "${JAVA_HOME:-}" ]]; then
  export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
  export PATH="$JAVA_HOME/bin:$PATH"
fi

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Error: missing $label: $path" >&2
    exit 1
  fi
}

run_cmd() {
  printf '\nRun:'
  printf ' %q' "$@"
  printf '\n'
  if is_true "$DRY_RUN"; then
    return 0
  fi
  "$@"
}

dataset_processed_file() {
  case "$1" in
    spider_dk) echo "$DATA_DIR/dev_spider_dk.json" ;;
    spider_syn) echo "$DATA_DIR/dev_spider_syn.json" ;;
    spider_realistic) echo "$DATA_DIR/dev_spider_realistic.json" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$OUTPUT_DIR"

require_path "$PIPELINE_PY" "pipeline helper"
require_path "$OMNI_DIR/process_dataset.py" "process_dataset.py"
require_path "$OMNI_DIR/infer.py" "infer.py"
require_path "$OMNI_DIR/evaluate_spider.py" "evaluate_spider.py"

prepare_args=(
  python3 "$PIPELINE_PY"
  --omni_dir "$OMNI_DIR"
  --data_dir "$DATA_DIR"
  --datasets "$DATASETS"
)
if is_true "$DRY_RUN"; then
  prepare_args+=(--dry_run)
fi
prepare_args+=(prepare-data --download_timeout "$DOWNLOAD_TIMEOUT")
if is_true "$DOWNLOAD_TEST_SUITE"; then
  prepare_args+=(--download_test_suite)
fi
if is_true "$REQUIRE_TEST_SUITE"; then
  prepare_args+=(--require_test_suite)
fi
run_cmd "${prepare_args[@]}"

if is_true "$BUILD_INDEXES"; then
  index_args=(python3 "$PIPELINE_PY" --omni_dir "$OMNI_DIR" --data_dir "$DATA_DIR" --datasets "$DATASETS")
  if is_true "$DRY_RUN"; then
    index_args+=(--dry_run)
  fi
  index_args+=(build-indexes)
  if is_true "$FORCE"; then
    index_args+=(--force)
  fi
  run_cmd "${index_args[@]}"
fi

if is_true "$RUN_PROCESS"; then
  process_args=(python3 "$PIPELINE_PY" --omni_dir "$OMNI_DIR" --data_dir "$DATA_DIR" --datasets "$DATASETS")
  if is_true "$DRY_RUN"; then
    process_args+=(--dry_run)
  fi
  process_args+=(process --value_limit "$VALUE_LIMIT")
  if is_true "$FORCE"; then
    process_args+=(--force)
  fi
  run_cmd "${process_args[@]}"
fi

if is_true "$PREPARE_ONLY"; then
  echo "PREPARE_ONLY=true; stop before GPU inference."
  exit 0
fi

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
  dataset="${dataset// /}"
  [[ -n "$dataset" ]] || continue

  input_file="$(dataset_processed_file "$dataset")"
  dataset_out="$OUTPUT_DIR/$dataset"
  pred_file="$dataset_out/greedy_search_.json"
  result_file="$dataset_out/result_greedy.json"
  mkdir -p "$dataset_out"

  require_path "$input_file" "$dataset processed prompt file"

  if ! is_true "$SKIP_INFERENCE"; then
    if is_true "$FORCE" || [[ ! -f "$pred_file" ]]; then
      run_cmd python3 "$OMNI_DIR/infer.py" \
        --pretrained_model_name_or_path "$MODEL_PATH" \
        --input_file "$input_file" \
        --output_file "$pred_file" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --n "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE"
    else
      echo "Skip existing greedy predictions: $pred_file"
    fi
  fi

  require_path "$pred_file" "$dataset greedy predictions"

  if is_true "$RUN_EVAL"; then
    eval_args=(
      python3 "$PIPELINE_PY"
      --omni_dir "$OMNI_DIR"
      --data_dir "$DATA_DIR"
      --datasets "$DATASETS"
      evaluate
      --dataset "$dataset"
      --pred "$pred_file"
      --output_json "$result_file"
    )
    if [[ -n "$TEST_SUITE_DB_PATH" ]]; then
      eval_args+=(--test_suite_db_path "$TEST_SUITE_DB_PATH")
    fi
    if is_true "$REQUIRE_TEST_SUITE"; then
      eval_args+=(--require_test_suite)
    fi
    run_cmd "${eval_args[@]}"
  fi
done

if is_true "$RUN_EVAL"; then
  run_cmd python3 "$PIPELINE_PY" \
    --omni_dir "$OMNI_DIR" \
    --data_dir "$DATA_DIR" \
    --datasets "$DATASETS" \
    collect-results \
    --output_dir "$OUTPUT_DIR"
fi
