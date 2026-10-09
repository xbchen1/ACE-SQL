#!/usr/bin/env bash
# Pinned upstream CUDA 12.9.1 runtime. Defaults to checking, not training.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SLIME_IMAGE='slimerl/slime@sha256:6e7748eb401c9bfbf99ca0e1d81a2f0772dd4400d8b860f71c09eaf0ee3f5120'
MODE=preflight
DRY_RUN=false
for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=true ;;
    preflight|prepare|train|shell) MODE="$argument" ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/container.sh [--dry-run] [preflight|prepare|train|shell]
Default: preflight on the pinned CUDA 12.9.1 container, without training.
Set ACE_SQL_MODEL_PATH, ACE_SQL_REFERENCE_PATH, ACE_SQL_SLIME_DATA,
ACE_SQL_SLIME_OUTPUT and optionally ACE_SQL_SLIME_RESUME to absolute host
directories. Existing directories outside the repo are mounted at the same
paths. Set ACE_SQL_ASSETS_DIR to mount a common data/database directory.
The wrapper installs only ACE-SQL package dependencies and the pinned slime source
inside the disposable container; it does not change the host Python/CUDA stack.
USAGE
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

COMMAND=(docker run --rm --platform linux/amd64 --gpus all --ipc=host --network=host
  --ulimit memlock=-1 --ulimit stack=67108864
  --mount "type=bind,src=${REPO_ROOT},dst=${REPO_ROOT}"
  --workdir "${REPO_ROOT}"
  --entrypoint /bin/bash)
if [[ "$MODE" == shell ]]; then
  if [[ -t 0 && -t 1 ]]; then COMMAND+=(-it); else COMMAND+=(-i); fi
fi

# Forward only this task's settings. Override interpreter/checkout with the
# container paths below, never a host Python executable or host CUDA install.
while IFS= read -r variable; do
  case "$variable" in
    ACE_SQL_PYTHON|ACE_SQL_SLIME_DIR|ACE_SQL_REPO_ROOT) ;;
    ACE_SQL_*) COMMAND+=(--env "$variable") ;;
  esac
done < <(compgen -e)
COMMAND+=(--env "ACE_SQL_REPO_ROOT=${REPO_ROOT}"
  --env ACE_SQL_PYTHON=python3 --env ACE_SQL_SLIME_DIR=/root/slime
  --env "ACE_SQL_EXPECTED_CUDA=${ACE_SQL_EXPECTED_CUDA:-12.9}")

MOUNTED=()
for variable in ACE_SQL_MODEL_PATH ACE_SQL_REFERENCE_PATH ACE_SQL_SLIME_DATA ACE_SQL_SLIME_OUTPUT ACE_SQL_SLIME_RESUME ACE_SQL_ASSETS_DIR ACE_SQL_TRAIN_DB_ROOT ACE_SQL_DEV_DB_ROOT; do
  directory="${!variable:-}"
  [[ -z "$directory" ]] && continue
  [[ "$directory" = /* ]] || { printf '%s must be an absolute directory\n' "$variable" >&2; exit 2; }
  case "$directory" in "$REPO_ROOT"|"$REPO_ROOT"/*) continue ;; esac
  duplicate=false
  for mounted in "${MOUNTED[@]-}"; do [[ "$directory" == "$mounted" ]] && duplicate=true; done
  "$duplicate" && continue
  if [[ ! -d "$directory" && "$DRY_RUN" == false ]]; then
    if [[ "$variable" =~ ^ACE_SQL_(SLIME_OUTPUT|REFERENCE_PATH|SLIME_DATA)$ && ( "$MODE" == train || "$MODE" == prepare ) ]]; then
      mkdir -p -- "$directory"
    else
      printf '%s directory does not exist: %s\n' "$variable" "$directory" >&2
      exit 2
    fi
  fi
  # The output/resume may share one directory. Mount exactly the requested path.
  COMMAND+=(--mount "type=bind,src=${directory},dst=${directory}")
  MOUNTED+=("$directory")
done

IFS= read -r -d '' BOOTSTRAP <<'CONTAINER' || true
set -euo pipefail
pin=6195417d45f272b72d04f619ea334613135d1c1f
git -C /root/slime fetch --depth 1 origin "$pin"
git -C /root/slime checkout --detach "$pin"
python3 -m pip install --no-deps --no-build-isolation -e /root/slime
python3 -m pip install --no-deps "numpy>=1.24,<2" pyarrow==25.0.1 sqlglot==30.21.0
python3 -m pip install --no-deps --no-build-isolation "$ACE_SQL_REPO_ROOT"
export PYTHONPATH="/root/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export ACE_SQL_HOME="$ACE_SQL_REPO_ROOT"
cd "$ACE_SQL_REPO_ROOT"
case "$1" in
  preflight)
    exec python3 -m ace_sql.runtime.environment --slime-dir /root/slime \
      --expected-cuda "${ACE_SQL_EXPECTED_CUDA:-12.9}" \
      --expected-gpus "${ACE_SQL_NUM_GPUS:-8}" --gpu-smoke ;;
  prepare) python3 -m ace_sql.runtime.prepare ;;
  train)
    python3 -m ace_sql.runtime.prepare
    exec python3 -m ace_sql.training.launch ;;
  shell) exec bash ;;
esac
CONTAINER
COMMAND+=("$SLIME_IMAGE" -lc "$BOOTSTRAP" ace-slime "$MODE")
if "$DRY_RUN"; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  exec "${COMMAND[@]}"
fi
