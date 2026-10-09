#!/usr/bin/env bash
set -euo pipefail
# Current manuscript entrypoint: shared-policy GRPO on slime, without projection.
export ACE_SQL_GRAD_PROJ_ENABLED=False
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_rl_slime.sh" "$@"
