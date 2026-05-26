#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

pattern='(/roo''t/|/ho''me/|auto''dl|wandb_''v1|ACE-SQL-''main|Train''set|ace_sql_system_''data)'

if rg -n "${pattern}" \
    --glob '!*.parquet' \
    --glob '!.git/**' \
    --glob '!.run/**' \
    .; then
    echo "Found machine-specific absolute paths or secrets." >&2
    exit 1
fi

python3 - <<'PY'
from pathlib import Path

paths = [
    "data/train.parquet",
    "data/validation.parquet",
]
needles = (
    "/roo" + "t/",
    "/ho" + "me/",
    "auto" + "dl",
    "ACE-SQL-" + "main",
    "Train" + "set",
    "ace_sql_system_" + "data",
)
bad = []
try:
    import pandas as pd
except Exception:
    pd = None

if pd is not None:
    for path in paths:
        df = pd.read_parquet(path)
        for col in df.columns:
            values = df[col].astype(str)
            mask = values.apply(lambda value: any(needle in value for needle in needles))
            if mask.any():
                bad.append((path, col, int(mask.sum()), values[mask].iloc[0][:200]))
else:
    encoded_needles = [needle.encode() for needle in needles]
    for path in paths:
        blob = Path(path).read_bytes()
        if any(needle in blob for needle in encoded_needles):
            bad.append((path, "raw-bytes", 1, "machine-specific byte pattern"))
if bad:
    for path, col, count, sample in bad:
        print(f"{path}:{col}: {count} machine-specific values; sample={sample!r}")
    raise SystemExit(1)
PY

echo "Path check OK."
