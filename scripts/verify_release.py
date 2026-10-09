#!/usr/bin/env python3
"""Check release completeness, immutable data hashes, and RL schema contracts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'RL'))
from ace_sql_slime.data import read_parquet_records, validate_splits


def main():
    manifest = json.loads((ROOT / 'docs/data_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f'Missing release data: {relative}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'Data fingerprint mismatch: {relative}')
    for name in ('domain1_alpaca_think.json', 'domain2_alpaca_think.json'):
        rows = json.loads((ROOT / 'SFT/data' / name).read_text())
        if len(rows) != 7092 or any(not {'instruction', 'input', 'output'} <= row.keys() for row in rows):
            raise ValueError(f'Unexpected SFT records: {name}')
    train = read_parquet_records(ROOT / 'RL/data/train.parquet')
    validation = read_parquet_records(ROOT / 'RL/data/validation.parquet')
    pool = json.loads((ROOT / 'RL/data/initial_pool.json').read_text())
    report = validate_splits(train, validation, pool)
    if (len(train), len(validation), len(pool)) != (2913, 40, 2953):
        raise ValueError('Unexpected released RL split sizes')
    for relative in ('RL/ace_sql_slime/core.py', 'RL/ace_sql_slime/loss.py', 'RL/ace_sql_slime/rollout.py',
                     'RL/scripts/train_rl_slime.sh', 'SFT/data/dataset_info.json'):
        if not (ROOT / relative).is_file():
            raise ValueError(f'Missing runtime file: {relative}')
    docs = [ROOT / 'README.md', ROOT / 'RL/README.md', ROOT / 'SFT/README.md',
            ROOT / 'EVAL/README.md', ROOT / 'audits/README.md', *sorted((ROOT / 'docs').glob('*.md'))]
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if '://' not in target and not target.startswith('#'):
                if not (doc.parent / target.split('#')[0]).exists():
                    raise ValueError(f'Broken local link in {doc.relative_to(ROOT)}: {target}')
    print(json.dumps({'status': 'passed', 'sft_rows': 14184, 'rl': report,
                      'data_sha256': 'all matched', 'local_document_links': 'all resolved'}, indent=2))


if __name__ == '__main__':
    main()
