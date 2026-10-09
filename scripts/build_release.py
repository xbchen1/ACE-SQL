#!/usr/bin/env python3
"""Build separate software/data ZIPs from Git's current file inventory.

Both archives use the same ACE-SQL/ prefix; extract them together for a complete
source snapshot. Untracked files must be staged before packaging. No .git,
models, databases, logs, local environments, or generated runs are included.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    'data/sft/domain1_alpaca_think.json', 'data/sft/domain2_alpaca_think.json',
    'data/sft/dataset_info.json', 'data/rl/train.parquet',
    'data/rl/validation.parquet', 'data/rl/initial_pool.json',
}
ALLOWED_ROOTS = {'src', 'configs', 'data', 'tests', 'assets', 'audits', 'docs', 'scripts'}
ROOT_FILES = {'README.md', 'LICENSE', '.gitignore', 'pyproject.toml', 'Makefile'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--anonymous', action='store_true', help='Omit author-identifying preprint/dataset badges')
    args = parser.parse_args()
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files = sorted(p for p in files if p and (p in ROOT_FILES or Path(p).parts[0] in ALLOWED_ROOTS))
    if any(Path(p).parts[0] in {'RL', 'SFT', 'EVAL'} for p in files):
        raise ValueError('Legacy source roots are not part of this release')
    if not DATA <= set(files) or 'src/ace_sql/rewards/engine.py' not in files:
        raise SystemExit('Stage the complete release, including both SFT datasets and the installed package sources, first.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 1, 'anonymous': args.anonymous, 'archives': {}}
    for category in ('software', 'data'):
        name = f'ACE-SQL-{category}.zip'
        entries = []
        with zipfile.ZipFile(args.output_dir / name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in files:
                if (relative in DATA) != (category == 'data'):
                    continue
                path = ROOT / relative
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f'Missing file or unsupported symlink: {relative}')
                content = path.read_bytes()
                if args.anonymous and relative == 'README.md':
                    text = content.decode()
                    text = '\n'.join(line for line in text.splitlines()
                                     if not line.startswith('[![arXiv]') and not line.startswith('[![Dataset]')) + '\n'
                    text = text.replace('The linked arXiv preprint and dataset release may precede this revision.',
                                        'This archive accompanies the October manuscript revision.')
                    content = text.encode()
                info = zipfile.ZipInfo('ACE-SQL/' + relative, date_time=(2026, 10, 9, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (path.stat().st_mode & 0o777) << 16
                archive.writestr(info, content)
                entries.append({'path': relative, 'sha256': hashlib.sha256(content).hexdigest()})
        blob = (args.output_dir / name).read_bytes()
        with zipfile.ZipFile(args.output_dir / name) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Archive integrity failure: {name}')
        manifest['archives'][name] = {'sha256': hashlib.sha256(blob).hexdigest(),
                                      'bytes': len(blob), 'files': entries}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({name: {'files': len(item['files']), 'bytes': item['bytes'], 'sha256': item['sha256']}
                      for name, item in manifest['archives'].items()}, indent=2))


if __name__ == '__main__':
    main()
