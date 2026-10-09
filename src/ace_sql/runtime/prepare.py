"""Prepare the reference checkpoint and tokenizer-filtered data from one config."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ace_sql.training.config import load_config, project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=os.environ.get('ACE_SQL_CONFIG'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    settings = load_config(args.config)
    root = project_root()
    source = {name: Path(os.environ.get(env, root / 'data/rl' / file)).resolve()
              for name, env, file in [('train', 'ACE_SQL_TRAIN_FILE', 'train.parquet'),
                                      ('validation', 'ACE_SQL_VAL_FILE', 'validation.parquet'),
                                      ('initial_pool', 'ACE_SQL_INITIAL_POOL_PATH', 'initial_pool.json')]}
    tasks = [[sys.executable, '-m', 'ace_sql.runtime.environment', '--slime-dir', settings.slime,
              '--expected-cuda', settings.expected_cuda, '--expected-gpus', str(settings.gpus), '--gpu-smoke'],
             [sys.executable, '-m', 'ace_sql.runtime.metadata_patch', settings.slime]]
    reference = Path(settings.reference)
    if not (reference / 'latest_checkpointed_iteration.txt').is_file():
        if reference.exists() and any(reference.iterdir()):
            raise FileExistsError('Reference directory is nonempty but has no completed checkpoint')
        model_config = Path(settings.model) / 'config.json'
        if model_config.exists():
            model = json.loads(model_config.read_text())
            actual = {(36, 4096): '8B', (36, 2560): '4B'}.get((model.get('num_hidden_layers'), model.get('hidden_size')))
            if model.get('model_type') != 'qwen3' or actual != settings.model_size:
                raise ValueError('model_size does not match the supplied dense Qwen3 checkpoint')
        convert = 'source "$1"; shift; exec torchrun "$@" "${MODEL_ARGS[@]}"'
        tasks.append(['bash', '-c', convert, 'ace-sql-convert',
                      str(Path(settings.slime) / f'scripts/models/qwen3-{settings.model_size}.sh'),
                      '--standalone', f'--nproc_per_node={settings.gpus}',
                      str(Path(settings.slime) / 'tools/convert_hf_to_torch_dist.py'),
                      '--hf-checkpoint', settings.model, '--save', settings.reference])
    manifest_path = Path(settings.data) / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for key, file in source.items():
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if manifest['source_sha256'][key] != digest:
                raise ValueError('Prepared data uses different inputs; choose a fresh data directory')
        if not manifest.get('prompt_filter') or manifest['prompt_filter'].get('tokenizer') != settings.model:
            raise ValueError('Prepared data must be filtered with this model tokenizer; choose a fresh data directory')
    else:
        tasks.append([sys.executable, '-m', 'ace_sql.data.rl', '--train', str(source['train']),
                      '--validation', str(source['validation']), '--initial-pool', str(source['initial_pool']),
                      '--output-dir', settings.data, '--tokenizer', settings.model])
    if args.dry_run:
        print(json.dumps(tasks, indent=2))
        return
    for task in tasks:
        subprocess.run(task, check=True)


if __name__ == '__main__':
    main()
