"""BIRD pipeline for a shared-policy Hugging Face checkpoint exported by slime."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ace_sql.training.checkpoints import require_hf_checkpoint
from . import config


def command(module, **options):
    argv = [sys.executable, '-m', 'ace_sql.evaluation.' + module]
    for key, value in options.items():
        if value is True:
            argv.append('--' + key)
        elif value is not False and value is not None:
            argv.extend(['--' + key, str(value)])
    return argv


def build_stages(model, output, *, samples=1, temperature=0.0, merge='union', tensor_parallel=1, timeout=180):
    output = Path(output)
    ret_prompt, ret, merged, sql_prompt, sql = [output / name for name in (
        'retriever_prompts.json', 'retriever_outputs.json', 'retriever_merged.json', 'sql_prompts.json', 'sql_outputs.json')]
    infer = {'model_path': model, 'tensor_parallel_size': tensor_parallel, 'temperature': temperature,
             'num_samples': samples, 'max_tokens': 2048}
    return [
        (command('prepare_eval_data', dev_data_path=config.BIRD_DEV_DATA, tables_path=config.BIRD_TABLES,
                 retriever_prompt_path=config.RETRIEVER_PROMPT_PATH, output_path=ret_prompt,
                 db_path=config.BIRD_DB_PATH, db_content_index_path=config.BIRD_DB_CONTENT_INDEX_PATH,
                 value_limit_num=config.VALUE_LIMIT_NUM), ret_prompt),
        (command('retriever_infer', input_file=ret_prompt, output_file=ret, **infer), ret),
        (command('merge_retriever_outputs', input_file=ret, output_file=merged, mode=merge,
                 no_add_primary_keys=True), merged),
        (command('generate_sql_prompts', retriever_output_path=merged, db_path=config.BIRD_DB_PATH,
                 output_path=sql_prompt, value_limit_num=config.VALUE_LIMIT_NUM,
                 db_content_index_path=config.BIRD_DB_CONTENT_INDEX_PATH), sql_prompt),
        (command('sql_infer', input_file=sql_prompt, output_file=sql, **infer), sql),
        (command('evaluation_maj', pred=sql, gold=config.BIRD_DEV_DATA, db_path=config.BIRD_DB_PATH,
                 mode='greedy_search' if samples == 1 else 'major_voting', timeout=timeout), None),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Complete shared-policy HF export')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--merge', choices=['union', 'maj'], default='union')
    parser.add_argument('--tensor-parallel', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=int(os.environ.get('SQL_EVAL_TIMEOUT', '180')))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.samples < 1 or args.tensor_parallel < 1 or args.timeout < 1 or args.temperature < 0:
        parser.error('samples, tensor parallelism and timeout must be positive; temperature nonnegative')
    stages = build_stages(args.model, args.output, samples=args.samples, temperature=args.temperature,
                          merge=args.merge, tensor_parallel=args.tensor_parallel, timeout=args.timeout)
    if args.dry_run:
        print(json.dumps([cmd for cmd, _ in stages], indent=2))
        return
    model = require_hf_checkpoint(args.model)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty evaluation output directory to avoid mixing checkpoint predictions')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps({**vars(args), 'model': str(model)}, indent=2) + '\n')
    for index, (cmd, artifact) in enumerate(stages):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        (output / f'stage_{index}.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
        if artifact is not None and (not artifact.is_file() or not artifact.stat().st_size):
            raise RuntimeError(f'Evaluation stage did not produce {artifact}')
    match = re.search(r'EX Accuracy \((?:greedy search|major voting)\):\s*([\d.]+)', result.stdout)
    if not match:
        raise RuntimeError('Benchmark evaluator did not report execution accuracy; see final stage log')
    summary = {'execution_accuracy': float(match.group(1)), 'model': str(model),
               'samples': args.samples, 'temperature': args.temperature}
    (output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
