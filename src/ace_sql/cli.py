"""CLI for the package's data, training, runtime, and evaluation components."""
import argparse
import os
from pathlib import Path
import runpy
import sys

COMMANDS = {
    'data': 'ace_sql.data.rl',
    'train': 'ace_sql.training.launch',
    'prepare': 'ace_sql.runtime.prepare',
    'check-env': 'ace_sql.runtime.environment',
    'patch-slime': 'ace_sql.runtime.metadata_patch',
    'eval-bird': 'ace_sql.evaluation.bird',
    'eval-spider': 'ace_sql.evaluation.eval_spider',
    'eval-robustness': 'ace_sql.evaluation.spider_robustness_pipeline',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACE-SQL: slime training and shared-policy evaluation')
    parser.add_argument('--project', type=Path, help='Base directory for data, model and output paths')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.project:
        os.environ['ACE_SQL_HOME'] = str(args.project.expanduser().resolve())
    sys.argv = ['ace-sql ' + args.command, *args.arguments]
    runpy.run_module(COMMANDS[args.command], run_name='__main__')
