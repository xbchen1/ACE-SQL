"""Configuration, packaging boundaries, and CPU benchmark orchestration."""
import importlib.resources
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from unittest.mock import patch

import pytest
from ace_sql.training.config import load_config, TrainConfig
from ace_sql.training.launch import build_plan
from ace_sql.training.checkpoints import require_hf_checkpoint
from ace_sql.evaluation import bird
from ace_sql.evaluation.inference import format_result
from ace_sql.schema.retrieval import parse_retriever_output

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_precedence_and_paths(tmp_path):
    config = tmp_path / 'settings.toml'
    config.write_text('[paths]\noutput="experiment"\n[training]\nseed=17\n[method]\npool_gamma=0.2\n')
    result = load_config(config, root=tmp_path, environ={'ACE_SQL_SEED': '18', 'ACE_SQL_LENGTH_PENALTY': 'false'},
                         overrides={'seed': 19})
    assert result.seed == 19 and result.pool_gamma == 0.2 and not result.length_penalty
    assert result.output == str(tmp_path / 'experiment')
    plan = build_plan(result)
    assert plan['steps'] == 910 and plan['trajectories_per_step'] == 256
    assert plan['custom_config']['ace_pool_gamma'] == 0.2
    assert 'ace_sql.training.rollout.generate_rollout' in plan['command']


@pytest.mark.parametrize('values', [{'gpus': 3}, {'questions': True}, {'pool_gamma': float('nan')},
                                   {'temperature': True}, {'learning_rate': 0}, {'prompt_tokens': 8192},
                                   {'output': 3}, {'length_penalty': 'false'}])
def test_invalid_configurations_fail_before_launch(values):
    with pytest.raises(ValueError):
        TrainConfig(**values).validate()


def test_named_controls_and_unknown_settings(tmp_path):
    for name, field, expected in [('frozen', 'target_mode', 'frozen'), ('gold', 'target_mode', 'gold'),
                                  ('dense', 'target_mode', 'dense'), ('current-only', 'pool_gamma', 0),
                                  ('no-decay', 'pool_gamma', 1), ('no-schedule', 'ramp_ratio', 0),
                                  ('long-schedule', 'ramp_ratio', .5), ('no-length-penalty', 'length_penalty', False)]:
        config = load_config(ROOT / f'configs/slime/{name}.toml', environ={}, root=tmp_path)
        assert getattr(config, field) == expected
        assert config.output != TrainConfig().resolved(tmp_path).output
    bad = tmp_path / 'bad.toml'
    bad.write_text('[method]\npool_gama=0.3\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_config(bad, environ={})


def test_package_resources_and_cli_outside_checkout(tmp_path):
    for package, resource in [('ace_sql.schema', 'templates/retriever_prompt.txt'),
                              ('ace_sql.runtime', 'patches/slime-v0.2.3-ace-metadata.patch'),
                              ('ace_sql.evaluation.omnisql_eval.test_suite_sql_eval', 'LICENSE')]:
        assert importlib.resources.files(package).joinpath(resource).read_text().strip()
    result = subprocess.run([sys.executable, '-m', 'ace_sql', '--project', str(tmp_path), 'train', '--dry-run'],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    assert str(tmp_path / 'models/sft_checkpoint') in result.stdout
    assert 'ace_sql.training.loss.policy_loss' in result.stdout
    assert not list(tmp_path.iterdir())


def test_complete_hf_boundary_rejects_missing_and_external_shards(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='Incomplete'):
        require_hf_checkpoint(tmp_path)
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
    (tmp_path / 'part1.safetensors').write_bytes(b'a')
    with pytest.raises(ValueError, match='Incomplete'):
        require_hf_checkpoint(tmp_path)
    (tmp_path / 'part2.safetensors').write_bytes(b'b')
    assert require_hf_checkpoint(tmp_path) == tmp_path
    index.write_text(json.dumps({'weight_map': {'a': '../escaped.safetensors'}}))
    with pytest.raises(ValueError, match='outside'):
        require_hf_checkpoint(tmp_path)


def test_bird_stages_use_one_model_and_packaged_modules(tmp_path):
    stages = bird.build_stages('/shared/hf', tmp_path)
    inference = [cmd for cmd, _ in stages if '--model_path' in cmd]
    assert len(inference) == 2
    for cmd in inference:
        assert cmd[cmd.index('--model_path') + 1] == '/shared/hf'
        assert cmd[cmd.index('--num_samples') + 1] == '1'
    for cmd, _ in stages:
        assert cmd[:2] == [sys.executable, '-m']
        result = subprocess.run(cmd + ['--help'], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_shared_parsing_keeps_final_selection_and_no_schema_fallback():
    schema = {'table_names_original': ['t'], 'column_names_original': [[-1, '*'], [0, 'a'], [0, 'b']]}
    assert parse_retriever_output('<think>[t.b]</think> [t.a]', schema) == {1}
    assert parse_retriever_output('{"t": ["a"]}', schema) == {1}
    assert parse_retriever_output('[]', schema) == set()
    assert format_result({'db_id': 'toy'}, ['```sql\nSELECT a FROM t\n```'], 'generator')['pred_sqls'] == ['SELECT a FROM t']


def test_bird_pipeline_with_real_sqlite_and_mocked_model_only(tmp_path):
    model, output, db_root = tmp_path / 'model', tmp_path / 'output', tmp_path / 'databases'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'fixture-only')
    (db_root / 'toy').mkdir(parents=True)
    with sqlite3.connect(db_root / 'toy/toy.sqlite') as connection:
        connection.execute('CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)')
        connection.execute("INSERT INTO t VALUES (1, 'x')")
    gold, tables = tmp_path / 'dev.json', tmp_path / 'tables.json'
    gold.write_text(json.dumps([{'question_id': 0, 'db_id': 'toy', 'question': 'List a', 'SQL': 'SELECT a FROM t', 'difficulty': 'simple'}]))
    columns = [[-1, '*'], [0, 'a'], [0, 'b']]
    tables.write_text(json.dumps([{'db_id': 'toy', 'table_names_original': ['t'], 'table_names': ['t'],
                                 'column_names_original': columns, 'column_names': columns,
                                 'column_types': ['text', 'number', 'text'], 'primary_keys': [1], 'foreign_keys': []}]))
    actual_run = subprocess.run
    def run(cmd, **kwargs):
        if cmd[2].endswith(('retriever_infer', 'sql_infer')):
            data = json.loads(Path(cmd[cmd.index('--input_file') + 1]).read_text())
            role = 'retriever' if cmd[2].endswith('retriever_infer') else 'generator'
            response = '<think>columns</think> [t.a]' if role == 'retriever' else '```sql\nSELECT a FROM t\n```'
            Path(cmd[cmd.index('--output_file') + 1]).write_text(json.dumps([format_result(row, [response], role) for row in data]))
            return subprocess.CompletedProcess(cmd, 0, 'mocked model generation only\n', '')
        if cmd[2].endswith('evaluation_maj'):
            cmd = cmd + ['--num_cpus', '1']
        return actual_run(cmd, **kwargs)
    with patch.multiple(bird.config, BIRD_DEV_DATA=str(gold), BIRD_TABLES=str(tables), BIRD_DB_PATH=str(db_root),
                        BIRD_DB_CONTENT_INDEX_PATH=str(tmp_path/'no-index')), patch.object(bird.subprocess, 'run', run):
        bird.main(['--model', str(model), '--output', str(output)])
    result = json.loads((output / 'result.json').read_text())
    assert result['execution_accuracy'] == 1.0
    assert len(list(output.glob('stage_*.log'))) == 6
    prompts = json.loads((output / 'sql_prompts.json').read_text())
    assert prompts[0]['pred_indices'] == [1] if 'pred_indices' in prompts[0] else 'a' in prompts[0]['selected_schema']
    with pytest.raises(FileExistsError):
        bird.main(['--model', str(model), '--output', str(output)])


def test_packaged_spider_evaluator_runs_outside_checkout(tmp_path):
    from ace_sql.evaluation.omnisql_eval.evaluate_spider import run_spider_eval
    (tmp_path / 'toy').mkdir()
    with sqlite3.connect(tmp_path / 'toy/toy.sqlite') as connection:
        connection.execute('CREATE TABLE t (a INTEGER)')
        connection.execute('INSERT INTO t VALUES (1)')
    gold, pred = tmp_path / 'gold.sql', tmp_path / 'pred.json'
    gold.write_text('SELECT a FROM t\ttoy\n')
    pred.write_text('[{"pred_sqls": ["SELECT a FROM t"]}]')
    with patch.dict(os.environ, {'SQL_VALIDATION_TIMEOUT': '5'}):
        ex, ts = run_spider_eval(str(gold), str(pred), str(tmp_path), '', 'greedy_search', False)
    assert ex == 1.0 and ts is None


def test_robustness_dry_run_uses_two_stage_pipeline(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'ace_sql', 'eval-robustness',
                             '--data_dir', str(tmp_path / 'input'), '--datasets', 'spider_syn', '--dry_run',
                             'run', '--model', str(tmp_path / 'hf'), '--output_dir', str(tmp_path / 'out')],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['command'][2] == 'ace_sql.evaluation.eval_spider'
    assert plan['environment']['SPIDER_DB_PATH'] == str(tmp_path / 'input/spider/database')
    assert not list(tmp_path.iterdir())
