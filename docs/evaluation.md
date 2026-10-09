# Shared-policy benchmark evaluation

Evaluation consumes a complete Hugging Face checkpoint exported by slime, typically `RUN/hf/rollout_ID`. The same checkpoint performs retrieval and SQL generation. There is no FSDP state-dict merger or LoRA mounting step. `training.checkpoints` rejects missing/empty weights, missing indexed shards and weight indices that point outside the export directory.

Use a separate compatible vLLM environment and install `python -m pip install -e '.[evaluation]'`. Dataset paths can be supplied through [evaluation.env.example](../configs/evaluation.env.example). Prompt resources come from the installed package; outputs belong in the project or a user-specified directory.

## BIRD

```bash
export BIRD_DEV_DATA=/path/to/bird/dev.json
export BIRD_TABLES=/path/to/bird/dev_tables.json
export BIRD_DB_PATH=/path/to/bird/dev_databases
export BIRD_DB_CONTENT_INDEX_PATH=/path/to/bird/db_contents_index

ace-sql eval-bird --model /path/to/run/hf/rollout_80 --output .runs/bird_80 --dry-run
ace-sql eval-bird --model /path/to/run/hf/rollout_80 --output .runs/bird_80
```

The pipeline prepares retrieval prompts, runs retrieval, unions selected columns, prepares SQL prompts, runs generation, and applies BIRD execution evaluation. It writes each intermediate JSON, six stage logs, `protocol.json` and `result.json`. A fresh output directory is required for each checkpoint. Failures preserve the corresponding stage log.

Defaults are one sample per role, temperature zero, 2,048 response tokens and 180 seconds per evaluated SQL. `result.json.execution_accuracy` is a fraction in [0, 1]. Sampling/majority evaluation is explicit via `--samples`, `--temperature` and `--merge`; it is not the manuscript greedy protocol. `--tensor-parallel` controls inference GPU count.

Lucene value retrieval is optional: if no index is present, prompt construction uses sampled database values and reports the skipped index. Indexed retrieval needs the Pyserini/JVM environment and NLTK tokenizer resources appropriate for the installed versions. Record this choice when comparing runs.

## Spider dev/test

```bash
export SPIDER_DATA_DIR=/path/to/spider_data
ace-sql eval-spider --model_path /path/to/run/hf/rollout_80 \
  --output_dir .runs/spider_dev_80 --split dev
ace-sql eval-spider --model_path /path/to/run/hf/rollout_80 \
  --output_dir .runs/spider_test_80 --split test
```

Greedy retrieval and generation use the same shared inference implementation as BIRD. `--pipeline_mode full_schema` is an explicit SQL-only evaluation control. `--skip_inference` evaluates existing prediction files; otherwise use an empty output directory. The bundled test-suite evaluator runs as an installed Python module. An external evaluator can be selected explicitly with `TEST_SUITE_EVAL_DIR`.

## Spider-DK, Spider-Syn and Spider-Realistic

Prepare benchmark metadata and links to the supplied Spider databases, then run the shared-policy two-stage pipeline:

```bash
export SPIDER_SOURCE=/path/to/spider_data
ace-sql eval-robustness prepare-data
ace-sql eval-robustness run --model /path/to/run/hf/rollout_80 \
  --output_dir .runs/robustness_80
```

Preparation may download benchmark metadata; SQLite databases must be supplied separately. Work files default to `.runs/robustness`, never site-packages. `--data_dir` and `--datasets` are global options, before the subcommand. Spider-Syn uses its perturbed `SpiderSynQuestion` text. Every dataset is evaluated through the same retrieval/generation path as Spider, with per-dataset results and `summary_greedy.json`.

```bash
ace-sql eval-robustness --data_dir /assets/robustness \
  --datasets spider_syn --dry_run run --model /path/to/hf --output_dir .runs/syn
```

The retained `build-indexes`, `process`, `evaluate` and `collect-results` helpers support separately prepared baseline/analysis predictions. Their processed full-schema prompts are not inputs to the two-stage `run` command. Test-suite databases are optional and are not distributed here.

## Evidence and execution rules

BIRD retains set-based benchmark matching; Spider retains its official evaluator behavior. Training rewards separately use ordered rows/columns and a 30-second SQLite limit. CPU tests execute the BIRD prompt/merge/SQL/scorer stages on a small database with only model output mocked, and run the packaged Spider evaluator on a fixture. This verifies wiring and package boundaries, not real-model accuracy or GPU inference compatibility.
