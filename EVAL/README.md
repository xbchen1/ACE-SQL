# Text-to-SQL Evaluation Framework

This package contains the core code for three evaluation pipelines:

- `eval_bird_maj.sh`: BIRD dev evaluation with schema retrieval, SQL generation, execution evaluation, and optional coverage analysis.
- `eval_spider.sh`: Spider dev/test greedy evaluation, with either RAG schema selection or full-schema prompting.
- `eval_spider_robustness_greedy.sh`: Spider-DK, Spider-Syn, and Spider-Realistic greedy evaluation using the included OmniSQL-compatible helpers.

Datasets, databases, model checkpoints, generated predictions, logs, and local cluster files are intentionally not included.

## Setup

```bash
pip install -r requirements.txt
```

Set paths through environment variables. See `examples/env.example` for the common variables.

## Run

BIRD greedy evaluation:

```bash
MODEL_PATH=/path/to/retriever_or_merged_model \
SQL_MODEL=/path/to/sql_generator_model \
BIRD_DEV_DATA=/path/to/bird/dev.json \
BIRD_TABLES=/path/to/bird/dev_tables.json \
BIRD_DB_PATH=/path/to/bird/dev_databases \
bash eval_bird_maj.sh
```

Spider dev/test:

```bash
MODEL_PATH=/path/to/model \
SPIDER_DATA_DIR=/path/to/spider_data \
bash eval_spider.sh
```

Spider robustness:

```bash
MODEL_PATH=/path/to/model \
SPIDER_SOURCE=/path/to/spider_data \
bash eval_spider_robustness_greedy.sh
```

For Spider robustness, `prepare-data` can download Spider-DK, Spider-Syn, and Spider-Realistic metadata, but you still need the Spider database files. Test-suite databases are optional unless you set `REQUIRE_TEST_SUITE=true`.

## Notes

- The scripts are plain Bash wrappers, not cluster-specific job scripts. Submit them with your own scheduler wrapper if needed.
- Evaluation defaults use greedy decoding: `NUM_SAMPLES=1` and `TEMPERATURE=0.0`.
- Default paths are relative placeholders under this package. Use environment variables for real datasets and models.
- `omnisql_eval/test_suite_sql_eval/` contains only the evaluator code; it does not include test-suite databases.
