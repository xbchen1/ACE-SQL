# ACE-SQL RL

This directory contains the reinforcement-learning stage for ACE-SQL: joint two-pass GRPO over schema retrieval and SQL generation, with PCGrad enabled for the shared-policy update.

Included runtime data:

- `data/train.parquet`: 2,913 RL training rows
- `data/initial_pool.json`: empirical pool entries for training and validation examples
- `data/validation.parquet`: 40 validation rows
- `data/train_summary.json`: construction summary for the curated RL split

Not included:

- model weights
- SQLite databases
- W&B runs
- checkpoints, Ray temporary files, object spill files, or generated outputs

## Run

Place or link the SFT checkpoint at `models/sft_checkpoint`, or set `ACE_SQL_MODEL_PATH` to another model directory.

Database mounts are expected under `external/train_databases`, `external/dev_databases`, or `external/databases`. You can override them with `ACE_SQL_TRAIN_DB_ROOT`, `ACE_SQL_DEV_DB_ROOT`, and `ACE_SQL_LOOSE_DB_ROOTS`.

```bash
bash scripts/train_rl_pcgrad.sh
```

PCGrad is enabled by default:

```bash
ACE_SQL_GRAD_PROJ_MODE=symmetric bash scripts/train_rl_pcgrad.sh
```

Supported projection modes in the bundled update path are `symmetric`, `generator_dominant`, and `norm_equalized`.

## Checks

```bash
make syntax
make check-paths
```

The training script writes checkpoints and temporary files under `.run/` by default.
