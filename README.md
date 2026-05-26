# ACE-SQL

This repository contains the code and curated data for ACE-SQL, a two-stage Text-to-SQL training recipe that first builds a supervised retriever-generator cold start and then jointly optimizes both roles with reinforcement learning. It also includes the evaluation scripts used for BIRD and Spider-style benchmarks.

## Repository Layout

- `SFT/`: supervised fine-tuning cold start built with LLaMA-Factory.
- `RL/`: joint two-pass GRPO training for schema retrieval and SQL generation, with PCGrad support.
- `EVAL/`: BIRD, Spider, and Spider robustness evaluation framework.

The repository is organized so each stage can be used independently after local paths are configured.

## What Is Included

SFT data:

- `SFT/data/domain1_alpaca_think.json`: 7,092 retriever-format samples.
- `SFT/data/domain2_alpaca_think.json`: 7,092 generator-format samples.
- `SFT/data/dataset_info.json`: LLaMA-Factory dataset registry.

RL data:

- `RL/data/train.parquet`: 2,913 RL training rows.
- `RL/data/validation.parquet`: 40 validation rows.
- `RL/data/initial_pool.json`: empirical column-set pool initialized from SFT rollouts.
- `RL/data/train_summary.json`: construction summary for the curated RL split.

Evaluation code:

- BIRD greedy evaluation with retrieval, SQL generation, execution scoring, and coverage analysis.
- Spider dev/test evaluation with RAG schema selection or full-schema prompting.
- Spider-DK, Spider-Syn, and Spider-Realistic greedy evaluation helpers.

## What Is Not Included

The release intentionally excludes large or machine-specific assets:

- base model weights and fine-tuned checkpoints
- benchmark SQLite databases
- generated predictions, logs, W&B runs, and cluster output files
- Ray temporary files, object spill files, and training checkpoints

Place local model files under the stage-specific `models/` directories, place local databases under `RL/external/` or `EVAL/data/`, or override paths with the environment variables described below and in the stage READMEs.

## Setup

Use separate environments for the three stages if possible, since SFT, RL, and vLLM evaluation often require different CUDA/package stacks.

For SFT, install LLaMA-Factory and the dependencies expected by your local training environment. The bundled config uses Qwen3-8B by default.

For RL:

```bash
cd RL
pip install -r requirements.txt
```

For evaluation:

```bash
cd EVAL
pip install -r requirements.txt
```

## Supervised Fine-Tuning

The SFT stage initializes the shared policy for both roles. Place or link the base model at `SFT/models/Qwen3-8B`, or edit `SFT/configs/sft_qwen3_8b.yaml`.

```bash
cd SFT
bash scripts/train_sft.sh
```

The script runs full-parameter SFT with DeepSpeed ZeRO-3, four GPUs, and offline W&B mode by default. See `SFT/README.md` for the exact data files and stage notes.

## Reinforcement Learning

The RL stage starts from the SFT checkpoint and performs joint two-pass GRPO over retriever and generator outputs. PCGrad is enabled in the bundled update path.

Place or link the SFT checkpoint at `RL/models/sft_checkpoint`, or set:

```bash
export ACE_SQL_MODEL_PATH=/path/to/sft_checkpoint
```

Database mounts are expected under one of:

- `RL/external/train_databases`
- `RL/external/dev_databases`
- `RL/external/databases`

You can override them with:

```bash
export ACE_SQL_TRAIN_DB_ROOT=/path/to/train_databases
export ACE_SQL_DEV_DB_ROOT=/path/to/dev_databases
export ACE_SQL_LOOSE_DB_ROOTS=/path/to/other_databases
```

Run training:

```bash
cd RL
bash scripts/train_rl_pcgrad.sh
```

Useful lightweight checks:

```bash
cd RL
make syntax
make check-paths
```

Supported gradient projection modes are `symmetric`, `generator_dominant`, and `norm_equalized`:

```bash
ACE_SQL_GRAD_PROJ_MODE=symmetric bash scripts/train_rl_pcgrad.sh
```

## Evaluation

Evaluation defaults use greedy decoding: `NUM_SAMPLES=1` and `TEMPERATURE=0.0`.

BIRD:

```bash
cd EVAL
MODEL_PATH=/path/to/retriever_or_merged_model \
SQL_MODEL=/path/to/sql_generator_model \
BIRD_DEV_DATA=/path/to/bird/dev.json \
BIRD_TABLES=/path/to/bird/dev_tables.json \
BIRD_DB_PATH=/path/to/bird/dev_databases \
bash eval_bird_maj.sh
```

Spider dev/test:

```bash
cd EVAL
MODEL_PATH=/path/to/model \
SPIDER_DATA_DIR=/path/to/spider_data \
bash eval_spider.sh
```

Spider robustness:

```bash
cd EVAL
MODEL_PATH=/path/to/model \
SPIDER_SOURCE=/path/to/spider_data \
bash eval_spider_robustness_greedy.sh
```

See `EVAL/examples/env.example` for the common path variables.

## Reproducing The Pipeline

A typical end-to-end workflow is:

1. Run SFT with the balanced retriever/generator data in `SFT/data/`.
2. Use the resulting checkpoint as `RL/models/sft_checkpoint`.
3. Mount the SynSQL/BIRD/Spider SQLite databases locally.
4. Run RL with `RL/scripts/train_rl_pcgrad.sh`.
5. Evaluate checkpoints with the scripts in `EVAL/`.

The exact model checkpoints and external benchmark databases are intentionally not part of the repository, so absolute scores require the same local assets used in the paper experiments.

## Notes For Reviewers

- The SFT split is balanced across roles: 7,092 retriever samples and 7,092 generator samples.
- The RL split contains 2,913 hard question-database pairs plus the empirical pool used for retriever credit assignment.
- The evaluation scripts are plain Bash wrappers and can be submitted through a local scheduler if needed.
- Local outputs are ignored by `.gitignore`; generated checkpoints and predictions should not be committed.

