# Joint GRPO on slime

Install the package from the repository root. Both roles share one full-parameter Qwen3 policy; slime owns the optimizer, Megatron actor/reference models and SGLang serving workers. See [architecture](architecture.md) and the [pinned runtime](runtime.md).

## Configuration and preparation

`configs/slime/paper.toml` is the full paper recipe. Other TOMLs override method defaults and choose distinct output directories. Precedence is defaults → TOML → explicit `ACE_SQL_*` variables → CLI `--seed`, `--output`, `--resume`. Unknown settings and invalid values fail before launch. Relative asset paths resolve against `--project`, `ACE_SQL_HOME`, or the working directory; the configuration filename itself is relative to the invoking directory.

```bash
export ACE_SQL_CONFIG="$PWD/configs/slime/paper.toml"
export ACE_SQL_MODEL_PATH=/assets/models/sft_checkpoint
export ACE_SQL_REFERENCE_PATH=/assets/models/sft_torch_dist
export ACE_SQL_TRAIN_DB_ROOT=/assets/databases/train
export ACE_SQL_DEV_DB_ROOT=/assets/databases/dev

ace-sql prepare --dry-run
ace-sql train --dry-run
bash scripts/container.sh preflight
bash scripts/container.sh prepare
bash scripts/container.sh train
```

Preparation checks the GPU environment, applies the metadata patch, converts the SFT checkpoint to Megatron reference format, and converts the included data using that checkpoint's tokenizer. Existing prepared data must match the source hashes and tokenizer; otherwise use a new data directory. For 4B, select `model_size="4B"` in a TOML copy and supply Qwen3-4B weights. `ACE_SQL_ASSETS_DIR` mounts a common external asset directory in the container.

Inside an already prepared compatible runtime, `ace-sql prepare` and `ace-sql train` also work directly. Do not install a second GPU dependency stack over the pinned container.

## Paper recipe

| Setting | Value |
| --- | --- |
| Policy / hardware | Shared Qwen3-8B or 4B, full parameters; 8 H20 GPUs |
| Samples | 8 retrievals → union schema → 8 SQLs per question |
| Pool | `0.5 * historical counts + current matched SQL counts`; update before target selection |
| Targets | All tied highest-frequency column sets accepted |
| Advantages | Per question and role, unbiased sample std plus `1e-6` |
| Role loss | `(1-lambda/2) * L_ret + (lambda/2) * L_gen` |
| Schedule | Lambda 0 → 1 over first 25% of updates |
| Optimizer | Adam, LR `1e-6`, 5% warmup, 5 epochs |
| KL / PPO clip | Frozen-reference KL `0.001`, clip `0.2` |
| Batch | 16 questions, 256 trajectories per update |
| Length / sampling | 4,096 prompt + 2,048 response tokens; temperature 1 |
| Validation / save | Every 40 / 80 steps and final checkpoint |

Without tokenizer filtering, 2,913 questions with `drop_last` give 910 updates over five epochs. The launcher derives the actual count from prepared data. Slime's sample count is 16 because it includes both roles. The custom reducer applies `2 * role_weight` to compensate for the combined batch denominator, including KL. Context parallelism is fixed at one.

Only execution-matched SQL and exact-target retrievals receive the length penalty: zero through 512 tokens, linearly reaching −0.5 at 2,048 tokens. Malformed, empty and capped retrieval outputs receive zero. SQL whose columns cannot be mapped cannot enter the pool. Training uses ordered row/column execution comparison and a 30-second timeout.

## Controls and seeds

```bash
ace-sql train --config configs/slime/frozen.toml --dry-run
ace-sql train --config configs/slime/gold.toml --dry-run
ace-sql train --config configs/slime/dense.toml --dry-run
ace-sql train --config configs/slime/current-only.toml --dry-run
ace-sql train --config configs/slime/no-decay.toml --dry-run
ace-sql train --config configs/slime/no-schedule.toml --dry-run
ace-sql train --config configs/slime/long-schedule.toml --dry-run
ace-sql train --config configs/slime/no-length-penalty.toml --dry-run
```

For a container run, select the same file through `ACE_SQL_CONFIG`. Each invocation runs one seed; use a distinct output directory per seed. The exact four manuscript seed IDs are not available in this release. The dense control is maximum pool-route coverage minus `0.5 * out-of-pool-column fraction`.

## Resume and artifacts

```bash
ace-sql train --config configs/slime/paper.toml \
  --resume /assets/runs/ace_sql --output /assets/runs/ace_sql
```

A tensor checkpoint must have its corresponding `rollout/ace_state_ID.json`. The data source rejects missing state, incompatible method versions, changed dataset fingerprints/seed/batch/schedule/controls and nonsequential rollout IDs. It persists the empirical pool, shuffled question order, cursor and SQL history. Do not mix a tensor checkpoint and pool state from different updates.

The run contains `launch.json`, `ace_config.json`, `environment.json`, `ace_audit/rollout_ID.json`, and HF exports under `hf/rollout_ID`. Evaluation validates that a complete HF weight file or every indexed shard exists.

## Validation

```bash
python -m pip install -e '.[test]'
make syntax test verify
# Optional upstream integration tests (slime commit recorded in runtime.md):
SLIME_SOURCE_DIR=/path/to/pinned/slime \
ACE_SQL_TEST_SLIME_DIR=/path/to/pinned/slime \
ACE_SQL_TEST_MEGATRON_ARGS=/path/to/pinned/megatron/training/arguments.py \
make test
```

Tests execute SQLite and loss computations on CPU; model responses are mocked. They test role separation, weighted PPO/KL gradients, pool update timing, controls, failed-rollout rollback, checkpoint replay, package resources and BIRD orchestration. Slime source checks verify the metadata transport and option contracts. They do not establish distributed training, SGLang weight synchronization, HF export on GPU, or manuscript performance.
