# ACE-SQL joint GRPO on slime

The current manuscript recipe is implemented in `ace_sql_slime/`. Both roles share one full-parameter Qwen3 policy. `scripts/train_rl.sh` invokes the slime launcher and disables gradient projection.

The adapter has CPU regression coverage. Its eight-H20 execution, distributed kernels, and checkpoint export still require a GPU smoke run in the [pinned environment](configs/slime-h20-compatibility.md).

## Prepare the runtime and assets

Use the pinned slime v0.2.3 / CUDA 12.9 image and wrapper described in the environment guide. Do not install the legacy `requirements.txt` into this runtime. Model weights and SQLite databases are external:

```bash
export ACE_SQL_MODEL_PATH=/assets/models/sft_checkpoint
export ACE_SQL_REFERENCE_PATH=/assets/models/sft_torch_dist
export ACE_SQL_TRAIN_DB_ROOT=/assets/databases/train
export ACE_SQL_DEV_DB_ROOT=/assets/databases/dev
```

The path resolver also accepts `ACE_SQL_LOOSE_DB_ROOTS` for additional database directories. The same SFT checkpoint initializes both roles and the frozen reference policy.

Inside the pinned container, convert the SFT checkpoint once (8B example):

```bash
export ACE_SQL_SLIME_DIR=/root/slime
source "$ACE_SQL_SLIME_DIR/scripts/models/qwen3-8B.sh"
torchrun --standalone --nproc_per_node=8 \
  "$ACE_SQL_SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" --hf-checkpoint "$ACE_SQL_MODEL_PATH" \
  --save "$ACE_SQL_REFERENCE_PATH"
```

For 4B, use the corresponding `qwen3-4B.sh` model configuration. From `RL/`, convert the included Parquet files and filter overlong prompts with the actual checkpoint tokenizer:

```bash
python -m ace_sql_slime.data \
  --train data/train.parquet --validation data/validation.parquet \
  --initial-pool data/initial_pool.json --output-dir .run/slime_data \
  --tokenizer "$ACE_SQL_MODEL_PATH"

bash scripts/train_rl.sh --dry-run
bash scripts/train_rl.sh
```

The converter checks split separation, metadata consistency, schema column indices, and initial-pool coverage. Its manifest records input hashes and any filtered question IDs. Without length filtering there are 2,913 training questions, 40 validation questions, and 2,953 pool keys. With question batch size 16 and `drop_last`, five epochs contain 910 updates; actual filtered row counts determine the final step count.

## Training contract

| Setting | Default |
| --- | --- |
| Policy | Shared Qwen3-8B or Qwen3-4B; full parameter tuning |
| Rollouts | 8 retrieval outputs, union schema, 8 SQL outputs per question |
| Pool | Decay old counts by 0.5, add current correct SQL column sets, then score retrieval |
| Rewards | Exact pool target / execution match, with matched-only length penalty |
| Advantages | Per question and per role; sample standard deviation with `1e-6` epsilon |
| Role weights | Retriever 1 → 0.5; generator 0 → 0.5 over the first 25% of steps |
| Optimizer | One joint update per rollout; Adam, LR `1e-6`, 5% warmup |
| KL / clip | Frozen-reference KL `0.001`; PPO clip `0.2` |
| Batch | 16 questions = 256 trajectories; eight GPUs by default |
| Lengths | Prompt 4,096; response 2,048 tokens |
| Epochs / sampling | 5; temperature 1.0 |
| Validation / save | Every 40 / 80 steps, plus final save |

Slime's `n_samples_per_prompt=16` counts both roles. The custom reward hook separates these into two groups of eight. The loss reducer applies `2 * role_weight` to compensate for the combined-batch denominator, including the KL term. Context parallelism is fixed at one.

Malformed retriever outputs and outputs reaching the retrieval response cap receive zero reward. Empty parsed retrievals do not trigger a gold-schema or full-schema fallback. SQL that cannot be mapped to schema columns cannot enter the pool. Training preserves row and column order when comparing execution results; benchmark evaluation keeps each benchmark's evaluator.

## Controls

Every invocation runs one seed. Set `ACE_SQL_SEED` and a distinct `ACE_SQL_SLIME_OUTPUT` for each run; the exact four manuscript seed IDs and checkpoints are not bundled.

```bash
# Frozen initial pool / static gold-column targets / dense coverage-noise reward
ACE_SQL_TARGET_MODE=frozen bash scripts/train_rl.sh
ACE_SQL_TARGET_MODE=gold bash scripts/train_rl.sh
ACE_SQL_TARGET_MODE=dense bash scripts/train_rl.sh

# Decay, schedule, and length controls
ACE_SQL_POOL_GAMMA=0 bash scripts/train_rl.sh
ACE_SQL_POOL_GAMMA=1 bash scripts/train_rl.sh
ACE_SQL_RAMP_RATIO=0 bash scripts/train_rl.sh
ACE_SQL_RAMP_RATIO=0.5 bash scripts/train_rl.sh
ACE_SQL_LENGTH_PENALTY=false bash scripts/train_rl.sh
```

The dense control implements the manuscript's maximum pool-route coverage minus `0.5 * out_of_pool_column_fraction`. It does not use the legacy `v6` shaping function. Single-stage training, checkpoint-swap, and perturbation experiment runners are not included in this adapter.

## Resume and inspect

```bash
ACE_SQL_SLIME_RESUME=/assets/runs/ace_sql \
ACE_SQL_SLIME_OUTPUT=/assets/runs/ace_sql \
bash scripts/train_rl.sh
```

A model checkpoint must have the matching `rollout/ace_state_ID.json`. The adapter persists the pool, question order, cursor, and SQL history. It rejects missing state, changed method semantics, different data/seed/batch/schedule/control settings, and nonsequential rollout IDs. Pre-October-9 adapter checkpoints cannot resume under the updated union/current-pool behavior.

`launch.json`, `ace_config.json`, and `environment.json` record the resolved configuration. `ace_audit/rollout_ID.json` records SQL and reward details. HF exports are saved under `hf/rollout_ID`; check export completeness before evaluation.

## Checks and code map

```bash
make syntax
make test
# Optional: execute upstream CPU loss/metadata contract tests too
SLIME_SOURCE_DIR=/path/to/pinned/slime ACE_SQL_TEST_SLIME_DIR=/path/to/pinned/slime make test
```

- `ace_sql_slime/core.py`: union schema, role weights, reward state.
- `ace_sql_slime/rollout.py`: SGLang requests, validation, transactional recovery.
- `ace_sql_slime/loss.py`: independent advantages and weighted PPO/KL.
- `ace_sql_slime/data.py`: data conversion and validation.
- `ace_sql_slime/launch.py`: launch configuration and controls.
- `patches/`: metadata transport patch for the fixed slime version.

Legacy verl sources, configurations, and `train_rl_pcgrad.sh` remain for compatibility only. They are not part of the current training entrypoint.
