# Slime repository architecture

The package separates the Text-to-SQL method from distributed execution. There is one training backend: pinned slime with Megatron and SGLang. Framework sources are external runtime dependencies, not a second trainer maintained in this repository.

## Training lifecycle

1. `training.config` validates the TOML recipe, environment overrides and CLI overrides, then resolves paths. `runtime.prepare` and `training.launch` use this same configuration.
2. `data.rl` validates the immutable Parquet records, preserves original messages/metadata, filters prompts with the actual SFT tokenizer, and writes JSONL plus hashes. Preparation converts the same SFT checkpoint into the frozen Megatron reference format.
3. `training.rollout.ACEDataSource` owns shuffled question order, cursor, pool state and checkpoint compatibility. The rollout samples eight retrievals from one SGLang policy, builds their schema union, then samples eight SQL responses from that policy.
4. `rewards.engine.RewardEngine` scores one question at a time. `rewards.execution` owns SQLite time limits; `rewards.sql` owns formatting and execution credit; `rewards.pool` owns decay, tied targets and retrieval credit. These modules import neither torch nor slime.
5. `training.loss` normalizes rewards separately by question and role, then computes weighted PPO and reference KL using slime's reduction contract. Slime runs one shared-policy optimizer update and synchronizes weights with SGLang.
6. Slime persists model/optimizer state and exports HF weights. The data source persists matching pool, question cursor and history. A failed rollout restores its pre-rollout state; evaluation does not mutate it.

## Package and artifact boundaries

- `schema` is shared by training and benchmark prompting. Empty selections stay empty; schema parsing does not add missing gold columns.
- `evaluation.inference` is the only first-party vLLM implementation. Retrieval and SQL commands use the same full-parameter HF checkpoint. BIRD and Spider keep their benchmark execution rules, separate from training reward comparison.
- Runtime templates, the pinned metadata patch and the third-party evaluator license ship in the Python wheel. User models, datasets, `.runs/` and external framework clones do not.
- Dataset/config directories belong to the project checkout. `ACE_SQL_HOME` or `--project` identifies that checkout when running an installed package from another directory.
- The metadata patch is narrowly scoped to the pinned slime version: carry per-sample role weights through data splitting and batch loading. No slime trainer is copied into ACE-SQL.

## Migration from the prior layout

| Previous path / behavior | Current owner |
| --- | --- |
| `RL/ace_sql_slime/core.py` | `schema/rollout.py`, `training/schedule.py`, `rewards/engine.py` |
| `RL/rewards/joint_reward.py` | Independent `rewards/sql.py`, `pool.py`, `engine.py` |
| `RL/utils/` and `EVAL/src/utils/` | Shared `schema/` and `data/paths.py` |
| `RL/ace_sql_slime/{rollout,loss,launch,data}.py` | `training/` and `data/rl.py` |
| Scattered launch environment settings | Validated `configs/slime/*.toml` |
| SFT/RL embedded datasets | `data/sft/` and `data/rl/`, unchanged fingerprints |
| `EVAL/` scripts and checkpoint merging | `evaluation/`, direct complete HF export input |
| `RL/trainer/`, vendored verl, PCGrad | Removed from this release; previous implementations remain in project history |

The data-source checkpoint schema and method identifier remain compatible with the immediately preceding October 9 union/current-pool slime implementation. Pre-October method states remain incompatible. This preserves semantic checks while removing the dependency on its old Python module names.

CPU tests establish method and packaging behavior. Real distributed H20 training and published benchmark numbers require external weights/databases and a GPU validation run.
