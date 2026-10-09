# Alignment with the October 9, 2026 manuscript

The source of this update is `ARR_Oct2026/paper_v3.tex` and its compiled 19-page PDF, dated October 9, 2026. [manuscript_manifest.json](manuscript_manifest.json) records their SHA-256 hashes. The manuscript itself is maintained separately; old revision directories and simulated experiment drafts are not repository inputs.

## Method-to-code map

| Manuscript contract | Implementation |
| --- | --- |
| §3.2: shared SFT cold start | `SFT/configs/sft_qwen3_8b.yaml`, `SFT/scripts/train_sft.sh` |
| §3.3.1: union of eight retrieval samples | `RL/ace_sql_slime/core.py::union_columns` |
| Algorithm 1: update pool before reading the target | `RL/rewards/joint_reward.py::JointRewardManager.compute_rewards` |
| Pool decay 0.5; tied maxima accepted | `JointRewardManager._target_sets_from_pool` |
| §3.3.2: matched-only 512–2048-token length penalty | `compute_length_penalty`, `compute_sql_reward`, `compute_pool_exact_retriever_reward` |
| §3.3.3: independent per-question, per-role advantages | `RL/ace_sql_slime/loss.py::reward_postprocess` |
| Complementary weights, one shared-policy update | `core.py::role_weights`, `loss.py::weighted_sample_reducer` |
| Frozen reference KL | `loss.py::policy_loss`, `launch.py::build_command` |
| Appendix E: slime, five RL epochs, eight H20 GPUs | `RL/ace_sql_slime/launch.py` and the pinned runtime contract |
| Appendix E: ordered training execution, 30-second timeout | `RL/rewards/joint_reward.py`, `RL/scripts/sql_utils.py` |
| Greedy evaluation, no pool update | `rollout.py::_rollout_impl`, `EVAL/` |
| Frozen/gold/dense target, decay/schedule/length controls | `ACE_SQL_TARGET_MODE`, `ACE_SQL_POOL_GAMMA`, `ACE_SQL_RAMP_RATIO`, `ACE_SQL_LENGTH_PENALTY` |

## Changes from the previous repository state

- Replaced majority-vote pruning with union pruning in the current RL adapter. A column proposed by only one of eight samples is retained.
- Read the retrieval target after adding this rollout's correct SQL column sets. Validation and frozen-pool controls neither decay nor change saved counts.
- Made `train_rl.sh` invoke slime. The legacy verl/PCGrad launcher is retained but is not the default recipe.
- Updated SFT to one epoch and eight GPUs. Its total sequence budget is 6,144 tokens. LLaMA-Factory's combined cutoff is not a separate enforcement of both component caps.
- Changed training execution matching from unordered bags to ordered rows and columns; numeric cells retain the bundled grader's six-decimal normalization. BIRD evaluation retains its benchmark set comparison and now defaults to 180 seconds.
- Added configurable paper target/decay/schedule/length controls and method-aware checkpoint compatibility checks.
- Fixed an ignore pattern that excluded `RL/ace_sql_slime/core.py` from Git.
- Restored the missing retriever SFT file byte-for-byte from the existing published Git snapshot; its digest matches the historical overlap audit. No dataset labels or examples were invented.
- Updated the overview image and README results to the manuscript's four-seed means (4B: 64.7; 8B: 66.3 BIRD Dev) and 8B output cost (0.87k).

## Evidence and limits

CPU regression tests cover union recall, current-pool targets, tied targets, gamma=0, frozen/gold/dense controls, length and schedule boundaries, ordered SQL execution, role-separated normalization, weighted PPO and KL gradients, failed-rollout rollback, checkpoint replay, and read-only validation. Optional integration tests execute selected loss functions and metadata transport from the pinned upstream slime source.

These checks validate code behavior on fixtures. They do not reproduce manuscript scores or establish successful H20 distributed training. The pinned image still needs a real two-update run that includes SGLang weight synchronization and checkpoint recovery.

The bundled SFT snapshots contain 7,092 records per role but no shared sample IDs. Their historical audit found 3,189 normalized questions in common; row-wise pairing and the exact data/checkpoint association for all manuscript runs remain unverified. The SynSQL SQLite files, initial sampling logs, model weights, exact four seed identifiers, and per-seed prediction files are not bundled. See [data.md](data.md).

Single-stage training, checkpoint-swap evaluation, database perturbation, and all manuscript analysis figures are not automated by this release. Current control switches provide runnable recipes, not archived evidence that those experiments were executed here.

Legacy verl code has additional historical options and should not be used to infer the current slime method. Old adapter checkpoints are rejected after the union and pool-timing change, even when tensor shapes would otherwise permit loading.
