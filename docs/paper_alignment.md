# Alignment with the October 9, 2026 manuscript

The source is `ARR_Oct2026/paper_v3.tex` and its compiled 19-page PDF, dated October 9, 2026. [manuscript_manifest.json](manuscript_manifest.json) records their SHA-256 hashes. The manuscript is maintained separately. This repository implements its shared-policy slime recipe as an installable package.

## Method-to-code map

All module paths below are under `src/ace_sql/` unless stated otherwise.

| Manuscript contract | Implementation |
| --- | --- |
| §3.2: shared full-parameter SFT cold start | `configs/sft/sft_qwen3_8b.yaml`, `scripts/train_sft.sh` |
| §3.3.1: union of eight retrieval samples | `schema/rollout.py::union_columns` |
| Algorithm 1: update pool before reading retrieval target | `rewards/engine.py::RewardEngine.compute_group` |
| Pool decay 0.5; tied maxima accepted | `rewards/pool.py::updated_pool`, `target_sets` |
| §3.3.2: matched-only 512–2048-token penalty | `rewards/sql.py::compute_length_penalty`, `rewards/pool.py::retrieval_reward` |
| §3.3.3: per-question, per-role advantages | `training/loss.py::reward_postprocess` |
| Complementary weights and one policy update | `training/schedule.py::role_weights`, `training/loss.py` |
| Frozen reference KL | `training/loss.py::policy_loss`, `training/launch.py` |
| Appendix E: slime, five epochs, eight H20 GPUs | `configs/slime/paper.toml`, `runtime/environment.py` |
| Appendix E: ordered training execution, 30-second timeout | `rewards/sql.py`, `rewards/execution.py`, `rewards/grading.py` |
| Greedy validation without pool update | `training/rollout.py` |
| Shared-policy benchmark evaluation | `evaluation/inference.py`, `bird.py`, `eval_spider.py`, `spider_robustness_pipeline.py` |
| Frozen/gold/dense, decay/schedule/length controls | `configs/slime/*.toml` |

## Repository changes

The first October update corrected union pruning, current-pool target timing, ordered execution and manuscript settings. This refactor changes the repository architecture: removes the vendored verl trainer and PCGrad implementation, replaces the joint reward manager with independent execution/pool/credit modules, unifies experiment configuration, shares schema and inference code, and makes HF exports the evaluation input boundary. Details and old-to-new paths are in [architecture.md](architecture.md).

The bundled data was relocated without changing its bytes. The overview and result table match manuscript-reported four-seed means (4B: 64.7; 8B: 66.3 BIRD Dev) and 8B output cost (0.87k tokens). These values were not produced by the refactor.

## Evidence and limits

CPU checks cover union recall, current-pool targets, ties, gamma=0, frozen/gold/dense controls, length/schedule boundaries, ordered SQL results, role-separated normalization, weighted PPO/KL gradients, rollback, checkpoint replay and read-only validation. Package/configuration checks and a BIRD pipeline fixture validate the reorganized code paths. Optional tests exercise pinned upstream slime loss and metadata transport code.

A real H20 run still needs model loading, two optimizer updates, SGLang weight synchronization and checkpoint recovery. No such GPU validation or new benchmark measurement was performed here.

The SFT snapshots contain 7,092 records per role without shared sample IDs. Their historical audit found 3,189 normalized questions in common; row-wise pairing and the exact data/checkpoint association for all manuscript runs remain unverified. SynSQL SQLite files, initial sampling logs, model weights, exact four seed IDs and per-seed predictions are not bundled. See [data.md](data.md).

Single-stage training, checkpoint-swap experiments, database perturbation experiments and all manuscript analysis figures are not automated by this release. Control recipes provide runnable configurations, not evidence that the corresponding manuscript experiments were rerun. Pre-October method checkpoints remain rejected even when tensor shapes match.
