# Storage Layout

This repo keeps only the code and the three files used by the RL PCGrad entrypoint:

- `data/train.parquet`
- `data/initial_pool.json`
- `data/validation.parquet`

Large local assets stay outside git:

- `models/`: base model or symlink to a model directory
- `external/`: optional database mounts
- `.run/`: checkpoints, Ray temp files, object spilling, logs, and wandb offline data

The default training script writes temporary and checkpoint files under `.run/` so it does not consume the smaller working filesystem.
