# ACE-SQL SFT

This directory contains the supervised fine-tuning stage used to initialize the ACE-SQL retriever-generator policy.

Included data:

- `data/domain1_alpaca_think.json`: 7,092 retriever-format samples
- `data/domain2_alpaca_think.json`: 7,092 generator-format samples
- `data/dataset_info.json`: LLaMA-Factory dataset registry

## Run

Place or link the base model at `models/Qwen3-8B`, or edit `configs/sft_qwen3_8b.yaml`.

```bash
bash scripts/train_sft.sh
```

The script runs full-parameter SFT with LLaMA-Factory, DeepSpeed ZeRO-3, four GPUs, and offline W&B mode by default.
