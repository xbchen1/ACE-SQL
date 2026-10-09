# ACE-SQL supervised fine-tuning

Mix the 7,092 retriever and 7,092 generator examples in one full-parameter LLaMA-Factory run. Both role files and their dataset registry are included in `data/`; see [data provenance](../docs/data.md).

The October 9 recipe uses eight GPUs, one epoch, learning rate `2e-5`, cosine decay, 5% warmup, BF16, and DeepSpeed ZeRO-3. `cutoff_len=6144` supplies the combined 4,096-prompt + 2,048-response budget; LLaMA-Factory's combined cutoff does not independently enforce each component's cap.

```bash
# Place the base checkpoint at models/Qwen3-8B, or override its path:
bash scripts/train_sft.sh model_name_or_path=/path/to/Qwen3-8B
```

`ACE_SQL_NUM_GPUS` defaults to eight. `ACE_SQL_SFT_CONFIG` selects another configuration file. For 4B, provide the Qwen3-4B model path and a distinct output directory:

```bash
bash scripts/train_sft.sh model_name_or_path=/path/to/Qwen3-4B output_dir=outputs/sft_qwen3_4b
```

W&B defaults to offline mode. Model weights and output checkpoints are not shipped. This release did not rerun SFT or verify that the retained role snapshots are the exact paired examples used for every reported run.
