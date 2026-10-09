# Included data and provenance

[The data manifest](data_manifest.json) identifies the exact bytes shipped in this repository.

| File | Records | Purpose |
| --- | ---: | --- |
| `data/sft/domain1_alpaca_think.json` | 7,092 | Retriever SFT examples |
| `data/sft/domain2_alpaca_think.json` | 7,092 | Generator SFT examples |
| `data/rl/train.parquet` | 2,913 | RL question/database pairs |
| `data/rl/validation.parquet` | 40 | Runtime validation questions |
| `data/rl/initial_pool.json` | 2,953 keys | Initial column-set counts for both splits |

The retriever SFT JSON was restored from the repository's existing committed snapshot, with the same SHA-256 as the historical overlap audit. The other dataset bytes are unchanged. Database-cell examples can contain literal file paths from the source data; they are not runtime filesystem locations.

The paper describes filtering hard SynSQL questions, unioning eight initialization retriever samples, sampling sixteen SQLs, and retaining questions with 2–14 correct SQLs. The retained data files provide the resulting prompts, metadata, and initial column counts; they do not contain a complete source-generation execution log.

The two SFT role snapshots have no shared sample ID. Their historical audit found 3,189 normalized questions in common, so the supplied files should not be described as a verified row-aligned set of 7,092 pairs. No re-pairing or synthetic replacement was performed in this update. The [historical overlap audit](../audits/README.md) records its own fixed input hashes and is not a new audit of the October manuscript's full experiment history.

The RL converter verifies disjoint question keys across training and validation, schema metadata, in-range column IDs, and complete initial-pool coverage. With the actual checkpoint tokenizer it also records prompt-length filtering. Runtime validation is separate from BIRD/Spider benchmark evaluation.

Supply externally:

- Qwen3 base weights, SFT/RL checkpoints, and the Megatron reference checkpoint.
- SynSQL training/validation SQLite files, mounted through `ACE_SQL_TRAIN_DB_ROOT`, `ACE_SQL_DEV_DB_ROOT`, or `ACE_SQL_LOOSE_DB_ROOTS`.
- BIRD/Spider metadata and databases for benchmark evaluation.
- Per-seed resolved configurations, prediction files, and checkpoint identifiers when reproducing reported measurements.

Download datasets under their original licenses and terms. The project code license does not replace upstream data/model terms.
