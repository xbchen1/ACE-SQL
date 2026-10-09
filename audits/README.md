# Training–Evaluation Overlap Audit

This directory preserves a historical, path-sanitized overlap audit of the retained data snapshots. It is not a new audit of all experiments in the October 9 manuscript. It checks question, SQL, schema-catalog, structural-schema, and database-ID evidence separately; it does not treat a generic SQL string on an unrelated schema as contamination.

Install the isolated audit dependencies from `requirements.txt`, then provide the benchmark metadata files without copying private filesystem paths into the outputs:

```bash
export ACE_SQL_AUDIT_BIRD_JSON=/path/to/BIRD/dev.json
export ACE_SQL_AUDIT_BIRD_TABLES=/path/to/BIRD/dev_tables.json
export ACE_SQL_AUDIT_SPIDER_JSON=/path/to/Spider/dev.json
export ACE_SQL_AUDIT_SPIDER_TABLES=/path/to/Spider/tables.json

python audits/benchmark_overlap_audit.py \
  --repo . \
  --bird-json "$ACE_SQL_AUDIT_BIRD_JSON" \
  --bird-tables "$ACE_SQL_AUDIT_BIRD_TABLES" \
  --spider-json "$ACE_SQL_AUDIT_SPIDER_JSON" \
  --spider-tables "$ACE_SQL_AUDIT_SPIDER_TABLES" \
  --output-dir audits/results
```

Both SFT files are now included. The retriever file was restored byte-for-byte from the published Git snapshot and matches the hash recorded in the historical result manifest. The audit runs from a source archive without Git history. Use `--retriever-sft-json` (or `ACE_SQL_AUDIT_RETRIEVER_SFT_JSON` in the notebook) to audit another snapshot; preserve a separate output directory for that run.

The result manifest stores input labels, SHA-256 hashes, row counts, and runtime versions, but never absolute local paths. SynSQL SQLite files were unavailable, so database-content equality remains unresolved. The audit covers the retained SFT role snapshots and 2,913-row RL split, not the full SynSQL-2.5M source pool.
