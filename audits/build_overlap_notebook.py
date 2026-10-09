#!/usr/bin/env python3
"""Build the reader-facing notebook for the benchmark-overlap audit."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "benchmark_overlap_audit.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# ACE-SQL benchmark-overlap audit

## tl;dr

This notebook checks question, SQL, schema, and database-identifier overlap separately. It requires cross-level evidence before calling a match contamination: a generic question or `COUNT(*)` query on an unrelated schema is only a lexical collision. The notebook displays the audit results; the CLI writes path-sanitized ledgers under `audits/results/`."""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

The intended use is to assess whether the 7,092-example SFT roles and the 2,913-example RL split overlap BIRD Dev or Spider Dev. Normalized question matching preserves entities and numbers. SQL matching preserves literals. Schema matching uses normalized user-defined `table.column` identifiers and excludes SQLite's internal `sqlite_*` tables. Near-question retrieval uses character 3-5 gram TF-IDF cosine together with token Jaccard; schema-column similarity is Jaccard, while a separate structural fingerprint retains coarse types, primary keys, and foreign keys. Database-content equality remains unresolved when the SynSQL SQLite files are absent.

### Key assumptions

- The SFT retriever file is read from the worktree when supplied, otherwise from the fixed Git `HEAD` blob.
- The generator prompt schema is reported as a prompt schema, not silently treated as a full catalog.
- Exact or near SQL alone is not labeled leakage without database/schema agreement."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import os
import json
import pandas as pd
from IPython.display import display

AUDIT_DIR = Path.cwd().resolve()
if not (AUDIT_DIR / 'benchmark_overlap_audit.py').exists():
    AUDIT_DIR = (AUDIT_DIR / 'audits').resolve()
REPO = AUDIT_DIR.parent
RESULT_DIR = AUDIT_DIR / 'results'

def required_input(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Set {name} to the corresponding benchmark JSON file')
    return Path(value)

BIRD_JSON = required_input('ACE_SQL_AUDIT_BIRD_JSON')
BIRD_TABLES = required_input('ACE_SQL_AUDIT_BIRD_TABLES')
SPIDER_JSON = required_input('ACE_SQL_AUDIT_SPIDER_JSON')
SPIDER_TABLES = required_input('ACE_SQL_AUDIT_SPIDER_TABLES')
retriever_value = os.environ.get('ACE_SQL_AUDIT_RETRIEVER_SFT_JSON')
RETRIEVER_SFT = Path(retriever_value) if retriever_value else None
RESULT_DIR.mkdir(parents=True, exist_ok=True)"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        """import sys
sys.path.insert(0, str(AUDIT_DIR))
from benchmark_overlap_audit import run_audit
from argparse import Namespace

args = Namespace(
    repo=REPO,
    retriever_sft_json=RETRIEVER_SFT,
    bird_json=BIRD_JSON,
    bird_tables=BIRD_TABLES,
    spider_json=SPIDER_JSON,
    spider_tables=SPIDER_TABLES,
    output_dir=RESULT_DIR,
)
results = run_audit(args)
display(pd.DataFrame(results['profiles']))
display(pd.DataFrame(results['manifest']['inputs']).T)"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame(results['question_summary']))
display(pd.DataFrame(results['sql_summary']))
display(pd.DataFrame(results['schema_summary']))
display(pd.DataFrame(results['db_id_summary']))"""
    ),
    nbf.v4.new_markdown_cell("### Inspect all exact SQL and near-schema candidates"),
    nbf.v4.new_code_cell(
        """sql_candidates = pd.DataFrame(results['sql_exact_candidates'])
schema_candidates = pd.DataFrame(results['schema_candidates'])
display(sql_candidates)
display(schema_candidates)"""
    ),
    nbf.v4.new_markdown_cell("### Highest question-similarity candidates"),
    nbf.v4.new_code_cell(
        """top_questions = pd.DataFrame(results['question_top_matches'])
display(top_questions.groupby(['benchmark', 'training_source'], group_keys=False).head(3))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

Interpret the executed tables conservatively. BIRD is clean under the tested exact and high-threshold near-overlap rules when no schema or database match appears. Spider requires a schema-level caveat if the RL `world` catalog matches `world_1`; without the SynSQL SQLite file, database-content identity cannot be ruled in or out. Any generic exact SQL collision on a different schema should be disclosed as a lexical collision, not promoted to contamination. This audit covers the retained 7,092-example-per-role SFT snapshots and 2,913-row RL split; it is not a substitute for a source-manifest audit of all SynSQL-2.5M records."""
    ),
]
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
