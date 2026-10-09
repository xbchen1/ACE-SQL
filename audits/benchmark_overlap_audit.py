#!/usr/bin/env python3
"""Reproducible training/evaluation overlap audit for ACE-SQL.

The audit is intentionally read-only with respect to all source datasets.  It
separates question, SQL, schema, and database-identity evidence so that a
generic COUNT(*) query is not mislabeled as leakage when its database schema is
different.  Database-content equality is left unresolved when the SynSQL
SQLite files are unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


QUESTION_RE = re.compile(
    r"\nQuestion:\s*\n(?P<question>.*?)(?:\n\s*Instructions:\s*\n)",
    re.IGNORECASE | re.DOTALL,
)
SCHEMA_RE = re.compile(
    r"Database Schema:\s*\n(?P<schema>.*?)(?:\nThis schema describes|\nQuestion:)",
    re.IGNORECASE | re.DOTALL,
)
SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_LITERAL_RE = re.compile(r"'(?:''|[^'])*'", re.DOTALL)
CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|([^\s(]+))"
    r"\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
IDENT_RE = re.compile(
    r"^\s*(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|([^\s,(]+))"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def normalize_sql_lexical(value: str) -> str:
    """Conservative SQL normalization that preserves literals and identifiers."""
    value = unicodedata.normalize("NFKC", value or "")
    literals: list[str] = []

    def stash_literal(match: re.Match[str]) -> str:
        token = f"__ace_sql_literal_{len(literals)}__"
        literals.append(match.group(0))
        return token

    value = SQL_LITERAL_RE.sub(stash_literal, value).casefold()
    value = re.sub(r"--[^\n]*", " ", value)
    value = re.sub(r"/\*.*?\*/", " ", value, flags=re.DOTALL)
    value = value.replace("`", "").replace('"', "")
    value = re.sub(r"\s+", "", value)
    for index, literal in enumerate(literals):
        value = value.replace(f"__ace_sql_literal_{index}__", literal)
    return value.rstrip(";")


def sql_template(value: str) -> str:
    value = normalize_sql_lexical(value)
    value = re.sub(r"'(?:''|[^'])*'", "?str", value)
    return re.sub(r"\b\d+(?:\.\d+)?\b", "?num", value)


def extract_sft_record(record: dict[str, Any], role: str) -> dict[str, Any]:
    question_match = QUESTION_RE.search(record["instruction"])
    schema_match = SCHEMA_RE.search(record["instruction"])
    if not question_match or not schema_match:
        raise ValueError(f"Could not parse {role} SFT prompt")
    result = {
        "question": question_match.group("question").strip(),
        "schema_ddl": schema_match.group("schema").strip(),
    }
    if role == "generator":
        final_region = record["output"].split("</think>", 1)[-1]
        blocks = SQL_FENCE_RE.findall(final_region)
        if len(blocks) != 1:
            raise ValueError(
                f"Expected one final SQL fence after </think>, found {len(blocks)}"
            )
        result["sql"] = blocks[0].strip()
    return result


def _normalized_identifier(value: str) -> str:
    return normalize_text(value)


def schema_columns_from_ddl(ddl: str) -> frozenset[str]:
    """Extract normalized table.column identifiers from generated DDL."""
    columns: set[str] = set()
    for match in CREATE_RE.finditer(ddl or ""):
        table = next(value for value in match.groups()[:4] if value is not None)
        table_norm = _normalized_identifier(table)
        if table_norm.startswith("sqlite_"):
            continue
        for line in match.group(5).splitlines():
            line = line.strip().rstrip(",")
            if not line or re.match(
                r"(?i)^(primary|foreign|constraint|unique|check)\b", line
            ):
                continue
            identifier_match = IDENT_RE.match(line)
            if not identifier_match:
                continue
            column = next(
                value for value in identifier_match.groups() if value is not None
            )
            if column != "*":
                columns.add(f"{table_norm}.{_normalized_identifier(column)}")
    return frozenset(columns)


def schema_columns_from_json(schema: dict[str, Any]) -> frozenset[str]:
    tables = schema.get("table_names_original") or schema.get("table_names") or []
    columns = (
        schema.get("column_names_original") or schema.get("column_names") or []
    )
    result: set[str] = set()
    for table_index, column in columns:
        if table_index is None or table_index < 0 or column == "*":
            continue
        table = _normalized_identifier(tables[table_index])
        if table.startswith("sqlite_"):
            continue
        result.add(f"{table}.{_normalized_identifier(column)}")
    return frozenset(result)


def _coarse_type(value: str) -> str:
    value = normalize_text(value)
    if any(token in value for token in ("int", "real", "number", "float", "double", "decimal")):
        return "number"
    if any(token in value for token in ("char", "text", "string")):
        return "text"
    if any(token in value for token in ("date", "time", "year")):
        return "time"
    if "bool" in value:
        return "boolean"
    return value or "unknown"


def schema_structure_from_json(schema: dict[str, Any]) -> tuple:
    """Canonical user-defined catalog including coarse types, PKs, and FKs."""
    tables = schema.get("table_names_original") or schema.get("table_names") or []
    columns = schema.get("column_names_original") or schema.get("column_names") or []
    types = schema.get("column_types") or []

    def reference(index: int) -> tuple[str, str] | None:
        if not isinstance(index, int) or not (0 <= index < len(columns)):
            return None
        table_index, column = columns[index]
        if table_index is None or table_index < 0 or column == "*":
            return None
        table = _normalized_identifier(tables[table_index])
        if table.startswith("sqlite_"):
            return None
        return table, _normalized_identifier(column)

    typed_columns = []
    for index, (table_index, column) in enumerate(columns):
        ref = reference(index)
        if ref is None:
            continue
        column_type = types[index] if index < len(types) else "unknown"
        typed_columns.append((*ref, _coarse_type(column_type)))

    primary_keys = []
    for key in schema.get("primary_keys") or []:
        indices = key if isinstance(key, list) else [key]
        refs = tuple(sorted(ref for index in indices if (ref := reference(index))))
        if refs:
            primary_keys.append(refs)

    foreign_keys = []
    for left, right in schema.get("foreign_keys") or []:
        left_ref, right_ref = reference(left), reference(right)
        if left_ref and right_ref:
            foreign_keys.append((left_ref, right_ref))

    return (
        tuple(sorted(typed_columns)),
        tuple(sorted(primary_keys)),
        tuple(sorted(foreign_keys)),
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_retriever_sft(repo: Path, explicit_path: Path | None) -> tuple[list, dict]:
    explicit_path = explicit_path or repo / "data/sft/domain1_alpaca_think.json"
    if explicit_path.exists():
        raw = explicit_path.read_bytes()
        origin = "provided-file:data/sft/domain1_alpaca_think.json"
    else:
        raw = subprocess.check_output(
            ["git", "-C", str(repo), "show", "HEAD:data/sft/domain1_alpaca_think.json"]
        )
        origin = "git:HEAD:data/sft/domain1_alpaca_think.json"
    return json.loads(raw), {"origin": origin, "sha256": sha256_bytes(raw)}


def nearest_question_matches(
    train_questions: list[str], benchmark_questions: list[str]
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(train_questions + benchmark_questions)
    train_matrix = matrix[: len(train_questions)]
    benchmark_matrix = matrix[len(train_questions) :]
    neighbor_count = min(10, len(train_questions))
    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count, metric="cosine", algorithm="brute"
    )
    neighbors.fit(train_matrix)
    distance, index = neighbors.kneighbors(benchmark_matrix)
    similarity = 1.0 - distance
    rows = []
    high_confidence_rows = 0
    for benchmark_index in range(len(benchmark_questions)):
        benchmark_tokens = set(normalize_text(benchmark_questions[benchmark_index]).split())
        candidate_rows = []
        for score, train_index in zip(similarity[benchmark_index], index[benchmark_index]):
            train_tokens = set(normalize_text(train_questions[int(train_index)]).split())
            token_union = benchmark_tokens | train_tokens
            token_jaccard = (
                len(benchmark_tokens & train_tokens) / len(token_union)
                if token_union
                else 1.0
            )
            short_question = len(benchmark_tokens) < 8
            high_confidence = (
                score >= (0.95 if short_question else 0.92)
                and token_jaccard >= 0.80
            )
            candidate_rows.append(
                (float(score), int(train_index), float(token_jaccard), high_confidence)
            )
        if any(candidate[3] for candidate in candidate_rows):
            high_confidence_rows += 1
        score, train_index, token_jaccard, high_confidence = candidate_rows[0]
        rows.append(
            {
                "benchmark_row": benchmark_index,
                "train_row": train_index,
                "char_tfidf_similarity": score,
                "token_jaccard": token_jaccard,
                "high_confidence_near": high_confidence,
                "benchmark_question": benchmark_questions[benchmark_index],
                "train_question": train_questions[train_index],
            }
        )
    frame = pd.DataFrame(rows).sort_values("char_tfidf_similarity", ascending=False)
    top_similarity = similarity[:, 0]
    summary = {
        "char_tfidf_max": float(top_similarity.max(initial=0.0)),
        "char_ge_0_85": int((top_similarity >= 0.85).sum()),
        "char_ge_0_90": int((top_similarity >= 0.90).sum()),
        "char_ge_0_95": int((top_similarity >= 0.95).sum()),
        "high_confidence_near": high_confidence_rows,
    }
    return frame, summary


def _benchmark_records(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                "benchmark": name,
                "row": index,
                "db_id": row["db_id"],
                "question": row["question"],
                "sql": row.get("SQL") or row.get("query") or "",
            }
        )
    return result


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    generator_path = repo / "data/sft/domain2_alpaca_think.json"
    rl_path = repo / "data/rl/train.parquet"

    generator_raw = load_json(generator_path)
    generator_rows = [extract_sft_record(row, "generator") for row in generator_raw]
    retriever_raw, retriever_manifest = load_retriever_sft(
        repo, args.retriever_sft_json
    )
    retriever_rows = [extract_sft_record(row, "retriever") for row in retriever_raw]
    rl_raw = pq.read_table(rl_path).to_pylist()
    rl_rows = []
    for row_index, row in enumerate(rl_raw):
        ground_truth = row["reward_model"]["ground_truth"]
        schema = json.loads(row["extra_info"]["db_info"])
        rl_rows.append(
            {
                "row": row_index,
                "db_id": ground_truth["db_id"],
                "question": ground_truth["question"],
                "sql": ground_truth["sql"],
                "schema": schema_columns_from_json(schema),
                "schema_structure": schema_structure_from_json(schema),
            }
        )

    bird_rows = _benchmark_records("BIRD", load_json(args.bird_json))
    spider_rows = _benchmark_records("Spider", load_json(args.spider_json))
    benchmark_rows = {"BIRD": bird_rows, "Spider": spider_rows}
    benchmark_table_paths = {
        "BIRD": args.bird_tables,
        "Spider": args.spider_tables,
    }

    train_questions = {
        "SFT-retriever": [row["question"] for row in retriever_rows],
        "SFT-generator": [row["question"] for row in generator_rows],
        "RL": [row["question"] for row in rl_rows],
    }
    train_sql = {
        "SFT-generator": [row["sql"] for row in generator_rows],
        "RL": [row["sql"] for row in rl_rows],
    }
    train_schema = {
        "SFT-retriever-full": [
            schema_columns_from_ddl(row["schema_ddl"]) for row in retriever_rows
        ],
        "SFT-generator-prompt": [
            schema_columns_from_ddl(row["schema_ddl"]) for row in generator_rows
        ],
        "RL-full": [row["schema"] for row in rl_rows],
    }
    retriever_question_counter = Counter(
        normalize_text(row["question"]) for row in retriever_rows
    )
    generator_question_counter = Counter(
        normalize_text(row["question"]) for row in generator_rows
    )
    role_question_overlap = sum(
        (retriever_question_counter & generator_question_counter).values()
    )

    manifest = {
        "repo": ".",
        "repo_head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_script": {
            "path": "audits/benchmark_overlap_audit.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "sft_generator": {
                "path": "data/sft/domain2_alpaca_think.json",
                "sha256": sha256_file(generator_path),
                "rows": len(generator_rows),
            },
            "sft_retriever": {**retriever_manifest, "rows": len(retriever_rows)},
            "rl_train": {
                "path": "data/rl/train.parquet",
                "sha256": sha256_file(rl_path),
                "rows": len(rl_rows),
            },
            "bird_json": {
                "path": "external:BIRD/dev.json",
                "sha256": sha256_file(args.bird_json),
                "rows": len(bird_rows),
            },
            "bird_tables": {
                "path": "external:BIRD/dev_tables.json",
                "sha256": sha256_file(args.bird_tables),
            },
            "spider_json": {
                "path": "external:Spider/dev.json",
                "sha256": sha256_file(args.spider_json),
                "rows": len(spider_rows),
            },
            "spider_tables": {
                "path": "external:Spider/tables.json",
                "sha256": sha256_file(args.spider_tables),
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sft_role_question_alignment": {
            "retriever_rows": len(retriever_rows),
            "generator_rows": len(generator_rows),
            "normalized_multiset_overlap": role_question_overlap,
            "retriever_only_multiset_rows": len(retriever_rows) - role_question_overlap,
            "generator_only_multiset_rows": len(generator_rows) - role_question_overlap,
            "row_aligned_normalized_questions": sum(
                normalize_text(retriever["question"])
                == normalize_text(generator["question"])
                for retriever, generator in zip(retriever_rows, generator_rows)
            ),
            "stable_sample_id_available": False,
        },
    }

    profiles = []
    for source, questions in train_questions.items():
        profiles.append(
            {
                "source": source,
                "rows": len(questions),
                "unique_normalized_questions": len(set(map(normalize_text, questions))),
            }
        )
    for source, sql_values in train_sql.items():
        profiles.append(
            {
                "source": source + " SQL",
                "rows": len(sql_values),
                "unique_normalized_sql": len(
                    set(map(normalize_sql_lexical, sql_values))
                ),
            }
        )

    question_summary = []
    question_top = []
    sql_summary = []
    sql_candidates = []
    schema_summary = []
    schema_candidates = []
    db_id_summary = []

    rl_db_ids = {normalize_text(row["db_id"]) for row in rl_rows}
    for benchmark, rows in benchmark_rows.items():
        benchmark_questions = [row["question"] for row in rows]
        benchmark_sql = [row["sql"] for row in rows]
        for source, questions in train_questions.items():
            train_exact = defaultdict(list)
            for index, question in enumerate(questions):
                train_exact[normalize_text(question)].append(index)
            exact_count = sum(
                normalize_text(question) in train_exact
                for question in benchmark_questions
            )
            near_frame, near_summary = nearest_question_matches(
                questions, benchmark_questions
            )
            question_summary.append(
                {
                    "benchmark": benchmark,
                    "training_source": source,
                    "normalized_exact": exact_count,
                    **near_summary,
                }
            )
            for _, candidate in near_frame.head(5).iterrows():
                question_top.append(
                    {
                        "benchmark": benchmark,
                        "training_source": source,
                        **candidate.to_dict(),
                    }
                )

        for source, sql_values in train_sql.items():
            exact_map = defaultdict(list)
            template_set = set()
            for index, sql in enumerate(sql_values):
                exact_map[normalize_sql_lexical(sql)].append(index)
                template_set.add(sql_template(sql))
            exact_rows = 0
            template_rows = 0
            for benchmark_row, sql in enumerate(benchmark_sql):
                normalized = normalize_sql_lexical(sql)
                template = sql_template(sql)
                if normalized in exact_map:
                    exact_rows += 1
                    for train_row in exact_map[normalized]:
                        sql_candidates.append(
                            {
                                "benchmark": benchmark,
                                "training_source": source,
                                "benchmark_row": benchmark_row,
                                "benchmark_db_id": rows[benchmark_row]["db_id"],
                                "benchmark_question": rows[benchmark_row]["question"],
                                "benchmark_sql": sql,
                                "train_row": train_row,
                                "train_question": train_questions[source][train_row],
                                "train_sql": sql_values[train_row],
                            }
                        )
                if template in template_set:
                    template_rows += 1
            sql_summary.append(
                {
                    "benchmark": benchmark,
                    "training_source": source,
                    "normalized_exact_rows": exact_rows,
                    "template_exact_rows": template_rows,
                }
            )

        dev_db_ids = {row["db_id"] for row in rows}
        benchmark_tables = [
            table
            for table in load_json(benchmark_table_paths[benchmark])
            if table["db_id"] in dev_db_ids
        ]
        benchmark_schemas = {
            table["db_id"]: schema_columns_from_json(table)
            for table in benchmark_tables
        }
        benchmark_structures = {
            table["db_id"]: schema_structure_from_json(table)
            for table in benchmark_tables
        }
        db_id_summary.append(
            {
                "benchmark": benchmark,
                "rl_db_id_exact": len(
                    rl_db_ids & {normalize_text(db_id) for db_id in dev_db_ids}
                ),
                "rl_unique_db_ids": len(rl_db_ids),
                "benchmark_unique_db_ids": len(dev_db_ids),
            }
        )
        for source, schemas in train_schema.items():
            unique_schemas = list(set(schemas))
            exact_catalog_count = 0
            exact_structure_count = 0
            ge_085 = 0
            ge_095 = 0
            for db_id, benchmark_schema in benchmark_schemas.items():
                scores = [jaccard(benchmark_schema, schema) for schema in unique_schemas]
                best_score = max(scores, default=0.0)
                best_schema = unique_schemas[scores.index(best_score)] if scores else frozenset()
                if best_score == 1.0:
                    exact_catalog_count += 1
                if best_score >= 0.85:
                    ge_085 += 1
                if best_score >= 0.95:
                    ge_095 += 1
                if best_score >= 0.85:
                    matching_rows = [
                        index for index, schema in enumerate(schemas) if schema == best_schema
                    ]
                    matching_db_ids = []
                    if source == "RL-full":
                        matching_db_ids = sorted(
                            {rl_rows[index]["db_id"] for index in matching_rows}
                        )
                        exact_structure_count += int(
                            any(
                                rl_rows[index]["schema_structure"]
                                == benchmark_structures[db_id]
                                for index in matching_rows
                            )
                        )
                    schema_candidates.append(
                        {
                            "benchmark": benchmark,
                            "training_source": source,
                            "benchmark_db_id": db_id,
                            "best_jaccard": best_score,
                            "training_rows": matching_rows,
                            "matched_training_row_count": len(matching_rows),
                            "training_db_ids": matching_db_ids,
                            "catalog_column_count": len(benchmark_schema),
                            "benchmark_db_question_count": sum(
                                row["db_id"] == db_id for row in rows
                            ),
                            "exact_structural_schema": (
                                any(
                                    rl_rows[index]["schema_structure"]
                                    == benchmark_structures[db_id]
                                    for index in matching_rows
                                )
                                if source == "RL-full"
                                else None
                            ),
                            "benchmark_only_columns": sorted(
                                benchmark_schema - best_schema
                            ),
                            "training_only_columns": sorted(
                                best_schema - benchmark_schema
                            ),
                        }
                    )
            schema_summary.append(
                {
                    "benchmark": benchmark,
                    "training_source": source,
                    "exact_column_catalog_dbs": exact_catalog_count,
                    "near_column_catalog_dbs_ge_0_85": ge_085,
                    "near_column_catalog_dbs_ge_0_95": ge_095,
                    "exact_structural_schema_dbs": (
                        exact_structure_count if source == "RL-full" else None
                    ),
                }
            )

    manual_review = []
    for candidate in question_top:
        if (
            candidate["benchmark"] == "BIRD"
            and candidate["training_source"] == "SFT-generator"
            and candidate["benchmark_row"] == 1056
            and candidate["train_row"] == 5983
        ):
            manual_review.append(
                {
                    "benchmark": "BIRD",
                    "training_source": "SFT-generator",
                    "benchmark_row": 1056,
                    "train_row": 5983,
                    "verdict": "near-semantic wording; not joint contamination",
                    "reason": (
                        "Both questions ask for a Netherlands football league/division, "
                        "but the training prompt uses a different one-table schema and a "
                        "different SQL route from the BIRD example."
                    ),
                    "reviewer_count": 1,
                    "reviewer_role": "author data audit",
                }
            )

    results = {
        "manifest": manifest,
        "profiles": profiles,
        "question_summary": question_summary,
        "question_top_matches": question_top,
        "sql_summary": sql_summary,
        "sql_exact_candidates": sql_candidates,
        "schema_summary": schema_summary,
        "schema_candidates": schema_candidates,
        "db_id_summary": db_id_summary,
        "manual_review": manual_review,
        "interpretation": {
            "high_confidence_joint_overlap_rule": (
                "Question/SQL evidence must co-occur with the same database identity "
                "or an exact/near-exact full-schema match; generic SQL alone is not leakage."
            ),
            "database_content_status": (
                "Unresolved because the SynSQL SQLite files referenced by the RL parquet "
                "are not bundled."
            ),
        },
    }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--retriever-sft-json", type=Path)
    parser.add_argument("--bird-json", type=Path, required=True)
    parser.add_argument("--bird-tables", type=Path, required=True)
    parser.add_argument("--spider-json", type=Path, required=True)
    parser.add_argument("--spider-tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "benchmark_overlap_audit_results.json"
    result_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=list) + "\n",
        encoding="utf-8",
    )
    for key in (
        "profiles",
        "question_summary",
        "question_top_matches",
        "sql_summary",
        "sql_exact_candidates",
        "schema_summary",
        "schema_candidates",
        "db_id_summary",
        "manual_review",
    ):
        pd.DataFrame(results[key]).to_csv(args.output_dir / f"{key}.csv", index=False)
    print(result_path)


if __name__ == "__main__":
    main()
