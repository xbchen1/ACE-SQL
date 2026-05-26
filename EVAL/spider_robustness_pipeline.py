#!/usr/bin/env python3
"""Utilities for Spider-DK/Syn/Realistic greedy evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OMNI_DIR = Path(os.environ.get("OMNI_DIR", ROOT / "omnisql_eval"))
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", OMNI_DIR / "data"))
DEFAULT_SPIDER_SOURCE = Path(os.environ.get("SPIDER_SOURCE", ROOT / "data" / "spider"))
DEFAULT_TEST_SUITE_EVAL_SOURCE = Path(
    os.environ.get("TEST_SUITE_EVAL_DIR", OMNI_DIR / "test_suite_sql_eval")
)

SPIDER_DK_REPO = "https://github.com/ygan/Spider-DK.git"
SPIDER_SYN_REPO = "https://github.com/ygan/Spider-Syn.git"
TEST_SUITE_REPO = "https://github.com/taoyds/test-suite-sql-eval.git"
TEST_SUITE_DRIVE_IDS = (
    "1iNa1WgA9tN_OFna08nq_tHZdXx9Lz2vO",
    "1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w",
)
REALISTIC_ZENODO_FILES = {
    "README.txt": "https://zenodo.org/api/records/5205322/files/README.txt/content",
    "spider-realistic.json": "https://zenodo.org/api/records/5205322/files/spider-realistic.json/content",
    "dev.json": "https://zenodo.org/api/records/5205322/files/dev.json/content",
    "tables.json": "https://zenodo.org/api/records/5205322/files/tables.json/content",
    "license": "https://zenodo.org/api/records/5205322/files/license/content",
}


DATASETS = {
    "spider_dk": {
        "input": "Spider-DK/Spider-DK.json",
        "processed": "dev_spider_dk.json",
        "gold": "Spider-DK/spider_dk_gold.sql",
        "db_path": "Spider-DK/database",
        "tables": "Spider-DK/tables.json",
        "source": "spider_dk",
        "official_metric": "EX",
    },
    "spider_syn": {
        "input": "Spider-Syn/dev.json",
        "processed": "dev_spider_syn.json",
        "gold": "Spider-Syn/spider_syn_gold.sql",
        "db_path": "spider/database",
        "tables": "spider/tables.json",
        "source": "spider_syn",
        "official_metric": "EX",
    },
    "spider_realistic": {
        "input": "spider-realistic/spider-realistic.json",
        "processed": "dev_spider_realistic.json",
        "gold": "spider-realistic/spider_realistic_gold.sql",
        "db_path": "spider/database",
        "tables": "spider/tables.json",
        "source": "spider_realistic",
        "official_metric": "EX",
    },
}

SPIDER_DK_QUERY_FIXES = {
    "SELECT DISTINCT T1.fname , T1.LName  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid":
    "SELECT DISTINCT T1.fname , T1.LName ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid",
}


def is_true(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "yes", "y"}


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False, timeout: int | None = None) -> None:
    print("Run:", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, timeout=timeout)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=src.is_dir())


def clone_repo(repo: str, dst: Path, dry_run: bool = False) -> None:
    if (dst / ".git").exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    run(["git", "clone", "--depth", "1", repo, str(dst)], dry_run=dry_run)


def download_file(url: str, dst: Path, timeout: int, dry_run: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    ensure_dir(dst.parent)
    print(f"Download: {url} -> {dst}", flush=True)
    if dry_run:
        return
    socket.setdefaulttimeout(timeout)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dst)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def patch_spider_dk_source(input_file: Path) -> int:
    if not input_file.exists():
        return 0
    data = read_json(input_file)
    patched = 0
    for item in data:
        query = item.get("query")
        if query in SPIDER_DK_QUERY_FIXES:
            item["query"] = SPIDER_DK_QUERY_FIXES[query]
            patched += 1
    if patched:
        write_json(input_file, data)
    return patched


def generate_gold(input_file: Path, gold_file: Path, question_key: str | None = None) -> int:
    data = read_json(input_file)
    ensure_dir(gold_file.parent)
    with open(gold_file, "w", encoding="utf-8") as f:
        for item in data:
            if question_key and question_key in item:
                item["question"] = item[question_key]
            sql = item.get("query") or item.get("SQL")
            db_id = item["db_id"]
            if not sql:
                raise ValueError(f"Missing SQL in {input_file}: {item}")
            f.write(f"{sql}\t{db_id}\n")
    return len(data)


def test_suite_db_path(omni_dir: Path) -> Path | None:
    candidates = [
        Path(os.environ["TEST_SUITE_DB_PATH"]) if os.environ.get("TEST_SUITE_DB_PATH") else None,
        omni_dir / "test_suite_sql_eval/test_suite_database",
        omni_dir / "test_suite_sql_eval/test_suite_database",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and any(candidate.glob("*/*.sqlite")):
            return candidate
    return None


def ensure_test_suite_eval(omni_dir: Path, downloads_dir: Path, dry_run: bool = False) -> None:
    dst = omni_dir / "test_suite_sql_eval"
    if (dst / "evaluation.py").exists():
        return
    if DEFAULT_TEST_SUITE_EVAL_SOURCE.exists():
        ensure_dir(dst)
        for src in DEFAULT_TEST_SUITE_EVAL_SOURCE.iterdir():
            if src.name == "__pycache__":
                continue
            link_or_copy(src, dst / src.name)
        return
    clone_repo(TEST_SUITE_REPO, downloads_dir / "test-suite-sql-eval", dry_run=dry_run)
    if not dry_run:
        shutil.copytree(downloads_dir / "test-suite-sql-eval", dst, dirs_exist_ok=True)


def try_download_test_suite_db(omni_dir: Path, downloads_dir: Path, timeout: int, dry_run: bool = False) -> None:
    if test_suite_db_path(omni_dir):
        return
    ensure_dir(downloads_dir)
    zip_path = downloads_dir / "test_suite_sql_eval.zip"
    for drive_id in TEST_SUITE_DRIVE_IDS:
        cmd = ["gdown", "--continue", "-O", str(zip_path), drive_id]
        try:
            run(cmd, cwd=downloads_dir, dry_run=dry_run, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"Warning: test-suite download failed for Google Drive id {drive_id}: {exc}", flush=True)
            continue
        if dry_run:
            return
        if zip_path.exists() and zip_path.stat().st_size > 0:
            run(["unzip", "-q", "-o", str(zip_path), "-d", str(omni_dir)], dry_run=dry_run)
            return


def prepare_data(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    omni_dir = Path(args.omni_dir)
    downloads_dir = Path(args.downloads_dir)
    spider_src = Path(args.spider_source)

    ensure_dir(data_dir)
    ensure_dir(downloads_dir)

    if not spider_src.exists():
        raise FileNotFoundError(f"Spider source is missing: {spider_src}")
    spider_dst = data_dir / "spider"
    ensure_dir(spider_dst)
    for name in [
        "database",
        "test_database",
        "dev.json",
        "test.json",
        "tables.json",
        "test_tables.json",
        "dev_gold.sql",
        "test_gold.sql",
    ]:
        src = spider_src / name
        if src.exists():
            link_or_copy(src, spider_dst / name)

    dk_dst = data_dir / "Spider-DK"
    if not (dk_dst / "Spider-DK.json").exists():
        dk_repo = downloads_dir / "Spider-DK"
        clone_repo(SPIDER_DK_REPO, dk_repo, dry_run=args.dry_run)
        if not args.dry_run:
            shutil.copytree(dk_repo, dk_dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
    if (dk_dst / "Spider-DK.json").exists():
        patched = patch_spider_dk_source(dk_dst / "Spider-DK.json")
        if patched:
            print(f"Patched Spider-DK source SQL fixes: {patched}", flush=True)
        ensure_dir(dk_dst / "database")
        for item in read_json(dk_dst / "Spider-DK.json"):
            db_id = item["db_id"]
            dst_db = dk_dst / "database" / db_id
            if dst_db.exists() or dst_db.is_symlink():
                continue
            for base in (spider_src / "database", spider_src / "test_database"):
                src_db = base / db_id
                if src_db.exists():
                    link_or_copy(src_db, dst_db)
                    break

    syn_dst = data_dir / "Spider-Syn"
    if not (syn_dst / "dev.json").exists():
        syn_repo = downloads_dir / "Spider-Syn"
        clone_repo(SPIDER_SYN_REPO, syn_repo, dry_run=args.dry_run)
        if not args.dry_run:
            shutil.copytree(syn_repo / "Spider-Syn", syn_dst, dirs_exist_ok=True)

    realistic_dst = data_dir / "spider-realistic"
    ensure_dir(realistic_dst)
    for filename, url in REALISTIC_ZENODO_FILES.items():
        download_file(url, realistic_dst / filename, args.download_timeout, dry_run=args.dry_run)

    ensure_test_suite_eval(omni_dir, downloads_dir, dry_run=args.dry_run)
    if args.download_test_suite:
        try_download_test_suite_db(omni_dir, downloads_dir, args.download_timeout, dry_run=args.dry_run)

    if not args.dry_run:
        counts = {
            "spider_dk": generate_gold(dk_dst / "Spider-DK.json", dk_dst / "spider_dk_gold.sql"),
            "spider_syn": generate_gold(syn_dst / "dev.json", syn_dst / "spider_syn_gold.sql", "SpiderSynQuestion"),
            "spider_realistic": generate_gold(
                realistic_dst / "spider-realistic.json",
                realistic_dst / "spider_realistic_gold.sql",
            ),
        }
        validate_data(data_dir, args.datasets.split(","))
        print("Prepared gold counts:", json.dumps(counts, indent=2), flush=True)

    ts_db = test_suite_db_path(omni_dir)
    if args.require_test_suite and any(ds in {"spider_syn", "spider_realistic"} for ds in args.datasets.split(",")):
        if ts_db is None:
            raise FileNotFoundError(
                "Missing Spider test-suite databases. Official Spider-Syn and Spider-Realistic "
                "numbers require test_suite_sql_eval/test_suite_database. Re-run with network "
                "access to Google Drive, set TEST_SUITE_DB_PATH, or set REQUIRE_TEST_SUITE=false "
                "only if you intentionally want EX fallback."
            )
    print(f"Test-suite DB: {ts_db or '<missing>'}", flush=True)


def validate_data(data_dir: Path, datasets: Iterable[str]) -> None:
    for dataset in datasets:
        if not dataset:
            continue
        spec = DATASETS[dataset]
        input_file = data_dir / spec["input"]
        gold_file = data_dir / spec["gold"]
        db_path = data_dir / spec["db_path"]
        tables = data_dir / spec["tables"]
        for path, label in [(input_file, "input"), (gold_file, "gold"), (db_path, "db_path"), (tables, "tables")]:
            if not path.exists():
                raise FileNotFoundError(f"{dataset}: missing {label}: {path}")
        missing = []
        for item in read_json(input_file):
            db_id = item["db_id"]
            if not (db_path / db_id / f"{db_id}.sqlite").exists():
                missing.append(db_id)
        if missing:
            sample = ", ".join(sorted(set(missing))[:10])
            raise FileNotFoundError(f"{dataset}: missing SQLite DBs under {db_path}: {sample}")
        print(f"{dataset}: input={len(read_json(input_file))}, dbs={len(set(i['db_id'] for i in read_json(input_file)))}", flush=True)


def load_used_db_ids(data_dir: Path, datasets: Iterable[str], db_group: str) -> set[str]:
    db_ids: set[str] = set()
    for dataset in datasets:
        if not dataset:
            continue
        spec = DATASETS[dataset]
        if db_group == "spider" and spec["db_path"] != "spider/database":
            continue
        if db_group == "spider_dk" and spec["db_path"] != "Spider-DK/database":
            continue
        for item in read_json(data_dir / spec["input"]):
            db_ids.add(item["db_id"])
    return db_ids


def index_ready(index_dir: Path) -> bool:
    return index_dir.exists() and any(index_dir.iterdir()) and any(index_dir.glob("segments_*"))


def build_indexes(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    sys.path.insert(0, str(Path(args.omni_dir)))
    from build_contents_index import build_content_index

    dataset_names = [ds for ds in args.datasets.split(",") if ds]
    jobs = [
        (
            "spider",
            data_dir / "spider/test_database",
            data_dir / "spider/db_contents_index",
            load_used_db_ids(data_dir, dataset_names, "spider"),
        ),
        (
            "spider_dk",
            data_dir / "Spider-DK/database",
            data_dir / "Spider-DK/db_contents_index",
            load_used_db_ids(data_dir, dataset_names, "spider_dk"),
        ),
    ]
    cwd = Path.cwd()
    os.chdir(Path(args.omni_dir))
    try:
        for group, db_root, index_root, db_ids in jobs:
            if not db_ids:
                continue
            ensure_dir(index_root)
            print(f"Build/check BM25 indexes for {group}: {len(db_ids)} dbs", flush=True)
            for db_id in sorted(db_ids):
                index_dir = index_root / db_id
                if index_ready(index_dir) and not args.force:
                    continue
                db_file = db_root / db_id / f"{db_id}.sqlite"
                if not db_file.exists() and group == "spider":
                    db_file = data_dir / "spider/database" / db_id / f"{db_id}.sqlite"
                if not db_file.exists():
                    raise FileNotFoundError(f"Cannot build index; missing DB file: {db_file}")
                if args.dry_run:
                    print(f"Would build index: {db_file} -> {index_dir}", flush=True)
                    continue
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                build_content_index(str(db_file), str(index_dir))
    finally:
        os.chdir(cwd)


def process_datasets(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    omni_dir = Path(args.omni_dir)
    for dataset in [ds for ds in args.datasets.split(",") if ds]:
        spec = DATASETS[dataset]
        output = data_dir / spec["processed"]
        if output.exists() and not args.force:
            print(f"Skip existing processed prompts: {output}", flush=True)
            continue
        cmd = [
            "python3",
            "process_dataset.py",
            "--input_data_file",
            f"./data/{spec['input']}",
            "--output_data_file",
            f"./data/{spec['processed']}",
            "--db_path",
            f"./data/{spec['db_path']}",
            "--tables",
            f"./data/{spec['tables']}",
            "--source",
            spec["source"],
            "--mode",
            "dev",
            "--value_limit_num",
            str(args.value_limit),
            "--db_content_index_path",
            "./data/Spider-DK/db_contents_index" if dataset == "spider_dk" else "./data/spider/db_contents_index",
        ]
        run(cmd, cwd=omni_dir, dry_run=args.dry_run)


def evaluate_predictions(args: argparse.Namespace) -> None:
    dataset = args.dataset
    spec = DATASETS[dataset]
    data_dir = Path(args.data_dir)
    omni_dir = Path(args.omni_dir)
    sys.path.insert(0, str(omni_dir))
    from evaluate_spider import run_spider_eval

    ts_path = Path(args.test_suite_db_path) if args.test_suite_db_path else test_suite_db_path(omni_dir)
    use_ts = spec["official_metric"] == "TS"
    if use_ts and ts_path is None and args.require_test_suite:
        raise FileNotFoundError(f"{dataset} requires test-suite DB for official TS metric")

    ex_acc, ts_acc = run_spider_eval(
        str(data_dir / spec["gold"]),
        str(Path(args.pred)),
        str(data_dir / spec["db_path"]),
        str(ts_path) if use_ts and ts_path else "",
        "greedy_search",
        True,
    )
    metric_used = spec["official_metric"] if (not use_ts or ts_acc is not None) else "EX_FALLBACK_NO_TEST_SUITE"
    official = ts_acc if use_ts and ts_acc is not None else ex_acc
    result = {
        "dataset": dataset,
        "mode": "greedy_search",
        "official_metric": spec["official_metric"],
        "metric_used": metric_used,
        "official_score": official,
        "ex_acc": ex_acc,
        "ts_acc": ts_acc,
        "pred": str(Path(args.pred)),
        "gold": str(data_dir / spec["gold"]),
        "db_path": str(data_dir / spec["db_path"]),
        "ts_db_path": str(ts_path) if ts_path else "",
    }
    output_json = Path(args.output_json)
    ensure_dir(output_json.parent)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def collect_results(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    results = []
    for dataset in [ds for ds in args.datasets.split(",") if ds]:
        path = output_dir / dataset / "result_greedy.json"
        if path.exists():
            results.append(read_json(path))
    summary = output_dir / "summary_greedy.json"
    ensure_dir(summary.parent)
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {summary}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni_dir", default=str(OMNI_DIR))
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--datasets", default="spider_dk,spider_syn,spider_realistic")
    parser.add_argument("--dry_run", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data")
    p.add_argument("--spider_source", default=str(DEFAULT_SPIDER_SOURCE))
    p.add_argument("--downloads_dir", default=str(OMNI_DIR / "_downloads"))
    p.add_argument("--download_timeout", type=int, default=60)
    p.add_argument("--download_test_suite", action="store_true")
    p.add_argument("--require_test_suite", action="store_true")
    p.set_defaults(func=prepare_data)

    p = sub.add_parser("build-indexes")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=build_indexes)

    p = sub.add_parser("process")
    p.add_argument("--force", action="store_true")
    p.add_argument("--value_limit", type=int, default=2)
    p.set_defaults(func=process_datasets)

    p = sub.add_parser("evaluate")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--pred", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--test_suite_db_path", default="")
    p.add_argument("--require_test_suite", action="store_true")
    p.set_defaults(func=evaluate_predictions)

    p = sub.add_parser("collect-results")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=collect_results)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
