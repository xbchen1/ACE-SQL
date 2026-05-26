import json
import argparse
import os
import random
import re
from pathlib import Path
from evaluate_bird import major_voting, mark_invalid_sqls
import tempfile
import subprocess

random.seed(42)

def _resolve_test_suite_eval_script():
    candidate_paths = [
        Path(os.environ["TEST_SUITE_EVAL_DIR"]) / "evaluation.py"
        if os.environ.get("TEST_SUITE_EVAL_DIR")
        else None,
        Path(__file__).resolve().parent / "test_suite_sql_eval" / "evaluation.py",
    ]

    for candidate in candidate_paths:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find test_suite_sql_eval/evaluation.py. Tried: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def _parse_execution_accuracy(stdout):
    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]

    for idx, line in enumerate(lines):
        if "EXECUTION ACCURACY" not in line:
            continue
        for next_line in lines[idx + 1:]:
            stripped = next_line.strip()
            if not stripped.startswith("execution"):
                continue
            values = re.findall(r'\d+\.\d+', stripped)
            if len(values) >= 5:
                return float(values[4])
            if values:
                return float(values[-1])

    match = re.search(r'(\d+\.\d+)\s*$', stdout.strip())
    if match:
        return float(match.group(1))

    raise RuntimeError(f"Failed to parse execution accuracy from evaluator stdout:\n{stdout}")

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type = str, default = "predict_dev.json")
    parser.add_argument('--gold', type = str, default = "./data/spider/dev_gold.sql")
    parser.add_argument('--db_path', type = str, default = "./data/spider/databases")
    parser.add_argument('--ts_db_path', type = str, default = "")
    parser.add_argument('--mode', type = str, default = "greedy_search")

    opt = parser.parse_args()

    return opt

def format_sql(sql):
    sql = sql.strip()
    # remove multi-line comments /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # remove single-line comments --
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)

    sql = sql.replace("\n", " ").replace("\t", " ")
    sql = sql.strip()
    
    if sql == "":
        sql = "Error SQL"

    return sql
    
def run_spider_eval(gold_file, pred_file, db_path, ts_db_path, mode, save_pred_sqls):
    assert mode in ["greedy_search", "major_voting"]
    gold_sqls = [line.split("\t")[0].strip() for line in open(gold_file).readlines()]
    db_ids = [line.split("\t")[1].strip() for line in open(gold_file).readlines()]
    pred = json.load(open(pred_file))
    pred_sql_key = "pred_sqls"
    # pred_sql_key = "responses"

    pred_sqls = []
    if mode == "greedy_search":
        pred_sqls = [pred_data[pred_sql_key][0] for pred_data in pred]
        assert len(pred_sqls) == len(db_ids)
        db_files = [os.path.join(db_path, db_id, db_id + ".sqlite") for db_id in db_ids]
        pred_sqls = mark_invalid_sqls(db_files, pred_sqls)
    elif mode == "major_voting":
        # perform major voting using the BIRD's evaluation script
        sampling_num = len(pred[0][pred_sql_key])
        print("sampling_num:", sampling_num)

        all_db_files = []
        for db_id in db_ids:
            all_db_files.extend([os.path.join(db_path, db_id, db_id + ".sqlite")] * sampling_num)

        all_pred_sqls = []
        for pred_data in pred:
            all_pred_sqls.extend(pred_data[pred_sql_key])
        assert len(all_db_files) == len(all_pred_sqls)

        pred_sqls = major_voting(all_db_files, all_pred_sqls, sampling_num, False)

    pred_sqls = [format_sql(pred_sql) for pred_sql in pred_sqls]
    assert len(pred_sqls) == len(gold_sqls)
    
    if save_pred_sqls:
        with open(pred_file[:-5] + f"_pred_{mode}_sqls.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(pred_sqls, indent=2 ,ensure_ascii=False))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as temp_file:
        for pred_sql in pred_sqls:
            temp_file.write(pred_sql + "\n")
        temp_file_name = temp_file.name
        print(temp_file_name)

    eval_script = _resolve_test_suite_eval_script()
    eval_cwd = eval_script.parent.parent
    
    print("Execution accuracy:")
    cmd = [
        "python3", "-u", "-m", "test_suite_sql_eval.evaluation",
        "--gold", gold_file,
        "--pred", temp_file_name,
        "--db", db_path,
        "--etype", "exec",
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, cwd=str(eval_cwd))
    stdout = result.stdout
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Spider evaluator failed with code {result.returncode}.\nSTDERR:\n{result.stderr}\nSTDOUT:\n{stdout}"
        )
    ex_acc = _parse_execution_accuracy(stdout)
    print(stdout)
    print("ex_acc:", ex_acc)

    ts_acc = None
    if ts_db_path != "":
        print("Test suit execution accuracy:")
        cmd = [
            "python3", "-u", "-m", "test_suite_sql_eval.evaluation",
            "--gold", gold_file,
            "--pred", temp_file_name,
            "--db", ts_db_path,
            "--etype", "exec",
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, cwd=str(eval_cwd))
        stdout = result.stdout
        print(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"Spider test-suite evaluator failed with code {result.returncode}.\nSTDERR:\n{result.stderr}\nSTDOUT:\n{stdout}"
            )
        ts_acc = _parse_execution_accuracy(stdout)
        print(stdout)
        print("ts_acc:", ts_acc)

    os.remove(temp_file_name)

    return ex_acc, ts_acc

if __name__ == "__main__":
    opt = parse_option()
    run_spider_eval(opt.gold, opt.pred, opt.db_path, opt.ts_db_path, opt.mode, False)
