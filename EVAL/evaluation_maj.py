"""
Evaluation script for GenLink pipeline results.
CPU-only execution with multiprocessing support.
Supports: greedy_search, major_voting, pass@k, and self_consistency modes.
"""

import sys
import sqlite3
import json
import argparse
import os
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp
import random
from config import BIRD_DEV_DATA, BIRD_DB_PATH

random.seed(42)

execution_results = None
evaluation_results = None


def parse_option():
    parser = argparse.ArgumentParser(description="Evaluate GenLink pipeline results on BIRD benchmark")
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to prediction file (final_results.json or mmsg_results.json)')
    parser.add_argument('--gold', type=str,
                        default=BIRD_DEV_DATA,
                        help='Path to gold standard file')
    parser.add_argument('--db_path', type=str,
                        default=BIRD_DB_PATH,
                        help='Path to database directory')
    parser.add_argument('--mode', type=str, default="self_consistency",
                        choices=["greedy_search", "major_voting", "pass@k", "self_consistency"],
                        help='Evaluation mode')
    parser.add_argument('--num_cpus', type=int, default=None,
                        help='Number of CPUs for parallel execution (default: all available)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout for each SQL execution (seconds)')
    parser.add_argument('--save_pred_sqls', action='store_true',
                        help='Save predicted SQLs to file')

    opt = parser.parse_args()
    return opt


def execute_sql(data_idx, db_file, sql):
    """Execute SQL and return results."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res)  # make set hashable
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1
    except:
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0


def compare_sql(question_id, db_file, question, ground_truth, pred_sql):
    """Compare predicted SQL with ground truth."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    correctness = 0

    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        if set(predicted_res) == set(ground_truth_res):
            correctness = 1
        conn.rollback()
    except:
        conn.rollback()
    finally:
        conn.close()
    return question_id, db_file, question, ground_truth, pred_sql, correctness


def compare_sql_wrapper(args, timeout):
    """Wrap compare_sql for timeout."""
    try:
        result = func_timeout(timeout, compare_sql, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = (*args, 0)
    except Exception as e:
        result = (*args, 0)
    return result


def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    """Wrap execute_sql for timeout."""
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        res = (data_idx, db_file, sql, None, 0)
    return res


def execute_callback_evaluate_sql(result):
    """Store the evaluation result."""
    question_id, db_file, question, ground_truth, pred_sql, correctness = result
    evaluation_results.append({
        "question_id": question_id,
        "db_file": db_file,
        "question": question,
        "ground_truth": ground_truth,
        "pred_sql": pred_sql,
        "correctness": correctness
    })
    sys.stdout.flush()
    sys.stderr.flush()


def execute_callback_execute_sqls(result):
    """Store the execution result."""
    data_idx, db_file, sql, query_result, valid = result
    execution_results.append({
        "data_idx": data_idx,
        "db_file": db_file,
        "sql": sql,
        "query_result": query_result,
        "valid": valid
    })


def evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=1, timeout=30):
    """Execute SQL evaluation in parallel."""
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_file, question, pred_sql, ground_truth in zip(
        range(len(db_files)), db_files, questions, pred_sqls, ground_truth_sqls
    ):
        pool.apply_async(
            compare_sql_wrapper,
            args=((question_id, db_file, question, ground_truth, pred_sql), timeout),
            callback=execute_callback_evaluate_sql
        )
    pool.close()
    pool.join()


def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=30):
    """Execute SQLs in parallel."""
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(range(len(sqls)), db_files, sqls):
        pool.apply_async(
            execute_sql_wrapper,
            args=(data_idx, db_file, sql, timeout),
            callback=execute_callback_execute_sqls
        )
    pool.close()
    pool.join()


def major_voting(db_files, pred_sqls, sampling_num, num_cpus=20, timeout=30, return_random_one_when_all_errors=True):
    """Perform major voting on multiple SQL predictions."""
    global execution_results
    mj_pred_sqls = []
    execution_results = []

    # Execute all sampled SQL queries
    print(f"\n[DEBUG] Starting major voting...")
    print(f"[DEBUG] Total SQL queries to execute: {len(pred_sqls)}")
    print(f"[DEBUG] Sampling num per question: {sampling_num}")
    print(f"[DEBUG] Number of questions: {len(pred_sqls) // sampling_num}")
    print(f"[DEBUG] Using {num_cpus} CPUs, timeout={timeout}s")
    print(f"[DEBUG] Executing SQL queries in parallel...")

    execute_sqls_parallel(db_files, pred_sqls, num_cpus=num_cpus, timeout=timeout)
    execution_results = sorted(execution_results, key=lambda x: x['data_idx'])

    valid_count = sum(1 for r in execution_results if r['valid'] == 1)
    print(f"[DEBUG] Execution completed. Total: {len(execution_results)}, Valid: {valid_count}, Invalid: {len(execution_results) - valid_count}")

    # Perform major voting
    for result_idx in range(0, len(execution_results), sampling_num):
        major_voting_counting = dict()
        execution_results_of_one_sample = execution_results[result_idx: result_idx + sampling_num]

        # If no predicted SQLs are valid
        if sum([res["valid"] for res in execution_results_of_one_sample]) == 0:
            if return_random_one_when_all_errors:
                mj_pred_sql = random.choice(execution_results_of_one_sample)["sql"]
            else:
                mj_pred_sql = "Error SQL"
            mj_pred_sqls.append(mj_pred_sql)
            continue

        for res in execution_results_of_one_sample:
            if res["valid"] == 1:
                if res["query_result"] in major_voting_counting:
                    major_voting_counting[res["query_result"]]["votes"] += 1
                else:
                    major_voting_counting[res["query_result"]] = {"votes": 1, "sql": res["sql"]}

        # Find the SQL with max votes
        major_vote = max(major_voting_counting.values(), key=lambda x: x["votes"])
        mj_pred_sql = major_vote["sql"]
        mj_pred_sqls.append(mj_pred_sql)

    return mj_pred_sqls


def compute_acc_by_diff(evaluation_results, gold_data):
    """Compute accuracy by difficulty level."""
    simple_results = []
    moderate_results = []
    challenging_results = []

    for i, content in enumerate(gold_data):
        if i >= len(evaluation_results):
            break
        difficulty = content.get('difficulty', 'simple')
        if difficulty == 'simple':
            simple_results.append(evaluation_results[i])
        elif difficulty == 'moderate':
            moderate_results.append(evaluation_results[i])
        elif difficulty == 'challenging':
            challenging_results.append(evaluation_results[i])

    simple_acc = sum([res['correctness'] for res in simple_results]) / len(simple_results) if simple_results else 0
    moderate_acc = sum([res['correctness'] for res in moderate_results]) / len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['correctness'] for res in challenging_results]) / len(challenging_results) if challenging_results else 0
    total_acc = sum([res['correctness'] for res in evaluation_results]) / len(evaluation_results) if evaluation_results else 0

    return {
        'simple': {'count': len(simple_results), 'accuracy': simple_acc * 100},
        'moderate': {'count': len(moderate_results), 'accuracy': moderate_acc * 100},
        'challenging': {'count': len(challenging_results), 'accuracy': challenging_acc * 100},
        'total': {'count': len(evaluation_results), 'accuracy': total_acc * 100}
    }


def print_results(acc_by_diff):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 90)
    print("BIRD Execution Accuracy Evaluation Results")
    print("=" * 90)
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [acc_by_diff[l]['count'] for l in levels]
    accs = [acc_by_diff[l]['accuracy'] for l in levels]
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *counts))
    print("-" * 90)
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy (%)', *accs))
    print("=" * 90)


def run_eval(gold_file, pred_file, db_path, mode, save_pred_sqls, num_cpus=20, timeout=30):
    """Run evaluation based on mode."""
    global evaluation_results

    print("=" * 60)
    print("GenLink Evaluation Script (CPU-only)")
    print("=" * 60)
    print(f"Prediction file: {pred_file}")
    print(f"Gold file: {gold_file}")
    print(f"Database path: {db_path}")
    print(f"Mode: {mode}")
    print(f"Num CPUs: {num_cpus}")
    print(f"Timeout: {timeout}s")
    print("=" * 60)

    gold = json.load(open(gold_file))
    pred_results = json.load(open(pred_file))

    db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]
    questions = [data["question"] for data in gold]

    # Get ground truth SQLs
    if "bird" in gold_file.lower():
        ground_truth_sqls = [data["SQL"] for data in gold]
    else:
        ground_truth_sqls = [data.get("query") or data.get("SQL") for data in gold]

    # Determine pred_sql_key based on file content
    if "optimal_sql" in pred_results[0]:
        # This is final_results.json from self-consistency selection
        pred_sql_key = "optimal_sql"
    elif "pred_sqls" in pred_results[0]:
        pred_sql_key = "pred_sqls"
    else:
        raise ValueError("Cannot find prediction key in pred_results")

    print(f"Using prediction key: {pred_sql_key}")

    if mode == "self_consistency":
        # Use optimal_sql from self-consistency selection
        if pred_sql_key != "optimal_sql":
            raise ValueError("self_consistency mode requires optimal_sql field (use final_results.json)")

        pred_sqls = [res["optimal_sql"] for res in pred_results]

        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_self_consistency_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(pred_sqls, indent=2, ensure_ascii=False))

        assert len(pred_results) == len(pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        print(f"\nEvaluating {len(pred_sqls)} predictions...")
        evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

        evaluation_results = sorted(evaluation_results, key=lambda x: x['question_id'])
        evaluation_scores = [res["correctness"] for res in evaluation_results]

        acc = sum(evaluation_scores) / len(evaluation_scores)
        print(f"\nEX Accuracy (self-consistency): {acc:.4f} ({sum(evaluation_scores)}/{len(evaluation_scores)})")

        acc_by_diff = compute_acc_by_diff(evaluation_results, gold)
        print_results(acc_by_diff)

        return acc, pred_sqls

    elif mode == "greedy_search":
        # Use first prediction
        if pred_sql_key == "optimal_sql":
            pred_sqls = [res["optimal_sql"] for res in pred_results]
        else:
            pred_sqls = [res[pred_sql_key][0] for res in pred_results]

        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_greedy_search_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(pred_sqls, indent=2, ensure_ascii=False))

        assert len(pred_results) == len(pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        print(f"\nEvaluating {len(pred_sqls)} predictions...")
        evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

        evaluation_results = sorted(evaluation_results, key=lambda x: x['question_id'])
        evaluation_scores = [res["correctness"] for res in evaluation_results]

        acc = sum(evaluation_scores) / len(evaluation_scores)
        print(f"\nEX Accuracy (greedy search): {acc:.4f} ({sum(evaluation_scores)}/{len(evaluation_scores)})")

        acc_by_diff = compute_acc_by_diff(evaluation_results, gold)
        print_results(acc_by_diff)

        return acc, pred_sqls

    elif mode == "major_voting":
        if pred_sql_key == "optimal_sql":
            raise ValueError("major_voting mode requires pred_sqls field (use mmsg_results.json)")

        sampling_num = len(pred_results[0][pred_sql_key])
        print(f"\n[DEBUG] Mode: major_voting")
        print(f"[DEBUG] sampling_num: {sampling_num}")
        print(f"[DEBUG] Total predictions: {len(pred_results)}")

        # Expand db_files for all samples
        db_files_expanded = []
        for gold_data in gold:
            db_files_expanded.extend([os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite")] * sampling_num)

        pred_sqls_expanded = []
        for pred_data in pred_results:
            pred_sqls_expanded.extend(pred_data[pred_sql_key])
        assert len(pred_sqls_expanded) == len(db_files_expanded)

        mj_pred_sqls = major_voting(db_files_expanded, pred_sqls_expanded, sampling_num, num_cpus=num_cpus, timeout=timeout)

        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_major_voting_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(mj_pred_sqls, indent=2, ensure_ascii=False))

        # Reset db_files
        db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]

        assert len(mj_pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        print(f"\nEvaluating {len(mj_pred_sqls)} predictions...")
        evaluate_sqls_parallel(db_files, questions, mj_pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

        evaluation_results = sorted(evaluation_results, key=lambda x: x['question_id'])
        evaluation_scores = [res["correctness"] for res in evaluation_results]

        acc = sum(evaluation_scores) / len(evaluation_scores)
        print(f"\nEX Accuracy (major voting): {acc:.4f} ({sum(evaluation_scores)}/{len(evaluation_scores)})")

        acc_by_diff = compute_acc_by_diff(evaluation_results, gold)
        print_results(acc_by_diff)

        return acc, mj_pred_sqls

    elif mode == "pass@k":
        if pred_sql_key == "optimal_sql":
            raise ValueError("pass@k mode requires pred_sqls field (use mmsg_results.json)")

        all_scores = []
        sampling_num = len(pred_results[0][pred_sql_key])
        print(f"sampling_num: {sampling_num}")

        db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]

        for sample_idx in range(sampling_num):
            pred_sqls_for_specific_sample_idx = [pred_data[pred_sql_key][sample_idx] for pred_data in pred_results]
            evaluation_results = []
            evaluate_sqls_parallel(db_files, questions, pred_sqls_for_specific_sample_idx, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)
            evaluation_results = sorted(evaluation_results, key=lambda x: x['question_id'])
            evaluation_scores = [res["correctness"] for res in evaluation_results]
            all_scores.append(evaluation_scores)

        pass_at_k_scores = [1 if any(column) else 0 for column in zip(*all_scores)]
        acc = sum(pass_at_k_scores) / len(pass_at_k_scores)
        print(f"\nEX Accuracy (pass@{sampling_num}): {acc:.4f} ({sum(pass_at_k_scores)}/{len(pass_at_k_scores)})")

        return acc, None

    else:
        raise ValueError("mode should be in [greedy_search, major_voting, pass@k, self_consistency]")


if __name__ == "__main__":
    opt = parse_option()
    # Use all available CPUs if not specified
    num_cpus = opt.num_cpus if opt.num_cpus else mp.cpu_count()
    print(f"[DEBUG] Using {num_cpus} CPUs (available: {mp.cpu_count()})")
    run_eval(opt.gold, opt.pred, opt.db_path, opt.mode, opt.save_pred_sqls, num_cpus, opt.timeout)
