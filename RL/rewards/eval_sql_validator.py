"""
使用eval的SQL验证逻辑（仅用于验证阶段）

从evaluate_bird.py移植的SQL执行和比较逻辑
"""

import sqlite3
from func_timeout import func_timeout, FunctionTimedOut


def execute_sql(db_file, sql, timeout=5):
    """执行SQL并返回结果"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        conn.rollback()
        conn.close()
        return execution_res, True
    except Exception as e:
        conn.rollback()
        conn.close()
        return None, False


def compare_sql(db_file, pred_sql, gold_sql, timeout=5):
    """
    比较预测SQL和金标SQL的执行结果

    Returns:
        correctness (int): 1表示匹配，0表示不匹配或执行失败
    """
    try:
        result = func_timeout(timeout, _compare_sql_impl, args=(db_file, pred_sql, gold_sql))
        return result
    except (KeyboardInterrupt, FunctionTimedOut, Exception):
        return 0


def _compare_sql_impl(db_file, pred_sql, gold_sql):
    """实际的SQL比较逻辑"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    correctness = 0

    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(gold_sql)
        ground_truth_res = cursor.fetchall()

        if set(predicted_res) == set(ground_truth_res):
            correctness = 1
        conn.rollback()
    except:
        conn.rollback()
    finally:
        conn.close()

    return correctness
