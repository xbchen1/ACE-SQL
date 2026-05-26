import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any


def _execute(sql: str, db_path: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(db_path) as conn:
        conn.text_factory = lambda value: value.decode(errors="ignore")
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()


def execute_sql_single(sql: str, db_path: str, timeout: int = 30):
    """Execute a single SQLite query and return `(rows, error)`."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute, sql, db_path)
            return future.result(timeout=timeout), None
    except FuturesTimeoutError as exc:
        return None, exc
    except Exception as exc:
        return None, exc
