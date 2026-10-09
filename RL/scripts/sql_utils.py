import sqlite3
import time
from contextlib import closing
from typing import Any


def _execute(sql: str, db_path: str, timeout: float) -> list[tuple[Any, ...]]:
    deadline = time.monotonic() + timeout
    with closing(sqlite3.connect(db_path, timeout=min(timeout, 5))) as conn:
        conn.text_factory = lambda value: value.decode(errors="ignore")
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()


def execute_sql_single(sql: str, db_path: str, timeout: int = 30):
    """Execute a single SQLite query and return `(rows, error)`."""
    try:
        if timeout <= 0:
            raise ValueError("SQL timeout must be positive")
        return _execute(sql, db_path, timeout), None
    except Exception as exc:
        return None, exc
