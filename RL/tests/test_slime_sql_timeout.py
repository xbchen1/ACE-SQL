import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from scripts.sql_utils import execute_sql_single


class SQLTimeoutTest(unittest.TestCase):
    def test_deadline_interrupts_query_and_does_not_wait_on_a_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "test.sqlite")
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE t(x INTEGER)")
                connection.execute("INSERT INTO t VALUES (1)")
            started = time.monotonic()
            rows, error = execute_sql_single(
                "WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t) SELECT sum(x) FROM t",
                database, timeout=0.02,
            )
            self.assertIsNone(rows)
            self.assertIsNotNone(error)
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertEqual(execute_sql_single("SELECT x FROM t", database)[0], [(1,)])
            self.assertIsNotNone(execute_sql_single("DELETE FROM t", database)[1])
            self.assertEqual(execute_sql_single("SELECT x FROM t", database)[0], [(1,)])


if __name__ == "__main__":
    unittest.main()
