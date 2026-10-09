from collections import Counter
from typing import Any


def _canonical_cell(value: Any):
    if isinstance(value, float):
        return round(value, 6)
    return value


def _canonical_row(row):
    if not isinstance(row, (list, tuple)):
        row = (row,)
    return tuple(_canonical_cell(value) for value in row)


def grade(gold_rows, pred_rows, grading_method: str = "multiset"):
    """Compare SQL denotations under bag semantics."""
    gold = [_canonical_row(row) for row in gold_rows]
    pred = [_canonical_row(row) for row in pred_rows]
    if grading_method != "multiset":
        return gold == pred, {"method": grading_method}
    return Counter(gold) == Counter(pred), {"method": "multiset"}
