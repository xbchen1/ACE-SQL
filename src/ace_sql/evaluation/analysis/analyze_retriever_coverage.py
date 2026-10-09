#!/usr/bin/env python3
"""
Analyze retriever schema linking quality.

Metrics (following standard schema linking evaluation):
  Recall:
    - TR (Count-based Table Recall)  = total_hit_tables / total_gold_tables
    - CR (Count-based Column Recall) = total_hit_cols / total_gold_cols
    - SR (Sample-based Recall)       = mean per-sample column recall
  Precision:
    - TP (Count-based Table Precision) = total_hit_tables / total_pred_tables
    - CP (Count-based Column Precision) = total_hit_cols / total_pred_cols
  Exact Match:
    - EM (Exact Match Rate) = samples where pred == gold exactly
"""

import json
import argparse
import re
from collections import defaultdict


def parse_gold_context(gold_context):
    """Parse CREATE TABLE statements into a set of (table, column) tuples."""
    columns = set()
    tables = re.findall(r'CREATE TABLE (\w+)\s*\((.*?)\);', gold_context, re.DOTALL)
    for table_name, body in tables:
        for line in body.split('\n'):
            line = line.strip().rstrip(',')
            if not line or line.startswith('PRIMARY') or line.startswith('FOREIGN') \
               or line.startswith('CONSTRAINT') or line.startswith(')'):
                continue
            m = re.match(r'"([^"]+)"\s+\w+', line)
            if m:
                columns.add((table_name.lower(), m.group(1).lower()))
            else:
                m = re.match(r'(\w+)\s+\w+', line)
                if m:
                    columns.add((table_name.lower(), m.group(1).lower()))
    return columns


def parse_predicted_columns(parsed_str):
    """Parse [table.col, table.col, ...] string into a set of (table, column) tuples."""
    if not parsed_str or not parsed_str.strip().startswith('['):
        return None
    inner = parsed_str.strip().strip('[]').strip()
    if not inner:
        return set()
    columns = set()
    for item in re.finditer(r'(\w+)\s*\.\s*(?:"([^"]+)"|(\w+))', inner):
        table = item.group(1).lower()
        col = (item.group(2) or item.group(3)).lower()
        columns.add((table, col))
    return columns


def extract_tables(col_set):
    """Extract table names from a set of (table, column) tuples."""
    return set(t for t, c in col_set)


def analyze(input_file, output_file=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = 0
    parse_fail = 0
    exact_match = 0
    # Column-level counts (for CR, CP)
    total_gold_cols = 0
    total_pred_cols = 0
    total_hit_cols = 0
    # Table-level counts (for TR, TP)
    total_gold_tables = 0
    total_pred_tables = 0
    total_hit_tables = 0
    # Sample-based recall sum (for SR)
    sr_sum = 0.0

    def _new_diff_stats():
        return {
            'total': 0, 'exact_match': 0, 'sr_sum': 0.0,
            'gold_cols': 0, 'pred_cols': 0, 'hit_cols': 0,
            'gold_tables': 0, 'pred_tables': 0, 'hit_tables': 0,
        }

    by_difficulty = defaultdict(_new_diff_stats)

    detailed = []

    for item in data:
        gold = parse_gold_context(item['gold_context'])
        pred = parse_predicted_columns(item['parsed_columns'])
        difficulty = item.get('difficulty', 'unknown')

        if pred is None:
            parse_fail += 1
            continue

        total += 1
        hit_cols = gold & pred
        is_exact = (gold == pred)

        gold_tabs = extract_tables(gold)
        pred_tabs = extract_tables(pred)
        hit_tabs = gold_tabs & pred_tabs

        # Sample-based column recall
        sr = len(hit_cols) / len(gold) if gold else (1.0 if not pred else 0.0)

        if is_exact:
            exact_match += 1
        sr_sum += sr
        total_gold_cols += len(gold)
        total_pred_cols += len(pred)
        total_hit_cols += len(hit_cols)
        total_gold_tables += len(gold_tabs)
        total_pred_tables += len(pred_tabs)
        total_hit_tables += len(hit_tabs)

        d = by_difficulty[difficulty]
        d['total'] += 1
        d['exact_match'] += int(is_exact)
        d['sr_sum'] += sr
        d['gold_cols'] += len(gold)
        d['pred_cols'] += len(pred)
        d['hit_cols'] += len(hit_cols)
        d['gold_tables'] += len(gold_tabs)
        d['pred_tables'] += len(pred_tabs)
        d['hit_tables'] += len(hit_tabs)

        detailed.append({
            'question_id': item.get('question_id'),
            'db_id': item.get('db_id'),
            'difficulty': difficulty,
            'gold_cols': len(gold),
            'pred_cols': len(pred),
            'hit_cols': len(hit_cols),
            'gold_tables': len(gold_tabs),
            'pred_tables': len(pred_tabs),
            'hit_tables': len(hit_tabs),
            'sr': round(sr, 4),
            'exact_match': is_exact,
        })

    # Compute global metrics
    def _safe_div(a, b):
        return a / b if b > 0 else 0.0

    TR = _safe_div(total_hit_tables, total_gold_tables)
    CR = _safe_div(total_hit_cols, total_gold_cols)
    SR = _safe_div(sr_sum, total)
    TP = _safe_div(total_hit_tables, total_pred_tables)
    CP = _safe_div(total_hit_cols, total_pred_cols)
    EM = _safe_div(exact_match, total)

    # Print results
    print("=" * 70)
    print("Schema Linking Evaluation")
    print("=" * 70)
    print(f"Total samples: {total}  (parse failed: {parse_fail})")
    print()
    print(f"{'Metric':<8} {'Description':<35} {'Value':>10}")
    print("-" * 58)
    print(f"{'EM':<8} {'Exact Match Rate':<35} {EM*100:>9.2f}%  ({exact_match}/{total})")
    print(f"{'TR':<8} {'Count-based Table Recall':<35} {TR*100:>9.2f}%  ({total_hit_tables}/{total_gold_tables})")
    print(f"{'CR':<8} {'Count-based Column Recall':<35} {CR*100:>9.2f}%  ({total_hit_cols}/{total_gold_cols})")
    print(f"{'SR':<8} {'Sample-based Recall':<35} {SR*100:>9.2f}%")
    print(f"{'TP':<8} {'Count-based Table Precision':<35} {TP*100:>9.2f}%  ({total_hit_tables}/{total_pred_tables})")
    print(f"{'CP':<8} {'Count-based Column Precision':<35} {CP*100:>9.2f}%  ({total_hit_cols}/{total_pred_cols})")

    # By difficulty
    print()
    print("=" * 70)
    print("By Difficulty")
    print("=" * 70)
    header = f"{'Diff':<13} {'N':>5} {'EM':>8} {'TR':>8} {'CR':>8} {'SR':>8} {'TP':>8} {'CP':>8}"
    print(header)
    print("-" * len(header))
    for diff in ['simple', 'moderate', 'challenging']:
        d = by_difficulty.get(diff)
        if d and d['total'] > 0:
            n = d['total']
            d_TR = _safe_div(d['hit_tables'], d['gold_tables'])
            d_CR = _safe_div(d['hit_cols'], d['gold_cols'])
            d_SR = _safe_div(d['sr_sum'], n)
            d_TP = _safe_div(d['hit_tables'], d['pred_tables'])
            d_CP = _safe_div(d['hit_cols'], d['pred_cols'])
            d_EM = _safe_div(d['exact_match'], n)
            print(f"{diff:<13} {n:>5} {d_EM*100:>7.2f}% {d_TR*100:>7.2f}% {d_CR*100:>7.2f}% {d_SR*100:>7.2f}% {d_TP*100:>7.2f}% {d_CP*100:>7.2f}%")
    print("=" * 70)

    # Save
    if output_file:
        summary = {
            'total': total,
            'parse_fail': parse_fail,
            'EM': round(EM, 4),
            'TR': round(TR, 4),
            'CR': round(CR, 4),
            'SR': round(SR, 4),
            'TP': round(TP, 4),
            'CP': round(CP, 4),
            'by_difficulty': {}
        }
        for diff, d in by_difficulty.items():
            if d['total'] > 0:
                n = d['total']
                summary['by_difficulty'][diff] = {
                    'total': n,
                    'EM': round(_safe_div(d['exact_match'], n), 4),
                    'TR': round(_safe_div(d['hit_tables'], d['gold_tables']), 4),
                    'CR': round(_safe_div(d['hit_cols'], d['gold_cols']), 4),
                    'SR': round(_safe_div(d['sr_sum'], n), 4),
                    'TP': round(_safe_div(d['hit_tables'], d['pred_tables']), 4),
                    'CP': round(_safe_div(d['hit_cols'], d['pred_cols']), 4),
                }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'detailed': detailed}, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze retriever column selection quality")
    parser.add_argument("--input", "-i", required=True, help="retriever_outputs_merged.json")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (optional)")
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
