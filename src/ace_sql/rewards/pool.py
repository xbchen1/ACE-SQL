"""Empirical counts and retrieval rewards from the paper's online pool."""
from collections import Counter
from ace_sql.rewards.sql import compute_length_penalty


def updated_pool(history, current_sets, gamma):
    counts = {columns: frequency * gamma for columns, frequency in history.items()}
    for columns, frequency in Counter(frozenset(s) for s in current_sets).items():
        counts[columns] = counts.get(columns, 0.0) + frequency
    return {columns: frequency for columns, frequency in counts.items() if frequency > 0}


def target_sets(counts):
    positive = {columns: float(count) for columns, count in counts.items() if count > 0}
    if not positive:
        return []
    highest = max(positive.values())
    return [set(columns) for columns, count in positive.items() if abs(count - highest) < 1e-9]


def retrieval_reward(text, columns, targets, counts, *, mode, response_tokens=None,
                     response_limit=2048, length_penalty=True):
    """Exact-target sparse reward, or the specified coverage/noise control."""
    detail = {'pred_col_cnt': len(columns), 'ret_case': '', 'ret_length_score': 0.0,
              'ret_resp_len': response_tokens or 0, 'pool_target_num_ties': len(targets)}
    if '<think>' not in text or '</think>' not in text or '[' not in text.split('</think>')[-1]:
        detail['ret_case'] = 'FORMAT_ERROR'
        return 0.0, detail
    if response_tokens is not None and response_tokens >= response_limit:
        detail['ret_case'] = 'TRUNCATED'
        return 0.0, detail
    if not columns:
        detail['ret_case'] = 'EMPTY'
        return 0.0, detail
    if mode == 'dense':
        support = [set(key) for key, count in counts.items() if count > 0 and key]
        if not support:
            detail['ret_case'] = 'NO_POOL_TARGET'
            return 0.0, detail
        union = set().union(*support)
        coverage = max(len(columns & route) / len(route) for route in support)
        noise = len(columns - union) / len(columns)
        detail.update(ret_case='POOL_DENSE', coverage_rate=coverage, noise_count=len(columns - union))
        return coverage - 0.5 * noise, detail
    exact = any(columns == target for target in targets)
    penalty = compute_length_penalty(response_tokens) if exact and length_penalty else 0.0
    detail.update(ret_case=('POOL_EXACT_MATCH' if exact else 'POOL_EXACT_MISS') if targets else 'NO_POOL_TARGET',
                  ret_length_score=penalty)
    return float(exact) + penalty, detail
