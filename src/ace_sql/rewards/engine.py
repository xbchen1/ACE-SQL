"""Trainer-independent execution credit, empirical pool, and durable state.

The engine scores exactly one question and two role groups at a time. Training
updates the pool before retrieval credit; evaluation never mutates it. There
is no dependency on slime, distributed training, or a legacy reward manager.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path
from threading import RLock

from ace_sql.data.paths import resolve_db_path
from ace_sql.data.rl import as_object, atomic_write_text, question_key, validate_metadata
from ace_sql.schema.retrieval import parse_retriever_output
from .pool import retrieval_reward, target_sets, updated_pool
from .sql import compute_sql_reward, extract_columns_from_sql, extract_sql_from_response

METHOD_VERSION = 'arr-20261009-union-updated-pool'


class RewardEngine:
    def __init__(self, initial_pool_path=None, max_workers=8, pool_gamma=0.5,
                 group_size=8, target_mode='online', length_penalty=True):
        if group_size <= 0 or max_workers <= 0:
            raise ValueError('group_size and max_workers must be positive')
        if not math.isfinite(pool_gamma) or not 0 <= pool_gamma <= 1:
            raise ValueError('pool_gamma must be finite and in [0, 1]')
        if target_mode not in {'online', 'frozen', 'gold', 'dense'}:
            raise ValueError('target_mode must be online, frozen, gold, or dense')
        self.group_size, self.pool_gamma = group_size, pool_gamma
        self.target_mode, self.length_penalty = target_mode, bool(length_penalty)
        self.pool = {}
        self._lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        if initial_pool_path:
            self.pool = self._decode_pool(json.loads(Path(initial_pool_path).read_text()))

    @staticmethod
    def _decode_pool(raw):
        result = {}
        for key, entries in as_object(raw, 'pool').items():
            result[key] = {}
            for columns, frequency in as_object(entries, 'pool entry').items():
                frequency = float(frequency)
                if not math.isfinite(frequency) or frequency < 0:
                    raise ValueError('pool frequency must be finite and nonnegative')
                result[key][frozenset(int(col) for col in columns.split(',') if col)] = frequency
        return result

    def compute_group(self, ret_texts, gen_texts, metadata, *, ret_resp_lens=None,
                      gen_resp_lens=None, update_pool=True, ret_response_length=2048):
        n = 1 if not update_pool and len(ret_texts) == len(gen_texts) == 1 else self.group_size
        if len(ret_texts) != n or len(gen_texts) != n:
            raise ValueError(f'both roles must have exactly {n} samples per question')
        gt, extra = validate_metadata(metadata)
        key = question_key(gt)
        database = Path(resolve_db_path(gt['db_path'], gt['db_id'])).resolve()
        if not database.is_file():
            raise FileNotFoundError(f'SQLite database missing for {key}: {database}')
        gt = {**gt, 'db_path': str(database)}
        schema = as_object(extra['db_info'], 'db_info')
        for name, lengths in (('ret_resp_lens', ret_resp_lens), ('gen_resp_lens', gen_resp_lens)):
            if lengths is not None and (len(lengths) != n or any(type(x) is not int or x < 0 for x in lengths)):
                raise ValueError(f'{name} must contain {n} nonnegative integer token lengths')
        ret_lengths = ret_resp_lens if ret_resp_lens is not None else [None] * n
        gen_lengths = gen_resp_lens if gen_resp_lens is not None else [None] * n
        with self._lock:
            if key not in self.pool:
                raise ValueError(f'no historical pool target for {key}')
            before = deepcopy(self.pool[key])
            try:
                futures = [self.executor.submit(compute_sql_reward, text, gt, length, self.length_penalty)
                           for text, length in zip(gen_texts, gen_lengths, strict=True)]
                scored = [future.result() for future in futures]
                sqls = [extract_sql_from_response(text)[0] for text in gen_texts]
                current = []
                for sql, (_, _, detail) in zip(sqls, scored, strict=True):
                    if detail['exec_status'] == 'Match' and sql:
                        ok, columns, _ = extract_columns_from_sql(sql, str(database), schema)
                        if ok and columns:
                            current.append(columns)
                counts = before
                if update_pool and self.target_mode in {'online', 'dense'}:
                    counts = updated_pool(before, current, self.pool_gamma)
                targets = [set(gt['gold_column_indices'])] if self.target_mode == 'gold' else target_sets(counts)
                ret_rewards, gen_rewards, details = [], [], []
                for text, length, sql, (reward, base, sql_detail) in zip(ret_texts, ret_lengths, sqls, scored, strict=True):
                    columns = parse_retriever_output(text, schema)
                    ret_reward, ret_detail = retrieval_reward(
                        text, columns, targets, counts, mode=self.target_mode,
                        response_tokens=length, response_limit=ret_response_length,
                        length_penalty=self.length_penalty)
                    ret_rewards.append(ret_reward)
                    gen_rewards.append(reward)
                    details.append({'db_id': gt['db_id'], 'gold_sql': gt['sql'], 'pred_sql': sql,
                                    'sql_reward': reward, 'sql_no_len': base, 'ret_reward': ret_reward,
                                    **sql_detail, **ret_detail})
                if update_pool and self.target_mode in {'online', 'dense'}:
                    self.pool[key] = counts
                return ret_rewards, gen_rewards, details
            except BaseException:
                self.pool[key] = before
                raise

    def state_dict(self):
        with self._lock:
            pool = {key: {','.join(map(str, sorted(columns))): frequency for columns, frequency in entries.items()}
                    for key, entries in self.pool.items()}
            return {'version': 2, 'method_version': METHOD_VERSION, 'group_size': self.group_size,
                    'target_mode': self.target_mode, 'length_penalty': self.length_penalty,
                    'pool_gamma': self.pool_gamma, 'pool': pool}

    def load_state_dict(self, state):
        if (state.get('version') != 2 or state.get('method_version') != METHOD_VERSION
                or state.get('group_size') != self.group_size or state.get('target_mode') != self.target_mode
                or state.get('length_penalty') != self.length_penalty):
            raise ValueError('incompatible reward checkpoint version or method')
        if state.get('pool_gamma') != self.pool_gamma:
            raise ValueError('reward checkpoint pool decay differs from current run')
        pool = self._decode_pool(state.get('pool'))
        with self._lock:
            self.pool = pool

    def save_pool(self, path):
        atomic_write_text(path, json.dumps(self.state_dict()['pool'], indent=2, allow_nan=False) + '\n')

    def close(self):
        self.executor.shutdown(wait=True)
