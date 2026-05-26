"""Configuration for Spider dev/test evaluation."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path | str) -> str:
    return os.environ.get(name, str(default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


SPIDER_DATA_DIR = Path(os.environ.get("SPIDER_DATA_DIR", PROJECT_ROOT / "data" / "spider"))

SPIDER_DEV_DATA = _env_path("SPIDER_DEV_DATA", SPIDER_DATA_DIR / "dev.json")
SPIDER_TEST_DATA = _env_path("SPIDER_TEST_DATA", SPIDER_DATA_DIR / "test.json")
SPIDER_TABLES = _env_path("SPIDER_TABLES", SPIDER_DATA_DIR / "tables.json")
SPIDER_TEST_TABLES = _env_path("SPIDER_TEST_TABLES", SPIDER_DATA_DIR / "test_tables.json")
SPIDER_DB_PATH = _env_path("SPIDER_DB_PATH", SPIDER_DATA_DIR / "database")
SPIDER_TEST_DB_PATH = _env_path("SPIDER_TEST_DB_PATH", SPIDER_DATA_DIR / "test_database")
SPIDER_DEV_GOLD = _env_path("SPIDER_DEV_GOLD", SPIDER_DATA_DIR / "dev_gold.sql")
SPIDER_TEST_GOLD = _env_path("SPIDER_TEST_GOLD", SPIDER_DATA_DIR / "test_gold.sql")

OMNISQL_EVAL_DIR = _env_path("OMNISQL_EVAL_DIR", PROJECT_ROOT / "omnisql_eval")
TEST_SUITE_EVAL_DIR = _env_path("TEST_SUITE_EVAL_DIR", Path(OMNISQL_EVAL_DIR) / "test_suite_sql_eval")

RETRIEVER_PROMPT_PATH = str(PROJECT_ROOT / "prompts" / "retriever_prompt.txt")

VLLM_CONFIG = {
    "dtype": os.environ.get("VLLM_DTYPE", "bfloat16"),
    "tensor_parallel_size": _env_int("TENSOR_PARALLEL_SIZE", 1),
    "max_model_len": _env_int("MAX_MODEL_LEN", 8192),
    "gpu_memory_utilization": float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95")),
    "swap_space": _env_int("VLLM_SWAP_SPACE", 42),
    "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "true").lower() in {"1", "true", "yes"},
    "disable_custom_all_reduce": os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "true").lower()
    in {"1", "true", "yes"},
    "trust_remote_code": os.environ.get("TRUST_REMOTE_CODE", "true").lower() in {"1", "true", "yes"},
    "max_num_seqs": _env_int("VLLM_MAX_NUM_SEQS", 256),
}

RETRIEVER_MAX_TOKENS = _env_int("RETRIEVER_MAX_TOKENS", 4096)
SQL_MAX_TOKENS = _env_int("SQL_MAX_TOKENS", 8192)
VALUE_LIMIT_NUM = _env_int("VALUE_LIMIT_NUM", 2)
