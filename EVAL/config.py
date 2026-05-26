"""Configuration for the BIRD evaluation pipeline.

All dataset, database, model, and output paths can be overridden with
environment variables. Defaults are intentionally relative placeholders so the
package can be published without local machine paths.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path | str) -> str:
    return os.environ.get(name, str(default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


# Model paths
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "models/base_model")
SQL_MODEL_PATH = os.environ.get("SQL_MODEL_PATH", BASE_MODEL_PATH)

# BIRD dataset paths
BIRD_DATA_DIR = Path(os.environ.get("BIRD_DATA_DIR", PROJECT_ROOT / "data" / "bird_dev"))
BIRD_DEV_DATA = _env_path("BIRD_DEV_DATA", BIRD_DATA_DIR / "dev.json")
BIRD_TABLES = _env_path("BIRD_TABLES", BIRD_DATA_DIR / "dev_tables.json")
BIRD_DB_PATH = _env_path("BIRD_DB_PATH", BIRD_DATA_DIR / "dev_databases")
BIRD_DB_CONTENT_INDEX_PATH = _env_path(
    "BIRD_DB_CONTENT_INDEX_PATH",
    BIRD_DATA_DIR / "db_contents_index",
)

# Compatibility variable kept for older modules; local templates are used.
GENERATE_PROMPTS_DIR = str(PROJECT_ROOT)

# Prompt and output paths
RETRIEVER_PROMPT_PATH = str(PROJECT_ROOT / "prompts" / "retriever_prompt.txt")
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))

# Optional veRL source tree for FSDP checkpoint merge. Leave empty if verl is
# installed in the active environment.
VERL_ROOT = os.environ.get("VERL_ROOT", "")

# vLLM configuration
VLLM_CONFIG = {
    "dtype": os.environ.get("VLLM_DTYPE", "bfloat16"),
    "tensor_parallel_size": _env_int("TENSOR_PARALLEL_SIZE", 1),
    "max_model_len": _env_int("MAX_MODEL_LEN", 8192),
    "gpu_memory_utilization": _env_float("GPU_MEMORY_UTILIZATION", 0.95),
    "swap_space": _env_int("VLLM_SWAP_SPACE", 42),
    "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "true").lower() in {"1", "true", "yes"},
    "disable_custom_all_reduce": os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "true").lower()
    in {"1", "true", "yes"},
    "trust_remote_code": os.environ.get("TRUST_REMOTE_CODE", "true").lower() in {"1", "true", "yes"},
    "max_num_seqs": _env_int("VLLM_MAX_NUM_SEQS", 256),
}

RETRIEVER_INFER_CONFIG = {
    "temperature": _env_float("RETRIEVER_TEMPERATURE", 0.0),
    "max_tokens": _env_int("RETRIEVER_MAX_TOKENS", 512),
    "stop_token_ids": [151645],
}

SQL_INFER_CONFIG = {
    "temperature": _env_float("SQL_TEMPERATURE", 0.0),
    "max_tokens": _env_int("SQL_MAX_TOKENS", 2048),
    "stop_token_ids": [151645],
}

VALUE_LIMIT_NUM = _env_int("VALUE_LIMIT_NUM", 2)


def get_checkpoint_output_dir(output_base: str, project_name: str, checkpoint_step: int) -> str:
    return os.path.join(output_base, f"{project_name}_ckpt{checkpoint_step}")


def print_config() -> None:
    print("=" * 60)
    print("BIRD evaluation config")
    print("=" * 60)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
    print(f"SQL_MODEL_PATH: {SQL_MODEL_PATH}")
    print(f"BIRD_DEV_DATA: {BIRD_DEV_DATA}")
    print(f"BIRD_TABLES: {BIRD_TABLES}")
    print(f"BIRD_DB_PATH: {BIRD_DB_PATH}")
    print(f"BIRD_DB_CONTENT_INDEX_PATH: {BIRD_DB_CONTENT_INDEX_PATH}")
    print(f"RETRIEVER_PROMPT_PATH: {RETRIEVER_PROMPT_PATH}")
    print(f"DEFAULT_OUTPUT_DIR: {DEFAULT_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
