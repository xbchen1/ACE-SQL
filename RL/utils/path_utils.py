import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = Path("models/sft_checkpoint")
DEFAULT_TRAIN_DB_ROOT = Path("external/train_databases")
DEFAULT_DEV_DB_ROOT = Path("external/dev_databases")
DEFAULT_LOOSE_DB_ROOTS = (Path("external/databases"),)


def _normalize(path: str | Path) -> str:
    return str(Path(path).expanduser())


def _dedupe_keep_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def get_model_path(path: Optional[str] = None) -> str:
    if path:
        return _normalize(path)
    return _normalize(os.environ.get("ACE_SQL_MODEL_PATH", DEFAULT_MODEL_PATH))


def get_train_db_root() -> str:
    return _normalize(os.environ.get("ACE_SQL_TRAIN_DB_ROOT", DEFAULT_TRAIN_DB_ROOT))


def get_dev_db_root() -> str:
    return _normalize(os.environ.get("ACE_SQL_DEV_DB_ROOT", DEFAULT_DEV_DB_ROOT))


def get_loose_db_roots() -> list[str]:
    raw = os.environ.get("ACE_SQL_LOOSE_DB_ROOTS")
    if not raw:
        return [_normalize(root) for root in DEFAULT_LOOSE_DB_ROOTS]
    parts = []
    for chunk in raw.replace(",", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(_normalize(chunk))
    return parts


def get_db_root(split: str) -> str:
    split = split.lower()
    if split == "train":
        return get_train_db_root()
    if split == "dev":
        return get_dev_db_root()
    raise ValueError(f"Unsupported split: {split}")


def build_db_path(db_id: str, split: str) -> str:
    return os.path.join(get_db_root(split), db_id, f"{db_id}.sqlite")


def infer_db_id(db_path: Optional[str], db_id: Optional[str] = None) -> str:
    if db_id:
        return db_id
    if not db_path:
        return ""
    basename = os.path.basename(db_path)
    if basename.endswith(".sqlite"):
        return basename[:-7]
    parent = os.path.basename(os.path.dirname(db_path))
    return parent


def resolve_db_path(db_path: Optional[str], db_id: Optional[str] = None) -> str:
    if db_path:
        normalized = _normalize(db_path)
        if os.path.exists(normalized):
            return normalized
    else:
        normalized = ""

    inferred_db_id = infer_db_id(normalized, db_id)
    path_lower = normalized.lower()

    hinted_roots = []
    if any(token in path_lower for token in ("bird_train", "train_databases")):
        hinted_roots.append(get_train_db_root())
        hinted_roots.extend(get_loose_db_roots())
    if any(token in path_lower for token in ("bird_dev", "dev_databases")):
        hinted_roots.append(get_dev_db_root())

    fallback_roots = [
        get_train_db_root(),
        *get_loose_db_roots(),
        get_dev_db_root(),
        _normalize(PROJECT_ROOT / "data" / "bird_train" / "train_databases"),
        _normalize(PROJECT_ROOT / "data" / "bird_dev" / "dev_databases"),
    ]

    candidate_roots = _dedupe_keep_order(hinted_roots + fallback_roots)
    if inferred_db_id:
        for root in candidate_roots:
            candidate = os.path.join(root, inferred_db_id, f"{inferred_db_id}.sqlite")
            if os.path.exists(candidate):
                return candidate

    if normalized:
        relative_candidates = _dedupe_keep_order([
            normalized,
            _normalize(PROJECT_ROOT / normalized),
        ])
        for candidate in relative_candidates:
            if os.path.exists(candidate):
                return candidate

    if inferred_db_id and candidate_roots:
        return os.path.join(candidate_roots[0], inferred_db_id, f"{inferred_db_id}.sqlite")
    return normalized
