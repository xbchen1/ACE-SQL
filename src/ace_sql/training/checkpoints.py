"""The evaluation boundary is a complete Hugging Face export from slime."""
import json
from pathlib import Path


def require_hf_checkpoint(path):
    root = Path(path).expanduser().resolve()
    config = root / 'config.json'
    if not config.is_file():
        raise ValueError(f'Expected a Hugging Face export with config.json: {root}')
    json.loads(config.read_text())
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = root / name
        if index.is_file():
            mapping = json.loads(index.read_text()).get('weight_map', {})
            if not mapping:
                raise ValueError(f'Empty model weight index: {index}')
            weights = [root / file for file in set(mapping.values())]
            if any(not p.resolve().is_relative_to(root) for p in weights):
                raise ValueError('Weight index points outside the checkpoint directory')
            break
    else:
        weights = [p for p in (root / 'model.safetensors', root / 'pytorch_model.bin') if p.is_file()]
    if not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError(f'Incomplete HF export; model weights or index shards missing: {root}')
    return root
