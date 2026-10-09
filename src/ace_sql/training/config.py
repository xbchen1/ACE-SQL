"""Validated experiment configuration shared by preparation and training."""
from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path
import tomllib


from ace_sql.paths import project_root


@dataclass(frozen=True)
class TrainConfig:
    model: str = 'models/sft_checkpoint'
    reference: str = 'models/sft_torch_dist'
    slime: str = 'external/slime'
    data: str = '.runs/data'
    output: str = '.runs/train'
    resume: str = ''
    model_size: str = '8B'
    seed: int = 42
    epochs: int = 5
    questions: int = 16
    gpus: int = 8
    learning_rate: float = 1e-6
    warmup_fraction: float = 0.05
    kl_coefficient: float = 0.001
    clip: float = 0.2
    prompt_tokens: int = 4096
    response_tokens: int = 2048
    temperature: float = 1.0
    save_interval: int = 80
    eval_interval: int = 40
    target_mode: str = 'online'
    pool_gamma: float = 0.5
    ramp_ratio: float = 0.25
    length_penalty: bool = True
    concurrency: int = 64
    expected_cuda: str = '12.9'

    def validate(self):
        for name in ('model', 'reference', 'slime', 'data', 'output', 'resume', 'expected_cuda'):
            if not isinstance(getattr(self, name), str) or (name != 'resume' and not getattr(self, name)):
                raise ValueError(f'{name} must be a nonempty string (resume may be empty)')
        for name in ('seed', 'epochs', 'questions', 'gpus', 'prompt_tokens', 'response_tokens',
                     'save_interval', 'eval_interval', 'concurrency'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be a valid positive integer (seed may be zero)')
        if self.gpus < 2 or self.gpus % 2 or self.questions * 16 % (self.gpus // 2):
            raise ValueError('Even GPU count and divisible data-parallel trajectory batch are required')
        for name in ('warmup_fraction', 'pool_gamma', 'ramp_ratio', 'clip'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        for name in ('learning_rate', 'kl_coefficient', 'temperature'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate == 0:
            raise ValueError('learning_rate must be positive')
        if type(self.length_penalty) is not bool:
            raise ValueError('length_penalty must be boolean')
        if self.model_size not in {'4B', '8B'} or self.target_mode not in {'online', 'frozen', 'gold', 'dense'}:
            raise ValueError('Unsupported model_size or target_mode')
        # The paper's reward and response cap are one contract.
        if (self.prompt_tokens, self.response_tokens) != (4096, 2048):
            raise ValueError('This paper recipe requires prompt_tokens=4096 and response_tokens=2048')
        return self

    def resolved(self, root=None):
        root = Path(root or project_root())
        values = asdict(self)
        for key in ('model', 'reference', 'slime', 'data', 'output', 'resume'):
            if values[key]:
                path = Path(values[key]).expanduser()
                values[key] = str((root / path).resolve())
        return TrainConfig(**values).validate()


SECTIONS = {
    'paths': {'model', 'reference', 'slime', 'data', 'output', 'resume'},
    'training': {'model_size', 'seed', 'epochs', 'questions', 'gpus', 'learning_rate', 'warmup_fraction',
                 'kl_coefficient', 'clip', 'prompt_tokens', 'response_tokens', 'temperature',
                 'save_interval', 'eval_interval', 'concurrency', 'expected_cuda'},
    'method': {'target_mode', 'pool_gamma', 'ramp_ratio', 'length_penalty'},
}
ENV = {
    'MODEL_PATH': 'model', 'REFERENCE_PATH': 'reference', 'SLIME_DIR': 'slime',
    'SLIME_DATA': 'data', 'SLIME_OUTPUT': 'output', 'SLIME_RESUME': 'resume',
    'SEED': 'seed', 'TOTAL_EPOCHS': 'epochs', 'TRAIN_BATCH_SIZE': 'questions', 'NUM_GPUS': 'gpus',
    'POOL_GAMMA': 'pool_gamma', 'RAMP_RATIO': 'ramp_ratio', 'TARGET_MODE': 'target_mode',
    'LENGTH_PENALTY': 'length_penalty', 'EXPECTED_CUDA': 'expected_cuda',
}


def load_config(path=None, *, overrides=None, environ=None, root=None):
    values = {}
    if path:
        document = tomllib.loads(Path(path).read_text())
        for section, settings in document.items():
            if section not in SECTIONS or not isinstance(settings, dict):
                raise ValueError(f'Unknown configuration section: {section}')
            unknown = set(settings) - SECTIONS[section]
            if unknown:
                raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')
            values.update(settings)
    env = os.environ if environ is None else environ
    types = {field.name: field.type for field in fields(TrainConfig)}
    for suffix, key in ENV.items():
        name = 'ACE_SQL_' + suffix
        if name not in env:
            continue
        value = env[name]
        if types[key] is bool:
            if value.lower() not in {'true', 'false', '1', '0'}:
                raise ValueError(f'{name} must be true or false')
            value = value.lower() in {'true', '1'}
        else:
            value = types[key](value)
        values[key] = value
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return TrainConfig(**values).validate().resolved(root)
