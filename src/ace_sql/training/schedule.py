"""Complementary retriever/generator weights from the paper."""
import math

def role_weights(step: int, total_steps: int, ramp_ratio: float = 0.25) -> tuple[float, float]:
    """Complementary paper weights; slime's first rollout/update is step zero."""
    if type(step) is not int or step < 0 or type(total_steps) is not int or total_steps <= 0:
        raise ValueError("step must be a nonnegative integer and total_steps a positive integer")
    if not math.isfinite(ramp_ratio) or not 0 <= ramp_ratio <= 1:
        raise ValueError("ramp_ratio must be finite and in [0, 1]")
    lam = 1.0 if ramp_ratio == 0 else min(1.0, step / (total_steps * ramp_ratio))
    return 1.0 - lam / 2.0, lam / 2.0
