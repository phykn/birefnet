import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule(
    optimizer: torch.optim.Optimizer, steps: int, min_lr: float, warmup: int = 0
) -> LambdaLR:
    """Ramp each group's lr up from min_lr over `warmup` steps, then decay it
    back to min_lr along a cosine that ends at `steps`."""
    if type(steps) is not int or steps < 1:
        raise ValueError("train.steps must be a positive integer")
    if type(warmup) is not int or not 0 <= warmup < steps:
        raise ValueError("train.warmup must be an integer in [0, train.steps)")
    peaks = [group["lr"] for group in optimizer.param_groups]
    for peak in peaks:
        if not (math.isfinite(min_lr) and math.isfinite(peak) and 0 < peak and 0 <= min_lr <= peak):
            raise ValueError("learning rates must be finite with 0 <= train.min_lr <= train.lr")

    def factor(peak: float):
        floor = min_lr / peak

        def at(step: int) -> float:
            if step < warmup:
                return floor + (1 - floor) * step / warmup
            progress = min(1.0, (step - warmup) / (steps - warmup))
            return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

        return at

    return LambdaLR(optimizer, [factor(peak) for peak in peaks])
