import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule(
    optimizer: torch.optim.Optimizer, steps: int, min_lr: float, warmup: int = 0
) -> LambdaLR:
    def factor(peak: float):
        floor = min_lr / peak

        def at(step: int) -> float:
            if step < warmup:
                return floor + (1 - floor) * step / warmup
            progress = (step - warmup) / (steps - warmup)
            return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

        return at

    return LambdaLR(optimizer, [factor(g["lr"]) for g in optimizer.param_groups])
