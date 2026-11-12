import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule(optimizer, steps, min_lr, warmup=0):
    def factor(peak):
        floor = min_lr / peak

        def at(step):
            if step < warmup:
                return floor + (1 - floor) * step / warmup
            progress = (step - warmup) / (steps - warmup)
            return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

        return at

    return LambdaLR(optimizer, [factor(g["lr"]) for g in optimizer.param_groups])
