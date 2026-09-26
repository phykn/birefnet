import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineSchedule(_LRScheduler):
    """Cosine annealing schedule with linear warmup and optional warm restarts.

    Args:
        optimizer: Wrapped optimizer.
        first_cycle_steps: Step size of the first cycle.
        cycle_mult: Cycle length multiplier applied at each restart.
        max_lr: Max learning rate of the first cycle.
        min_lr: Minimum learning rate.
        warmup_steps: Linear warmup step size (must be < first_cycle_steps).
        gamma: Per-cycle decay factor applied to max_lr.
        last_epoch: Index of the last epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float | list[float] = 0.1,
        min_lr: float | list[float] = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        if isinstance(first_cycle_steps, bool) or first_cycle_steps <= 0:
            raise ValueError("first_cycle_steps must be positive")
        if isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if warmup_steps >= first_cycle_steps:
            raise ValueError("warmup_steps must be less than first_cycle_steps")
        if cycle_mult <= 0:
            raise ValueError("cycle_mult must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        group_count = len(optimizer.param_groups)
        self.base_max_lrs = self._group_values(optimizer, max_lr, "max_lr", group_count)
        self.max_lrs = self.base_max_lrs.copy()
        self.min_lrs = self._group_values(optimizer, min_lr, "min_lr", group_count)
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)

        self.init_lr()

    @staticmethod
    def _group_values(
        optimizer: torch.optim.Optimizer,
        value: float | list[float],
        group_key: str,
        group_count: int,
    ) -> list[float]:
        if isinstance(value, list):
            if len(value) != group_count:
                raise ValueError(
                    f"{group_key} must have {group_count} values, got {len(value)}"
                )
            defaults = [float(item) for item in value]
        else:
            defaults = [float(value)] * group_count
        return [
            float(group.get(group_key, default))
            for group, default in zip(optimizer.param_groups, defaults)
        ]

    def init_lr(self):
        self.base_lrs = self.min_lrs.copy()
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group["lr"] = lr

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                (max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
            ]
        progress = (self.step_in_cycle - self.warmup_steps) / (
            self.cur_cycle_steps - self.warmup_steps
        )
        return [
            base_lr + (max_lr - base_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
        ]

    def step(self, epoch=None):
        if epoch is not None:
            raise TypeError(
                "CosineSchedule does not support epoch-resume; "
                "call step() without arguments."
            )
        self.last_epoch += 1
        self.step_in_cycle += 1
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle -= self.cur_cycle_steps
            self.cur_cycle_steps = (
                int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                + self.warmup_steps
            )

        self.max_lrs = [
            base_max_lr * (self.gamma**self.cycle) for base_max_lr in self.base_max_lrs
        ]
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group["lr"] = lr
