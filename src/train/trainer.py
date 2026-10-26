import os
from collections.abc import Callable
from math import isfinite

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..prepare.spec import PreprocessSpec
from ..precision import amp_enabled, autocast, autocast_dtype
from .checkpoint import CheckpointStore
from .ema import EMA
from .validate import Validator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_pairs: list[tuple[str, str]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        save_dir: str,
        predictor: Callable[..., np.ndarray],
        max_grad_norm: float = 1.0,
        accum_steps: int = 1,
        ema_decay: float = 0.99,
        preprocess: PreprocessSpec | None = None,
        ignore_index: int = 255,
        teacher_weight: float = 0.0,
        teacher_start: int = 1,
    ) -> None:
        if not isfinite(max_grad_norm) or not max_grad_norm > 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if (
            not isinstance(accum_steps, int)
            or isinstance(accum_steps, bool)
            or accum_steps < 1
        ):
            raise ValueError("accum_steps must be a positive integer")
        if not 0.0 <= teacher_weight < 1.0:
            raise ValueError("teacher_weight must be in [0, 1)")
        if type(teacher_start) is not int or teacher_start < 1:
            raise ValueError("teacher_start must be a positive integer")

        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = EMA(model, decay=ema_decay)
        self.save_dir = save_dir
        self.max_grad_norm = float(max_grad_norm)
        self.accum_steps = int(accum_steps)
        self.teacher_weight = float(teacher_weight)
        self.teacher_start = teacher_start
        self.trainable_params = tuple(model.list_trainable())
        if not self.trainable_params:
            raise RuntimeError("Trainer requires trainable model parameters")

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.best_region = float("-inf")

        self.use_amp = amp_enabled(self.device)
        self.amp_dtype = autocast_dtype(self.device)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        preprocess = preprocess or PreprocessSpec()
        self.validator = Validator(
            model=model,
            pairs=valid_pairs,
            preprocess=preprocess,
            predictor=predictor,
            ignore_index=ignore_index,
        )
        self.checkpoints = CheckpointStore(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=self.scaler,
            ema=self.ema,
            save_dir=save_dir,
            preprocess=preprocess,
            ignore_index=ignore_index,
        )
        self._writer: SummaryWriter | None = None
        self._train_iter = None

    def move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    def save(self) -> None:
        self.checkpoints.save(self.global_step, self.best_region)

    def load_resume(self, path: str) -> None:
        self.global_step, self.best_region = self.checkpoints.load_resume(path)
        self._train_iter = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def teacher_share(self) -> float:
        """How much the EMA teacher's prediction counts in the targets.

        Zero for the first `teacher_start` steps, while the average is still poor,
        then a linear rise to `teacher_weight` over the same number of steps.
        """
        progress = (self.global_step - self.teacher_start) / self.teacher_start
        return self.teacher_weight * min(1.0, max(0.0, progress))

    @torch.no_grad()
    def teach(self, inputs: torch.Tensor, prepared: tuple) -> torch.Tensor:
        """Class probabilities from the EMA weights, reusing the frozen features."""
        self.model.eval()
        try:
            out = functional_call(self.model, self.ema.params, (inputs, prepared), strict=False)
        finally:
            self.model.train()
        return out.logits[-1].float().softmax(1)

    def step(self) -> tuple[dict[str, float], bool]:
        if not self.model.training:
            self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum: dict[str, float] = {}
        share = self.teacher_share()
        for _ in range(self.accum_steps):
            batch = self.move(self.next_batch())
            with autocast(
                self.device, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                inputs = batch["image"]
                prepared = self.model.prepare(inputs)
                out = self.model(inputs, prepared)
                teacher = self.teach(inputs, prepared) if share > 0 else None
                loss_dict, loss = self.criterion(
                    out,
                    batch,
                    teacher=teacher,
                    teacher_weight=share,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at training step {self.global_step + 1}"
                )
            self.scaler.scale(loss / self.accum_steps).backward()
            for key, value in loss_dict.items():
                accum[key] = accum.get(key, 0.0) + value.item() / self.accum_steps

        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.trainable_params,
            self.max_grad_norm,
        )
        # With float16 the scaler skips such a step and lowers its scale instead.
        if not torch.isfinite(grad_norm) and not self.scaler.is_enabled():
            raise FloatingPointError(
                f"Non-finite gradient at training step {self.global_step + 1}"
            )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= old_scale
        if updated:
            self.ema.update()
            self.scheduler.step()
            self.global_step += 1
        accum["grad_norm"] = float(grad_norm)
        return accum, updated

    def _evaluate(self) -> None:
        if self._writer is None:
            raise RuntimeError("Validation logging requires an active writer")

        # Score the EMA weights, the same ones the teacher uses and best_miou.pth keeps.
        with self.ema.applied():
            valid_metrics = self.validator.validate()
            for key, value in valid_metrics.items():
                self._writer.add_scalar(f"valid/{key}", value, self.global_step)
            score = valid_metrics["miou"]
            if score > self.best_region:
                self.best_region = score
                self.checkpoints.save_best("miou", valid_metrics, self.global_step)

    def train(
        self,
        steps: int,
        val_freq: int = 500,
        save_freq: int = 1000,
    ) -> None:
        for name, value in (("steps", steps), ("val_freq", val_freq), ("save_freq", save_freq)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))
        self._writer = writer
        progress = tqdm(
            total=steps,
            initial=self.global_step,
            desc="Training",
        )
        skipped = 0
        last_validated: int | None = None
        last_saved: int | None = None

        try:
            while self.global_step < steps:
                losses, updated = self.step()
                progress.set_postfix(
                    {key: f"{value:.4f}" for key, value in losses.items()}
                )
                if not updated:
                    skipped += 1
                    if skipped >= 16:
                        raise FloatingPointError(
                            "Optimizer step was skipped 16 consecutive times"
                        )
                    continue
                skipped = 0
                progress.update(1)
                for key, value in losses.items():
                    writer.add_scalar(f"train/{key}", value, self.global_step)
                for index, group in enumerate(self.optimizer.param_groups):
                    name = group.get("name", str(index))
                    writer.add_scalar(f"lr/{name}", group["lr"], self.global_step)

                if self.global_step % val_freq == 0:
                    self._evaluate()
                    last_validated = self.global_step

                if self.global_step % save_freq == 0:
                    self.save()
                    last_saved = self.global_step

            if last_validated != self.global_step:
                self._evaluate()
                last_saved = None
            if last_saved != self.global_step:
                self.save()
        finally:
            progress.close()
            writer.flush()
            writer.close()
            self._writer = None
