import os
from collections.abc import Callable
from math import isfinite

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..prepare.spec import PreprocessSpec
from ..precision import autocast, autocast_dtype
from .checkpoint import CheckpointStore
from .schedule import CosineSchedule
from .ema import EMA
from .validate import Validator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineSchedule,
        save_dir: str,
        predictor: Callable[..., np.ndarray],
        max_grad_norm: float = 1.0,
        accum_steps: int = 1,
        ema_decay: float = 0.99,
        preprocess: PreprocessSpec | None = None,
        ignore_index: int = 255,
    ) -> None:
        if not isfinite(max_grad_norm) or not max_grad_norm > 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if (
            not isinstance(accum_steps, int)
            or isinstance(accum_steps, bool)
            or accum_steps < 1
        ):
            raise ValueError("accum_steps must be a positive integer")

        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = EMA(model, decay=ema_decay)
        self.save_dir = save_dir
        self.max_grad_norm = float(max_grad_norm)
        self.accum_steps = int(accum_steps)
        self.trainable_params = tuple(model.list_trainable())
        if not self.trainable_params:
            raise RuntimeError("Trainer requires trainable model parameters")

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.best_region = float("-inf")

        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = autocast_dtype(self.device)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        preprocess = preprocess or PreprocessSpec()
        self.validator = Validator(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=self.device,
            amp_dtype=self.amp_dtype,
            use_amp=self.use_amp,
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

    def step(self) -> tuple[dict[str, float], bool]:
        if not self.model.training:
            self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum: dict[str, float] = {}
        for _ in range(self.accum_steps):
            batch = self.move(self.next_batch())
            with autocast(
                self.device, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                inputs = batch["image"]
                out = self.model(inputs)
                loss_dict, loss = self.criterion(
                    out,
                    batch,
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
        if not torch.isfinite(torch.as_tensor(grad_norm)) and not (
            self.use_amp and self.amp_dtype == torch.float16
        ):
            raise FloatingPointError(
                f"Non-finite gradient at training step {self.global_step + 1}"
            )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= old_scale
        if updated:
            self.ema.update(self.model)
            self.scheduler.step()
            self.global_step += 1
        accum["grad_norm"] = float(grad_norm)
        return accum, updated

    def _evaluate(self) -> None:
        if self._writer is None:
            raise RuntimeError("Validation logging requires an active writer")

        valid_metrics = self.validator.validate()
        valid_metrics.update(self.validator.validate_deploy())
        for key, value in valid_metrics.items():
            self._writer.add_scalar(f"valid/{key}", value, self.global_step)
        score = valid_metrics["deploy_miou"]
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
