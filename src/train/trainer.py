from dataclasses import asdict
from pathlib import Path

import torch
from torch.func import functional_call
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import parse_config
from ..device import autocast, pick_amp_dtype
from ..evaluate import evaluate
from ..model.checkpoint import (
    load_trained,
    pack_checkpoint,
    write_checkpoint,
)
from .ema import EMA


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_pairs,
        criterion,
        optimizer,
        scheduler,
        save_dir,
        cfg,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_pairs = valid_pairs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.cfg = cfg
        self.ema = EMA(model, cfg.train.ema)
        self.trainable = [p for g in optimizer.param_groups for p in g["params"]]
        self.device = next(model.parameters()).device
        self.amp_dtype = pick_amp_dtype(self.device)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype == torch.float16
        )
        self.global_step = 0
        self.best_miou = float("-inf")
        self.metrics = {}
        self.batches = None

    def train(self):
        c = self.cfg.train
        self.writer = SummaryWriter(log_dir=self.save_dir / "logs")
        progress = tqdm(total=c.steps, initial=self.global_step, desc="Training")
        skipped = 0
        try:
            while self.global_step < c.steps:
                losses, updated = self.step()
                progress.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
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
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("lr", lr, self.global_step)
                self.metrics = {}
                if self.global_step % c.val_every == 0:
                    self.validate()
                if self.global_step % c.save_every == 0:
                    self.save()
            if self.global_step % c.val_every:
                self.validate()
                self.save()
            elif self.global_step % c.save_every:
                self.save()
        finally:
            progress.close()
            self.writer.close()

    def step(self):
        if not self.model.training:
            self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = self.cfg.train.accum
        share = self.teacher_share()
        totals = {}
        for _ in range(accum):
            batch = {
                key: value.to(self.device, non_blocking=self.device.type == "cuda")
                for key, value in self.next_batch().items()
            }
            with autocast(self.device, self.amp_dtype):
                inputs = batch["image"]
                if share > 0:
                    encoded = self.model.encode(inputs)
                    out = self.model(inputs, encoded)
                    teacher = self.predict_teacher(inputs, encoded)
                    parts, loss = self.criterion(
                        out, batch, teacher=teacher, teacher_weight=share
                    )
                else:
                    parts, loss = self.criterion(self.model(inputs), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at training step {self.global_step + 1}"
                )
            self.scaler.scale(loss / accum).backward()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value.item() / accum
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable, self.cfg.train.grad_clip
        )
        if not torch.isfinite(grad_norm) and not self.scaler.is_enabled():
            raise FloatingPointError(
                f"Non-finite gradient at training step {self.global_step + 1}"
            )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= old_scale
        if updated:
            self.ema.update(self.global_step)
            self.scheduler.step()
            self.global_step += 1
        totals["grad_norm"] = float(grad_norm)
        return totals, updated

    def next_batch(self):
        if self.batches is None:
            self.batches = iter(self.train_loader)
        try:
            return next(self.batches)
        except StopIteration:
            self.batches = iter(self.train_loader)
            return next(self.batches)

    def teacher_share(self):
        start = self.cfg.loss.teacher_start
        progress = (self.global_step - start) / start
        return self.cfg.loss.teacher * min(1.0, max(0.0, progress))

    @torch.no_grad()
    def predict_teacher(self, inputs, encoded):
        self.model.eval()
        out = functional_call(
            self.model, self.ema.params, (inputs, encoded), strict=False
        )
        self.model.train()
        return out.logits[-1].float().softmax(1)

    def validate(self):
        with self.ema.applied():
            self.metrics = evaluate(self.model, self.valid_pairs)
            for key, value in self.metrics.items():
                self.writer.add_scalar(f"valid/{key}", value, self.global_step)
            score = self.metrics.get("label_miou", self.metrics.get("miou"))
            if score > self.best_miou:
                self.best_miou = score
                state = pack_checkpoint(
                    self.model, self.cfg, self.global_step, self.metrics
                )
                write_checkpoint(state, self.save_dir / "weights/best_miou.pth")

    def save(self):
        args = (self.cfg, self.global_step, self.metrics)
        state = pack_checkpoint(self.model, *args)
        with self.ema.applied():
            averaged = pack_checkpoint(self.model, *args)
        write_checkpoint(averaged, self.save_dir / "weights/last.pth")
        state.update(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            ema=self.ema.state_dict(),
            best=self.best_miou,
        )
        write_checkpoint(state, self.save_dir / "weights/last.train.pth")

    def resume(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        saved = parse_config(state["config"])
        if asdict(saved) != asdict(self.cfg):
            raise ValueError("Resume checkpoint config differs from saved config.yaml")
        load_trained(self.model, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.ema.load_state_dict(state["ema"])
        self.global_step, self.best_miou = state["step"], state["best"]
