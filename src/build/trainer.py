import os
from math import isfinite
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..model import BiRefNet
from ..prepare.spec import PreprocessSpec
from ..predict.inference import predict_logits
from ..train.schedule import cosine_schedule
from ..train.trainer import Trainer
from .loss import build as build_loss


def build(
    cfg: Any,
    model: BiRefNet,
    train_loader: DataLoader,
    valid_pairs: list[tuple[str, str]],
    save_dir: str | os.PathLike[str],
) -> Trainer:
    criterion = build_loss(cfg)

    params = model.list_trainable()
    if not params:
        raise RuntimeError("No trainable parameters were selected")
    weight_decay = float(cfg.train.weight_decay)
    if not isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("train.weight_decay must be finite and non-negative")
    optimizer = torch.optim.AdamW(
        [{"name": "decoder", "params": params}],
        lr=float(cfg.train.lr),
        weight_decay=weight_decay,
    )
    scheduler = cosine_schedule(
        optimizer, steps=cfg.train.steps, min_lr=float(cfg.train.min_lr), warmup=cfg.train.warmup
    )
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_pairs=valid_pairs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=os.fspath(save_dir),
        predictor=predict_logits,
        max_grad_norm=cfg.train.grad_clip,
        accum_steps=cfg.train.accum,
        ema_decay=cfg.train.get("ema", 0.99),
        ignore_index=cfg.data.get("ignore_index", 255),
        teacher_weight=float(cfg.loss.get("teacher", 0.0)),
        teacher_start=cfg.loss.get("teacher_start", 1),
        preprocess=PreprocessSpec(
            is_sem=cfg.data.get("sem", False),
        ),
    )
