import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..model import BiRefNet
from ..prepare.spec import PreprocessSpec
from ..predict.inference import predict_logits
from ..train.schedule import CosineSchedule
from ..train.trainer import Trainer
from .loss import build as build_loss


def build(
    cfg: Any,
    model: BiRefNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    save_dir: str | os.PathLike[str],
) -> Trainer:
    criterion = build_loss(cfg)

    params = model.list_trainable()
    if not params:
        raise RuntimeError("No trainable parameters were selected")
    optimizer = torch.optim.AdamW(
        [{"name": "decoder", "params": params}],
        lr=float(cfg.train.max_lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = CosineSchedule(
        optimizer=optimizer,
        first_cycle_steps=cfg.train.steps,
        max_lr=cfg.train.max_lr,
        min_lr=cfg.train.min_lr,
        warmup_steps=cfg.train.warmup_steps,
    )
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=os.fspath(save_dir),
        predictor=predict_logits,
        max_grad_norm=cfg.train.max_grad_norm,
        accum_steps=cfg.train.accum_steps,
        ema_decay=cfg.train.get("ema_decay", 0.99),
        ignore_index=cfg.data.get("ignore_index", 255),
        preprocess=PreprocessSpec(
            is_sem=cfg.data.get("is_sem", False),
        ),
    )
