import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..model import BiRefNet
from ..prepare.spec import PreprocessSpec
from ..predict.inference import predict_logits
from ..train.objective import TrainLoss
from ..train.schedule import CosineSchedule
from ..train.teacher import Teacher
from ..train.trainer import Trainer


def create_run_dir(root: str | os.PathLike[str] = "run") -> str:
    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = parent / f"{stamp}_{uuid.uuid4().hex[:8]}"
    target.mkdir(exist_ok=False)
    return os.fspath(target)


def build(
    cfg: Any,
    model: BiRefNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    save_dir: str | os.PathLike[str] | None = None,
) -> Trainer:
    criterion = TrainLoss(
        lambda_cls=cfg.loss.lambda_cls,
        lambda_region=cfg.loss.lambda_region,
        lambda_boundary=cfg.loss.lambda_boundary,
        region_loss=cfg.loss.region_loss,
        boundary_radius=cfg.loss.boundary_radius,
        lambda_aux=cfg.loss.lambda_aux,
        teacher_confidence=cfg.teacher.confidence,
        min_gt_weight=cfg.teacher.min_gt_weight,
        lambda_teacher=cfg.teacher.loss_weight,
        num_classes=cfg.birefnet.num_classes,
        ignore_index=cfg.data.get("ignore_index", 255),
    )

    backbone = []
    decoder = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (backbone if name.startswith("bb.") else decoder).append(param)
    param_groups = []
    for name, params, scale in [("decoder", decoder, 1.0),
                                ("backbone", backbone, float(cfg.train.backbone_lr_scale))]:
        if params:
            param_groups.append(dict(name=name, params=params,
                                     lr=float(cfg.train.max_lr) * scale,
                                     max_lr=float(cfg.train.max_lr) * scale,
                                     min_lr=float(cfg.train.min_lr) * scale,
                                     weight_decay=float(cfg.train.weight_decay)))
    if not param_groups:
        raise RuntimeError("No trainable parameters were selected")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineSchedule(
        optimizer=optimizer,
        first_cycle_steps=cfg.train.steps,
        max_lr=cfg.train.max_lr,
        min_lr=cfg.train.min_lr,
        warmup_steps=cfg.train.warmup_steps,
    )
    if save_dir is None:
        target = create_run_dir()
    else:
        target = os.fspath(save_dir)
        os.makedirs(target, exist_ok=True)
    teacher = (Teacher(model, decay=cfg.teacher.decay, start=cfg.teacher.start,
                       ramp=cfg.teacher.ramp) if cfg.teacher.enabled else None)
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher=teacher,
        save_dir=target,
        predictor=predict_logits,
        max_grad_norm=cfg.train.max_grad_norm,
        accum_steps=cfg.train.accum_steps,
        ignore_index=cfg.data.get("ignore_index", 255),
        preprocess=PreprocessSpec(
            size=int(cfg.data.size),
            mode=cfg.data.get("mode", "rgb"),
        ),
    )
