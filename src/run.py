import random
import uuid
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import ROOT, load_config, read_config, save_config, resolve
from .data.io import pairs, read_sample
from .data.dataset import MaskDataset
from .data import split
from .device import pick_device
from .model.checkpoint import build
from .train.loss import TrainLoss
from .train.schedule import cosine_schedule
from .train.trainer import Trainer


def seed(value):
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)


def train(resume=None, config=None, runs=ROOT / "run", device=None):
    target = None
    if resume is not None:
        if config is not None:
            raise ValueError("--config cannot be combined with --resume")
        resume = resolve(resume).resolve()
        if resume.parent.name != "weights":
            raise ValueError("Resume must be inside run weights directory")
        target = resume.parent.parent
        cfg = read_config(target / "config.yaml")
    else:
        cfg = load_config(config)
    seed(cfg.train.seed)
    labels = resolve(cfg.data.label_dir) if cfg.data.label_dir else None
    available = pairs(
        resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir), labels
    )
    if len(available) < 2:
        raise ValueError("At least two pairs required")
    groups = (
        split.restore(available, split.load(target))
        if target
        else split.make(available, cfg.data.valid_ratio, cfg.train.seed)
    )
    if len(groups["train"]) < cfg.loader.batch_size:
        raise ValueError("loader.batch_size exceeds training pairs")
    unpainted = 0
    for pair in available:
        sample = read_sample(pair, cfg.model.classes)
        unpainted += len(sample) == 3 and bool((sample[2] == 255).all())
    if labels:
        print(f"images without label pixels: {unpainted}")
    device = device or pick_device()
    loader = DataLoader(
        MaskDataset(groups["train"], cfg),
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        drop_last=True,
        persistent_workers=cfg.loader.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    model = build(cfg).to(device)
    if device.type == "mps":
        model.bb.compile(dynamic=False)
    optimizer = torch.optim.AdamW(
        model.list_trainable(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = cosine_schedule(
        optimizer, cfg.train.steps, cfg.train.min_lr, cfg.train.warmup
    )
    if target is None:
        target = resolve(runs) / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
        )
        target.mkdir(parents=True)
        save_config(cfg, target / "config.yaml")
        split.save(groups, target)
    trainer = Trainer(
        model,
        loader,
        groups["valid"],
        TrainLoss(
            lambda_cls=cfg.loss.ce,
            lambda_region=cfg.loss.dice,
            lambda_boundary=cfg.loss.boundary,
            boundary_radius=cfg.loss.boundary_width,
            lambda_aux=cfg.loss.aux,
            num_classes=cfg.model.classes,
            label_blur=cfg.loss.label_blur,
            label_weight=cfg.loss.label_weight,
            mask_weight=cfg.loss.mask_weight,
            label_smooth=cfg.loss.label_smooth,
        ),
        optimizer,
        scheduler,
        target,
        cfg,
    )
    if resume is not None:
        trainer.load_resume(resume)
    stats = model.stats
    print(
        f"train={len(groups['train'])}, valid={len(groups['valid'])}, "
        f"trainable={stats['trainable'] / stats['total']:.2%}"
    )
    trainer.train(cfg.train.steps, cfg.train.val_every, cfg.train.save_every)
    print(f"Training finished: {target}")
    return target
