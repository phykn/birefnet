import uuid
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from .config import ROOT, load_config, save_config, resolve
from .data.io import pair_files
from .data.dataset import MaskDataset
from .data import split
from .device import pick_device
from .model.checkpoint import SEGMENTATION_HEADS, build_model
from .train.loss import TrainLoss
from .train.schedule import cosine_schedule
from .train.trainer import Trainer


def train(resume=None, config=None, runs=ROOT / "run", device=None):
    run_dir = None
    if resume is not None:
        resume = resolve(resume).resolve()
        if resume.parent.name != "weights":
            raise ValueError("Resume must be inside run weights directory")
        run_dir = resume.parent.parent
        cfg = load_config(run_dir / "config.yaml")
    else:
        cfg = load_config(config)
    labels = resolve(cfg.data.label_dir) if cfg.data.label_dir else None
    available = pair_files(
        resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir), labels
    )
    if len(available) < 2:
        raise ValueError("At least two pairs required")
    splits = (
        split.read_split_csv(available, run_dir)
        if run_dir
        else split.split_pairs(available, cfg.data.valid_ratio)
    )
    if len(splits["train"]) < cfg.loader.batch_size:
        raise ValueError("loader.batch_size exceeds training pairs")
    device = device or pick_device()
    loader = DataLoader(
        MaskDataset(splits["train"], cfg),
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        drop_last=True,
        persistent_workers=cfg.loader.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg).to(device)
    if device.type == "mps":
        model.bb.compile(dynamic=False)
    heads, rest = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_head = name.rsplit(".", 1)[0] in SEGMENTATION_HEADS
            (heads if is_head else rest).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": rest},
            {"params": heads, "lr": cfg.train.lr * cfg.train.head_lr_scale},
        ],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = cosine_schedule(
        optimizer, cfg.train.steps, cfg.train.min_lr, cfg.train.warmup
    )
    if run_dir is None:
        run_dir = resolve(runs) / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        run_dir.mkdir(parents=True)
        save_config(cfg, run_dir / "config.yaml")
        split.write_split_csv(splits, run_dir)
    trainer = Trainer(
        model,
        loader,
        splits["valid"],
        TrainLoss(cfg),
        optimizer,
        scheduler,
        run_dir,
        cfg,
    )
    if resume is not None:
        trainer.resume(resume)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainer.trainable)
    print(
        f"train={len(splits['train'])}, valid={len(splits['valid'])}, "
        f"trainable={trainable / total:.2%}"
    )
    trainer.train()
    print(f"Training finished: {run_dir}")
    return run_dir
