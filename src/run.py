import uuid
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .build.data import build as build_data
from .build.model import build as build_model
from .build.trainer import build as build_trainer
from .config import ROOT, load_config, read_config
from .data.split import load as load_splits
from .data.split import save as save_splits


def load_run(
    resume: str | Path | None = None,
    config: str | Path | None = None,
) -> tuple[DictConfig, Path | None, Path | None]:
    if resume is None:
        return load_config(config or ROOT / "config/train.yaml"), None, None
    if config is not None:
        raise ValueError("--config cannot override a resumed run's saved config")

    checkpoint = Path(resume).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "weights":
        raise ValueError("Resume checkpoint must be inside a run weights directory")
    run_dir = checkpoint.parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return read_config(config_path), checkpoint, run_dir


def create_run_dir(root: str | Path = "run") -> Path:
    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = parent / f"{stamp}_{uuid.uuid4().hex[:8]}"
    target.mkdir(exist_ok=False)
    return target


def train(
    resume: str | Path | None = None,
    config: str | Path | None = None,
) -> None:
    cfg, checkpoint, run_dir = load_run(resume, config)
    saved_splits = load_splits(run_dir) if run_dir is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, load_pretrained=checkpoint is None).to(device)
    train_loader, valid_loader, splits = build_data(cfg, saved_splits)
    target = run_dir if run_dir is not None else create_run_dir()
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=target,
    )
    if checkpoint is not None:
        trainer.load_resume(str(checkpoint))

    n_train = len(splits["train_image"])
    n_valid = len(splits["valid_image"])
    total, trainable = model.stats["total"], model.stats["trainable"]
    print(f"\n[Dataset] train={n_train}, valid={n_valid}")
    print(
        f"[BiRefNet] total={total:,}  trainable={trainable:,}  "
        f"ratio={trainable / total:.2%}"
    )

    if run_dir is None:
        OmegaConf.save(cfg, target / "config.yaml")
        save_splits(splits, target)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )
    print(f"\nTraining finished. Weights saved to: {target}")
