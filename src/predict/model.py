from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from ..build.model import build
from ..config import load_config
from ..model import BiRefNet
from ..model.checkpoint import read_checkpoint


def load_run_config(checkpoint: str | Path, config: str | Path | None = None) -> DictConfig:
    if config is None:
        saved = Path(checkpoint).parent.parent / "config.yaml"
        config = saved if saved.is_file() else None
    return load_config(config)


def load(cfg: Any, path: str | Path, device: torch.device) -> BiRefNet:
    checkpoint = read_checkpoint(path, cfg.birefnet.get("num_classes", 4))
    model = build(cfg, load_pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.loaded_meta = {
        key: value for key, value in checkpoint.items()
        if key not in {"model", "optimizer", "scheduler", "scaler", "ema", "teacher"}
    }
    return model.to(device).eval()
