from typing import Any

import torch

from ..model import BiRefNet
from ..model.checkpoint import load_base, read_checkpoint


def build(cfg: Any, load_pretrained: bool = True) -> BiRefNet:
    model = BiRefNet(
        **({"channels": list(cfg.birefnet.channels)} if "channels" in cfg.birefnet else {}),
        num_classes=cfg.birefnet.get("num_classes", 4),
    )
    path = str(cfg.birefnet.weight) if cfg.birefnet.get("weight") else None
    if load_pretrained and path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        load_base(model, state)
        print(f"[LOAD] {path}")
    return model


def build_predictor(cfg: Any, path: str, device: torch.device) -> BiRefNet:
    num_classes = cfg.birefnet.get("num_classes", 4)
    checkpoint = read_checkpoint(path, num_classes)
    model = build(cfg, load_pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.loaded_meta = {
        key: value for key, value in checkpoint.items()
        if key not in {"model", "optimizer", "scheduler", "scaler", "ema", "teacher"}
    }
    return model.to(device).eval()
