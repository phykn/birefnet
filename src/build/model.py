from typing import Any

import torch

from ..model import BiRefNet
from ..model.checkpoint import load_base


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
