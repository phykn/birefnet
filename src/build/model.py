from typing import Any

import torch
from omegaconf import OmegaConf

from ..model import BiRefNet
from ..model.checkpoint import load_base
from ..model.depth import Depth


def build(cfg: Any, load_pretrained: bool = True) -> BiRefNet:
    depth = cfg.model.get("depth", False)
    if not isinstance(depth, bool):
        raise ValueError("model.depth must be a bool")
    grad_checkpoint = OmegaConf.select(cfg, "train.grad_checkpoint", default=False)
    if not isinstance(grad_checkpoint, bool):
        raise ValueError("train.grad_checkpoint must be a bool")
    model = BiRefNet(
        **({"channels": list(cfg.model.channels)} if "channels" in cfg.model else {}),
        num_classes=cfg.model.get("classes", 4),
        depth=Depth.load(cfg.model.depth_weight) if depth else None,
        grad_checkpoint=grad_checkpoint,
    )
    path = str(cfg.model.weight) if cfg.model.get("weight") else None
    if load_pretrained and path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        load_base(model, state)
        print(f"[LOAD] {path}")
    return model
