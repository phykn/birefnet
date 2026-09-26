import os
import uuid
from pathlib import Path
from typing import Any

import torch

from ..model.checkpoint import pack_model, read_checkpoint
from ..prepare.spec import PreprocessSpec


class CheckpointStore:
    def __init__(self, *, model: Any, optimizer: Any, scheduler: Any,
                 scaler: Any, ema: Any, save_dir: str,
                 preprocess: PreprocessSpec, ignore_index: int = 255) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ema = ema
        self.save_dir = save_dir
        self.preprocess = preprocess
        self.ignore_index = ignore_index

    def save(self, global_step: int, best_region: float) -> None:
        root = Path(self.save_dir) / "weights"
        root.mkdir(parents=True, exist_ok=True)
        payload = pack_model(self.model, self.preprocess, self.ignore_index)
        ema_state = self.ema.state_dict()
        atomic_torch_save(payload, root / "last.pth")
        averaged = dict(payload)
        averaged["model"] = dict(payload["model"])
        averaged["model"].update(ema_state)
        atomic_torch_save(averaged, root / "last_ema.pth")
        payload.update(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            ema=ema_state,
            ema_decay=self.ema.decay,
            global_step=global_step, best_region=best_region,
        )
        atomic_torch_save(payload, root / "last.train.pth")

    def save_best(self, name: str, metrics: dict[str, float], global_step: int) -> None:
        root = Path(self.save_dir) / "weights"
        root.mkdir(parents=True, exist_ok=True)
        payload = pack_model(self.model, self.preprocess, self.ignore_index)
        payload["selection"] = {"name": name, "global_step": global_step,
                                "metrics": metrics}
        atomic_torch_save(payload, root / f"best_{name}.pth")

    def load_resume(self, path: str) -> tuple[int, float]:
        state = read_checkpoint(path, self.model.num_classes)
        if "optimizer" not in state:
            raise RuntimeError("Unsupported training checkpoint format; LoRA overlays cannot be resumed")
        if state.get("ignore_index", 255) != self.ignore_index:
            raise RuntimeError("Checkpoint ignore_index differs from configuration")
        if PreprocessSpec.from_meta(state) != self.preprocess:
            raise RuntimeError("Checkpoint preprocess differs from configuration")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if "ema" in state:
            self.ema.load_state_dict(state["ema"])
            self.ema.decay = float(state["ema_decay"])
        else:
            self.ema.load_state_dict({name: state["model"][name] for name in self.ema.params})
        return int(state["global_step"]), float(state["best_region"])


def atomic_torch_save(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
