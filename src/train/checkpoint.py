import os
import uuid
from pathlib import Path
from dataclasses import asdict
import torch
from ..config import parse
from ..model.checkpoint import pack_model, read_checkpoint, load_trained, trained_state


class CheckpointStore:
    def __init__(self, model, optimizer, scheduler, scaler, ema, save_dir, cfg):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ema = ema
        self.root = Path(save_dir) / "weights"
        self.cfg = cfg

    def save(self, step, best, metrics):
        payload = pack_model(self.model, self.cfg, step, metrics)
        ema = self.ema.state_dict()
        averaged = dict(payload)
        averaged["model"] = {**payload["model"], **trained_state(ema)}
        atomic_torch_save(averaged, self.root / "last.pth")
        payload.update(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            ema=ema,
            best=best,
        )
        atomic_torch_save(payload, self.root / "last.train.pth")

    def save_best(self, metrics, step):
        atomic_torch_save(
            pack_model(self.model, self.cfg, step, metrics), self.root / "best_miou.pth"
        )

    def load_resume(self, path):
        state = read_checkpoint(path)
        if asdict(parse(state["config"], ignore_unknown=True)) != asdict(self.cfg):
            raise ValueError("Resume checkpoint config differs from saved config.yaml")
        for key in ("optimizer", "scheduler", "scaler", "ema", "best", "step"):
            if key not in state:
                raise ValueError(f"Resume checkpoint missing {key}")
        load_trained(self.model, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.ema.load_state_dict(state["ema"])
        return state["step"], state["best"]


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
