from pathlib import Path
from typing import Any

from ..model.checkpoint import pack_model, read_checkpoint
from ..prepare.spec import PreprocessSpec
from ..storage import atomic_torch_save


class CheckpointStore:
    def __init__(self, *, model: Any, optimizer: Any, scheduler: Any,
                 scaler: Any, teacher: Any, save_dir: str,
                 preprocess: PreprocessSpec, ignore_index: int = 255) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.teacher = teacher
        self.save_dir = save_dir
        self.preprocess = preprocess
        self.ignore_index = ignore_index

    def save(self, global_step: int, best_region: float) -> None:
        root = Path(self.save_dir) / "weights"
        root.mkdir(parents=True, exist_ok=True)
        payload = pack_model(self.model, self.preprocess, self.ignore_index)
        atomic_torch_save(payload, root / "last.pth")
        payload.update(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            teacher=self.teacher.state_dict() if self.teacher is not None else None,
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
        if (state["teacher"] is None) != (self.teacher is None):
            raise RuntimeError("Checkpoint teacher configuration differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if self.teacher is not None:
            self.teacher.load_state_dict(state["teacher"])
        return int(state["global_step"]), float(state["best_region"])
