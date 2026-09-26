"""Audit one real training batch without an optimizer step."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build.data import build as build_data
from src.build.model import build as build_model
from src.config import load_config
from src.train.objective import TrainLoss


def audit(model, batch, criterion) -> dict:
    """Record and release each accumulated gradient; zero gradients are informational."""
    params = dict(model.named_parameters())
    records = {}
    hooks = []
    training = model.training
    model.zero_grad(set_to_none=True)

    def record(name, param):
        grad = param.grad
        if grad is not None:
            finite = bool(torch.isfinite(grad).all())
            nonzero = bool(torch.count_nonzero(grad))
            previous = records.get(name)
            records[name] = {
                "finite": finite and (previous is None or previous["finite"]),
                "nonzero": nonzero or (previous is not None and previous["nonzero"]),
            }
            param.grad = None

    try:
        model.train()
        for name, param in params.items():
            if param.requires_grad:
                hooks.append(param.register_post_accumulate_grad_hook(
                    lambda param, name=name: record(name, param)
                ))
        inputs = batch["image"]
        _, loss = criterion(model(inputs), batch)
        finite_loss = bool(torch.isfinite(loss))
        if finite_loss and loss.requires_grad:
            loss.backward()
        trainable = [name for name, param in params.items() if param.requires_grad]
        missing = [name for name in trainable if name not in records]
        nonfinite = [name for name, grad in records.items() if not grad["finite"]]
        zero = [name for name, grad in records.items() if not grad["nonzero"]]
        frozen_grad = [name for name, param in params.items()
                       if not param.requires_grad and param.grad is not None]
        return {
            "ok": bool(trainable) and finite_loss and not (missing or nonfinite or frozen_grad),
            "loss": float(loss.detach()) if finite_loss else None,
            "finite_loss": finite_loss,
            "trainable_tensors": len(trainable),
            "frozen_tensors": len(params) - len(trainable),
            "reached_tensors": len(records),
            "missing": missing,
            "nonfinite": nonfinite,
            "zero": zero,
            "frozen_grad": frozen_grad,
            "gradients": records,
        }
    finally:
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        model.train(training)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/train.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "run/checks/gradients.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.loader.batch = 1
    cfg.loader.num_workers = 0
    cfg.loader.persistent_workers = False
    cfg.loader.pin_memory = False
    weight = cfg.birefnet.get("weight")
    if weight and not Path(str(weight)).is_file():
        raise FileNotFoundError(f"Configured base checkpoint not found: {weight}")
    loader, _, _ = build_data(cfg)
    device = torch.device(args.device)
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    model = build_model(cfg).to(device)
    criterion = TrainLoss(
        lambda_cls=cfg.loss.lambda_cls,
        lambda_region=cfg.loss.lambda_region,
        lambda_boundary=cfg.loss.lambda_boundary,
        boundary_radius=cfg.loss.boundary_radius,
        lambda_aux=cfg.loss.lambda_aux,
        num_classes=cfg.birefnet.num_classes,
        ignore_index=cfg.data.get("ignore_index", 255),
    ).to(device)
    report = audit(model, batch, criterion)
    report.update(config=str(args.config), device=str(device), size=batch["image"].shape[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "gradients"}
    summary["output"] = str(args.output)
    print(json.dumps(summary, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
