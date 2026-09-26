"""Inspect real loader batches and save augmentation/mask previews."""
import argparse
import json
import random
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._preview import colorize, image_from_tensor, overlay, save_preview
from src.build.data import build as build_data
from src.config import load_config


def inspect(loader, num_classes: int, ignore_index: int, batches: int, output: Path) -> dict:
    if batches < 1:
        raise ValueError("batches must be positive")
    output.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(num_classes, dtype=np.int64)
    records = []
    for index, batch in enumerate(islice(loader, batches), 1):
        masks = batch["mask"].cpu().numpy()
        valid = (batch["valid"][:, 0].cpu().numpy() > 0.5) & (masks != ignore_index)
        if ((masks[valid] < 0) | (masks[valid] >= num_classes)).any():
            raise ValueError("Batch contains invalid class IDs")
        counts += np.bincount(masks[valid], minlength=num_classes)
        paths = []
        for item in range(len(masks)):
            image = image_from_tensor(batch["image"][item])
            colors = colorize(masks[item], num_classes, valid[item])
            panels = [("Input", image)]
            panels.extend([("Mask (gray = ignored)", colors), ("Overlay", overlay(image, colors)),
                           ("Valid pixels", valid[item].astype(np.uint8) * 255)])
            if "cut" in batch:
                panels.append(("Crop boundaries", (batch["cut"][item, 0].numpy() * 255).astype(np.uint8)))
            path = output / f"batch_{index:02d}_sample_{item + 1:02d}.png"
            save_preview(panels, path)
            paths.append(str(path))
        records.append({"batch": index, "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in batch.items()
        }, "ignored_pixels": int((~valid).sum()), "previews": paths})
    if not records:
        raise RuntimeError("Loader is empty")
    report = {"batches": records, "class_pixels": counts.tolist()}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("run/checks/data"))
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("batches must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    cfg.loader.num_workers = 0
    cfg.loader.persistent_workers = False
    cfg.loader.pin_memory = False
    train, valid, _ = build_data(cfg)
    loader = train if args.split == "train" else valid
    report = inspect(loader, int(cfg.birefnet.num_classes), int(cfg.data.get("ignore_index", 255)),
                     args.batches, args.output)
    print(f"{args.split}: {len(loader.dataset)} images, {len(report['batches'])} batches inspected")
    print(f"Class pixel counts: {report['class_pixels']}")
    print(f"Previews and summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
