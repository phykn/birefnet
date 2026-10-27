"""Predict one image and show the class map, overlay, and optional ground truth."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, overlay
from src.data.image import read_image, read_mask
from src.device import pick_device
from src.predict.inference import predict_logits, probabilities
from src.predict.model import load as load_model, load_run_config
from src.prepare.spec import PreprocessSpec


def run(model, image_path: Path, tiles=(1,), mask_path=None, class_id=None,
        ignore_index=None) -> tuple[np.ndarray, list]:
    if class_id is not None and not 0 <= class_id < model.num_classes:
        raise ValueError("class_id must be in [0, num_classes)")
    meta = getattr(model, "loaded_meta", None)
    spec = PreprocessSpec.from_meta(meta)
    if ignore_index is None:
        ignore_index = (meta or {}).get("ignore_index", 255)
    image = read_image(str(image_path))
    panels = [image]
    if mask_path is not None:
        target = read_mask(str(mask_path))
        if target.shape != image.shape[:2]:
            raise ValueError("Image and mask dimensions differ")
        panels.append(colorize(target, model.num_classes, target != ignore_index))
    logits = predict_logits(model, image, size=spec.size, is_sem=spec.is_sem, tiles=tiles)
    labels = logits.argmax(axis=0).astype(np.uint8)
    colors = colorize(labels, model.num_classes)
    panels.extend([colors, overlay(image, colors)])
    if class_id is not None:
        probs = probabilities(logits)
        panels.append(np.rint(probs[class_id] * 255).astype(np.uint8))
    return labels, panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="Optional ground-truth mask for comparison")
    parser.add_argument("--config", type=Path, help="Defaults to the checkpoint's run config, then model defaults")
    parser.add_argument("--device", help="Defaults to CUDA, then Apple GPU, then CPU, whichever is present")
    parser.add_argument("--tiles", type=int, nargs="+", default=[1])
    parser.add_argument("--class-id", type=int, help="Also show this class's probability map")
    args = parser.parse_args()
    cfg = load_run_config(args.weight, args.config)
    device = torch.device(args.device) if args.device else pick_device()
    model = load_model(cfg, args.weight, device)
    _, panels = run(model, args.image, args.tiles, args.mask, args.class_id)
    figure([panels])
    plt.show()


if __name__ == "__main__":
    main()
