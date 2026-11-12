import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, overlay
from src.data.io import read_image, read_sample
from src.device import pick_device
from src.predict import predict
from src.model.checkpoint import load as load_model


def run(
    model,
    image_path: Path,
    tiles=(1,),
    mask_path=None,
    class_id=None,
) -> tuple[np.ndarray, list]:
    if class_id is not None and not 0 <= class_id < model.num_classes:
        raise ValueError("class_id must be in [0, num_classes)")
    if mask_path is not None:
        image, target = read_sample((image_path, mask_path), model.num_classes)
        panels = [image, colorize(target, model.num_classes, target != 255)]
    else:
        image = read_image(image_path)
        panels = [image]
    result = predict(model, image, tiles=tiles)
    labels = result["classes"].astype(np.uint8)
    colors = colorize(labels, model.num_classes)
    panels.extend([colors, overlay(image, colors)])
    if class_id is not None:
        panels.append(np.rint(result["probs"][class_id] * 255).astype(np.uint8))
    return labels, panels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict one image and show the class map, overlay, and optional ground truth."
    )
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--mask", type=Path, help="Optional ground-truth mask for comparison"
    )
    parser.add_argument(
        "--device",
        help="Defaults to CUDA, then Apple GPU, then CPU, whichever is present",
    )
    parser.add_argument("--tiles", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--class-id", type=int, help="Also show this class's probability map"
    )
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    model = load_model(args.weight, device)
    _, panels = run(model, args.image, args.tiles, args.mask, args.class_id)
    figure([panels])
    plt.show()


if __name__ == "__main__":
    main()
