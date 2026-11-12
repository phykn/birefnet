import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._draw import colorize, overlay, show
from src.data.io import read_image, read_sample
from src.device import pick_device
from src.model.checkpoint import load_model
from src.predict import predict


def main():
    parser = argparse.ArgumentParser(
        description="Predict one image and show the class map, overlay, and optional ground truth."
    )
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="Optional ground-truth mask for comparison")
    parser.add_argument(
        "--device", help="Defaults to CUDA, then Apple GPU, then CPU, whichever is present"
    )
    parser.add_argument("--tiles", type=int, nargs="+", default=[1])
    parser.add_argument("--class-id", type=int, help="Also show this class's probability map")
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    model = load_model(args.weight, device)
    if args.class_id is not None and not 0 <= args.class_id < model.num_classes:
        parser.error(f"--class-id must be in [0, {model.num_classes})")
    if args.mask:
        image, target = read_sample((args.image, args.mask), model.num_classes)
        panels = [image, colorize(target)]
    else:
        image = read_image(args.image)
        panels = [image]
    result = predict(model, image, tiles=args.tiles)
    colors = colorize(result["classes"])
    panels += [colors, overlay(image, colors)]
    if args.class_id is not None:
        panels.append(np.rint(result["probs"][args.class_id] * 255).astype(np.uint8))
    show([panels])


if __name__ == "__main__":
    main()
