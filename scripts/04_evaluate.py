import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve
from src.data.io import pairs
from src.device import pick_device
from src.model.checkpoint import load
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Score a saved model on paired images and masks"
    )
    parser.add_argument("--weight", required=True)
    parser.add_argument("--images", default=str(resolve("data/test/image")))
    parser.add_argument("--masks", default=str(resolve("data/test/mask")))
    parser.add_argument("--tiles", type=int, nargs="+", default=[1])
    parser.add_argument("--device")
    args = parser.parse_args()
    model = load(
        args.weight, torch.device(args.device) if args.device else pick_device()
    )
    data = pairs(args.images, args.masks)
    result = evaluate(model, data, args.tiles)
    print(f"pairs: {len(data)}")
    for name, value in result.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
