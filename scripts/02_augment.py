import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, label_overlay, overlay
from src.data.dataset import MaskDataset
from src.data.io import pairs
from src.config import load_config, resolve
from src.prepare import MEAN, STD


def main() -> None:
    parser = argparse.ArgumentParser(description="Show augmented training samples")
    parser.add_argument("--config")
    cfg = load_config(parser.parse_args().config)
    labels = resolve(cfg.data.label_dir) if cfg.data.label_dir else None
    dataset = MaskDataset(
        pairs(resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir), labels), cfg
    )
    rows = []
    for index in random.sample(range(len(dataset)), 2):
        sample = dataset[index]
        pixels = sample["image"].transpose(1, 2, 0) * STD + MEAN
        image = np.rint(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)
        colors = colorize(sample["mask"], cfg.model.classes, sample["mask"] != 255)
        row = [image, colors, overlay(image, colors)]
        if "label" in sample:
            row.append(label_overlay(image, sample["label"], cfg.model.classes))
        rows.append(row)
    figure(rows)
    plt.show()


if __name__ == "__main__":
    main()
