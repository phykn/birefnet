import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, overlay
from src.data.dataset import MaskDataset
from src.data.io import pairs
from src.config import load_config, resolve
from src.prepare import MEAN, STD


def main() -> None:
    cfg = load_config()
    dataset = MaskDataset(
        pairs(resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir)), cfg
    )
    rows = []
    for index in random.sample(range(len(dataset)), 2):
        sample = dataset[index]
        pixels = sample["image"].transpose(1, 2, 0) * STD + MEAN
        image = np.rint(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)
        colors = colorize(sample["mask"], cfg.model.classes, sample["mask"] != 255)
        rows.append([image, colors, overlay(image, colors)])
    figure(rows)
    plt.show()


if __name__ == "__main__":
    main()
