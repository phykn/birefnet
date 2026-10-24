"""Show random training samples after augmentation."""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, overlay
from src.build.data import build
from src.config import load_config
from src.prepare.convert import MEAN, STD


def main() -> None:
    cfg = load_config()
    dataset = build(cfg)[0].dataset
    rows = []
    for index in random.sample(range(len(dataset)), 2):
        sample = dataset[index]
        gray = np.clip((sample["image"][0] * STD[0, 0, 0] + MEAN[0, 0, 0]) * 255, 0, 255).astype(np.uint8)
        image = np.repeat(gray[..., None], 3, axis=-1)
        colors = colorize(sample["mask"], cfg.model.classes, sample["valid"][0] > 0)
        rows.append([image, colors, overlay(image, colors)])
    figure(rows)
    plt.show()


if __name__ == "__main__":
    main()
