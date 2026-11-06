import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, overlay
from src.config import load_config, resolve
from src.data.io import read_pair, pairs as get_pairs


def main() -> None:
    cfg = load_config()
    pairs = get_pairs(resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir))
    rows = []
    for image_path, mask_path in random.sample(pairs, 2):
        image, mask = read_pair(image_path, mask_path, cfg.model.classes)
        colors = colorize(mask, cfg.model.classes, mask != 255)
        rows.append([image, colors, overlay(image, colors)])
    figure(rows)
    plt.show()


if __name__ == "__main__":
    main()
