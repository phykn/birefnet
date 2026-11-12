import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, label_overlay, overlay
from src.config import load_config, resolve
from src.data.io import pair_files, read_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Show source images with masks and labels")
    parser.add_argument("--config")
    cfg = load_config(parser.parse_args().config)
    labels = resolve(cfg.data.label_dir) if cfg.data.label_dir else None
    pairs = pair_files(resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir), labels)
    rows = []
    for pair in random.sample(pairs, 2):
        image, mask, *label = read_sample(pair, cfg.model.classes)
        colors = colorize(mask, cfg.model.classes, mask != 255)
        row = [image, colors, overlay(image, colors)]
        if label:
            row.append(label_overlay(image, label[0], cfg.model.classes))
        rows.append(row)
    figure(rows)
    plt.show()


if __name__ == "__main__":
    main()
