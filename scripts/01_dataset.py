"""Show random original image/mask pairs."""
import random
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._draw import colorize, figure, overlay
from src.config import load_config
from src.data.image import read_image, read_mask
from src.data.pairs import pair_files


def main() -> None:
    cfg = load_config()
    pairs = pair_files(glob(f"{cfg.data.image_dir}/*"), glob(f"{cfg.data.mask_dir}/*"))
    rows = []
    for image_path, mask_path in random.sample(pairs, 2):
        image = read_image(image_path)
        colors = colorize(read_mask(mask_path), cfg.model.classes)
        rows.append([image, colors, overlay(image, colors)])
    figure(rows)
    plt.show()


if __name__ == "__main__":
    main()
