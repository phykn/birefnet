import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._draw import make_row, show
from src.config import load_config, resolve
from src.data.dataset import MaskDataset
from src.data.io import pair_files
from src.data.model_input import MEAN, STD


def main():
    parser = argparse.ArgumentParser(description="Show augmented training samples")
    parser.add_argument("--config")
    cfg = load_config(parser.parse_args().config)
    labels = resolve(cfg.data.label_dir) if cfg.data.label_dir else None
    pairs = pair_files(resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir), labels)
    dataset = MaskDataset(pairs, cfg)
    rows = []
    for index in random.sample(range(len(dataset)), 2):
        sample = dataset[index]
        pixels = sample["image"].transpose(1, 2, 0) * STD + MEAN
        image = np.rint(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)
        rows.append(make_row(image, sample["mask"], sample.get("label")))
    show(rows)


if __name__ == "__main__":
    main()
