import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._draw import make_row, show
from src.config import load_config, resolve
from src.data.io import pair_files, read_sample


def main():
    parser = argparse.ArgumentParser(
        description="Show source images with masks and labels"
    )
    parser.add_argument("--config")
    cfg = load_config(parser.parse_args().config)
    labels = resolve(cfg.data.label_dir) if cfg.data.label_dir else None
    pairs = pair_files(resolve(cfg.data.image_dir), resolve(cfg.data.mask_dir), labels)
    show(
        [
            make_row(*read_sample(pair, cfg.model.classes))
            for pair in random.sample(pairs, 2)
        ]
    )


if __name__ == "__main__":
    main()
