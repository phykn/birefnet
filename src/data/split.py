import csv
import random
from pathlib import Path

from .pairs import Pair

NAMES = ("train", "valid")
Splits = dict[str, list[Pair]]


def load(run_dir: str | Path) -> Splits:
    root = Path(run_dir)
    splits: Splits = {}
    for name in NAMES:
        path = root / f"{name}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            if next(reader, None) != ["image", "mask"]:
                raise RuntimeError(f"Invalid split file: {path}")
            pairs = []
            for row in reader:
                if not row:
                    continue
                if len(row) != 2 or not all(row):
                    raise RuntimeError(f"Invalid split row: {path}:{reader.line_num}")
                pairs.append((row[0], row[1]))
        if not pairs:
            raise RuntimeError(f"Split file is empty: {path}")
        splits[name] = pairs
    return splits


def save(splits: Splits, run_dir: str | Path) -> None:
    root = Path(run_dir)
    for name in NAMES:
        pairs = splits[name]
        with (root / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["image", "mask"])
            writer.writerows((Path(image).name, Path(mask).name)
                             for image, mask in pairs)


def make(pairs: list[Pair], valid_ratio: float) -> Splits:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between zero and one")
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required")
    pairs = list(pairs)
    random.shuffle(pairs)
    valid_n = min(len(pairs) - 1, max(1, round(len(pairs) * valid_ratio)))
    return {
        "valid": pairs[:valid_n],
        "train": pairs[valid_n:],
    }


def restore(pairs: list[Pair], splits: Splits) -> Splits:
    available = {
        (Path(image).name, Path(mask).name): (image, mask) for image, mask in pairs
    }
    used: list[Pair] = []
    groups: Splits = {}
    for name in NAMES:
        saved = splits.get(name)
        if saved is None:
            raise RuntimeError(f"Saved split is incomplete: {name}")
        if not saved:
            raise RuntimeError(f"Saved split is empty: {name}")
        keys = [(Path(image).name, Path(mask).name) for image, mask in saved]
        missing = [key for key in keys if key not in available]
        if missing:
            raise RuntimeError(f"Saved split files are missing: {missing[:5]}")
        groups[name] = [available[key] for key in keys]
        used.extend(keys)

    if len(used) != len(set(used)):
        raise RuntimeError("Saved splits contain duplicate pairs")
    if set(used) != set(available):
        raise RuntimeError("Current dataset does not match the saved splits")
    return groups
