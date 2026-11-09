import csv
import random
from pathlib import Path

from .io import Pair

NAMES = ("train", "valid")
HEADERS = (["image", "mask"], ["image", "mask", "label"])
Splits = dict[str, list[Pair]]


def load(run_dir: str | Path) -> Splits:
    root = Path(run_dir)
    splits: Splits = {}
    for name in NAMES:
        path = root / f"{name}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header not in HEADERS:
                raise RuntimeError(f"Invalid split file: {path}")
            pairs = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header) or not all(row):
                    raise RuntimeError(f"Invalid split row: {path}:{reader.line_num}")
                pairs.append(tuple(row))
        if not pairs:
            raise RuntimeError(f"Split file is empty: {path}")
        splits[name] = pairs
    return splits


def save(splits: Splits, run_dir: str | Path) -> None:
    root = Path(run_dir)
    for name in NAMES:
        pairs = splits[name]
        with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS[len(pairs[0]) - 2])
            writer.writerows(tuple(Path(p).name for p in pair) for pair in pairs)


def make(pairs: list[Pair], valid_ratio: float, seed: int = 0) -> Splits:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between zero and one")
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required")
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    valid_n = min(len(pairs) - 1, max(1, round(len(pairs) * valid_ratio)))
    return {
        "valid": pairs[:valid_n],
        "train": pairs[valid_n:],
    }


def restore(pairs: list[Pair], splits: Splits) -> Splits:
    available = {tuple(Path(p).name for p in pair): pair for pair in pairs}
    widths = {len(key) for key in available} | {
        len(pair) for saved in splits.values() for pair in saved
    }
    if len(widths) > 1:
        raise RuntimeError(
            "Saved split and data.label_dir disagree on whether labels are used"
        )
    used: list[Pair] = []
    groups: Splits = {}
    for name in NAMES:
        saved = splits.get(name)
        if saved is None:
            raise RuntimeError(f"Saved split is incomplete: {name}")
        if not saved:
            raise RuntimeError(f"Saved split is empty: {name}")
        keys = [tuple(Path(p).name for p in pair) for pair in saved]
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
