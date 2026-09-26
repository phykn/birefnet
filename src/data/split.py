import csv
import random
from pathlib import Path

from .pairs import Pair

NAMES = ("train", "valid")
Groups = dict[str, list[Pair]]
Splits = dict[str, list[str]]


def load(run_dir: str | Path) -> Splits:
    root = Path(run_dir)
    splits: Splits = {}
    for name in NAMES:
        path = root / f"{name}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["image", "mask"]:
                raise RuntimeError(f"Invalid split file: {path}")
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"Split file is empty: {path}")
        splits[f"{name}_image"] = [row["image"] for row in rows]
        splits[f"{name}_mask"] = [row["mask"] for row in rows]
    return splits


def save(splits: Splits, run_dir: str | Path) -> None:
    root = Path(run_dir)
    for name in NAMES:
        images = splits[f"{name}_image"]
        masks = splits[f"{name}_mask"]
        if len(images) != len(masks):
            raise RuntimeError(f"Split image/mask counts differ: {name}")
        with (root / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["image", "mask"])
            writer.writerows(zip(images, masks))


def make(data: list[Pair], valid_ratio: float) -> Groups:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between zero and one")
    if len(data) < 2:
        raise ValueError("At least two image/mask pairs are required")
    data = list(data)
    random.shuffle(data)
    valid_n = min(len(data) - 1, max(1, round(len(data) * valid_ratio)))
    return {
        "valid": data[:valid_n],
        "train": data[valid_n:],
    }


def restore(data: list[Pair], splits: Splits) -> Groups:
    available = {
        (Path(image).name, Path(mask).name): (image, mask) for image, mask in data
    }
    used: list[Pair] = []
    groups: Groups = {}
    for name in NAMES:
        images = splits.get(f"{name}_image")
        masks = splits.get(f"{name}_mask")
        if images is None or masks is None or len(images) != len(masks):
            raise RuntimeError(f"Saved split is incomplete: {name}")
        keys = list(zip(images, masks))
        if not keys:
            raise RuntimeError(f"Saved split is empty: {name}")
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


def pack(groups: Groups) -> Splits:
    splits: Splits = {}
    for name in NAMES:
        pairs = groups[name]
        splits[f"{name}_image"] = [Path(image).name for image, _ in pairs]
        splits[f"{name}_mask"] = [Path(mask).name for _, mask in pairs]
    return splits
