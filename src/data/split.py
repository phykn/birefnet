import csv
import random
from pathlib import Path


def write_split_csv(
    splits: dict[str, list[tuple[str, ...]]], run_dir: str | Path
) -> None:
    for name in ("train", "valid"):
        pairs = splits[name]
        path = Path(run_dir) / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "mask", "label"][: len(pairs[0])])
            writer.writerows(tuple(Path(p).name for p in pair) for pair in pairs)


def split_pairs(
    pairs: list[tuple[str, ...]], valid_ratio: float, seed: int = 0
) -> dict[str, list[tuple[str, ...]]]:
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    valid_n = min(len(pairs) - 1, max(1, round(len(pairs) * valid_ratio)))
    return {"valid": pairs[:valid_n], "train": pairs[valid_n:]}


def read_split_csv(
    pairs: list[tuple[str, ...]], run_dir: str | Path
) -> dict[str, list[tuple[str, ...]]]:
    available = {tuple(Path(p).name for p in pair): pair for pair in pairs}
    width = len(pairs[0])
    groups = {}
    used = []
    for name in ("train", "valid"):
        path = Path(run_dir) / f"{name}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            header, *keys = [tuple(row) for row in csv.reader(file) if row]
        if len(header) != width:
            raise RuntimeError(
                "Saved split and data.label_dir disagree on whether labels are used"
            )
        if not keys:
            raise RuntimeError(f"Split file is empty: {path}")
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
