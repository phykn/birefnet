import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# Single-channel modes deeper than 8 bits; Pillow's RGB conversion would clip them at 255.
DEEP_MODES = {"I;16", "I;16L", "I;16B", "I;16N", "I", "F"}


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in DEEP_MODES:
            gray = stretch(np.asarray(image))
            return gray
        return cv2.cvtColor(
            np.asarray(image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY
        )


def stretch(values: np.ndarray) -> np.ndarray:
    """Map an image's own darkest to brightest value onto 0-255.

    Deep SEM images often use only part of their range, such as 12-bit data in a
    16-bit file, so a fixed division would leave few gray levels.
    """
    values = values.astype(np.float64)
    low, high = values.min(), values.max()
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.rint((values - low) / (high - low) * 255.0).astype(np.uint8)


def read_mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"P", "L", "I", "I;16", "1"}:
            raise ValueError(
                f"Mask must contain class indices, got mode {image.mode}: {path}"
            )
        return np.asarray(image, dtype=np.int64)


Pair = tuple[str, ...]
FOLDERS = ("images", "masks", "labels")


def index(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        if stem in result:
            raise ValueError(f"Duplicate stem {stem!r}: {result[stem]} vs {path}")
        result[stem] = path
    return result


def pair_files(*path_lists: list[str]) -> list[Pair]:
    """Match image, mask and optional label files by filename stem."""
    folders = [index(paths) for paths in path_lists]
    stems = set().union(*folders)
    missing = {
        f"missing_{name}": sorted(stems - set(folder))[:5]
        for name, folder in zip(FOLDERS, folders)
        if stems - set(folder)
    }
    if missing:
        detail = ", ".join(f"{key}={value}" for key, value in missing.items())
        raise ValueError(f"File stems do not match: {detail}")
    return [tuple(folder[stem] for folder in folders) for stem in sorted(stems)]


def pairs(images, masks, labels=None):
    folders = [images, masks] + ([labels] if labels else [])
    return pair_files(
        *([str(p) for p in Path(folder).iterdir() if p.is_file()] for folder in folders)
    )


def check_classes(values, path, classes, shape):
    invalid = (values != 255) & ((values < 0) | (values >= classes))
    if invalid.any():
        raise ValueError(
            f"{path}: invalid class indices {np.unique(values[invalid]).tolist()}"
        )
    if values.shape != shape:
        raise ValueError(f"{path} {values.shape} != image {shape}")


def read_pair(image_path, mask_path, classes):
    image, mask = read_image(image_path), read_mask(mask_path)
    check_classes(mask, mask_path, classes, image.shape[:2])
    return image, mask


def read_sample(paths: Pair, classes):
    """Read an image with its mask and, when present, its label; 255 in a label means unpainted."""
    image, mask = read_pair(paths[0], paths[1], classes)
    sample = [image, mask]
    for path in paths[2:]:
        label = read_mask(path)
        check_classes(label, path, classes, image.shape[:2])
        sample.append(label)
    return sample
