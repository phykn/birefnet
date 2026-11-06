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


Pair = tuple[str, str]


def index(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        if stem in result:
            raise ValueError(f"Duplicate stem {stem!r}: {result[stem]} vs {path}")
        result[stem] = path
    return result


def pair_files(image_paths: list[str], mask_paths: list[str]) -> list[Pair]:
    images = index(image_paths)
    masks = index(mask_paths)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask stems do not match: "
            f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


def pairs(images, masks):
    return pair_files(
        [str(p) for p in Path(images).iterdir() if p.is_file()],
        [str(p) for p in Path(masks).iterdir() if p.is_file()],
    )


def read_pair(image_path, mask_path, classes):
    image, mask = read_image(image_path), read_mask(mask_path)
    invalid = (mask != 255) & ((mask < 0) | (mask >= classes))
    if invalid.any():
        raise ValueError(
            f"{mask_path}: invalid class indices {np.unique(mask[invalid]).tolist()}"
        )
    if image.shape[:2] != mask.shape:
        raise ValueError(f"{image_path} {image.shape} != {mask_path} {mask.shape}")
    return image, mask
