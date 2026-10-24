import numpy as np
from PIL import Image

# Single-channel modes deeper than 8 bits; Pillow's RGB conversion would clip them at 255.
DEEP_MODES = {"I;16", "I;16L", "I;16B", "I;16N", "I", "F"}


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in DEEP_MODES:
            gray = stretch(np.asarray(image))
            return np.repeat(gray[..., None], 3, axis=2)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


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
            raise ValueError(f"Mask must contain class indices, got mode {image.mode}: {path}")
        return np.asarray(image, dtype=np.int64)
