import numpy as np
from PIL import Image


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"P", "L", "I", "I;16", "1"}:
            raise ValueError(f"Mask must contain class indices, got mode {image.mode}: {path}")
        return np.asarray(image, dtype=np.int64)
