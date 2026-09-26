import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalize(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x /= 255.0
    x -= MEAN
    x /= STD
    return x


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got shape {image.shape}")
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[..., :3]
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)


def _zscore(gray: np.ndarray, std_range: float = 3.0) -> np.ndarray:
    x = gray.astype(np.float32)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    low, high = mean - std_range * std, mean + std_range * std
    x = np.clip(x, low, high)
    x = (x - low) / max(high - low, 1e-6)
    return np.rint(x * 255.0).astype(np.uint8)


def _enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(_zscore(gray))
    return clahe.apply(_zscore(x))


def _percentile(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1, 99))
    if high <= low:
        return np.zeros_like(gray)
    x = (gray.astype(np.float32) - low) / (high - low)
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def convert(image: np.ndarray, is_sem: bool = False) -> np.ndarray:
    if not isinstance(is_sem, bool):
        raise ValueError("is_sem must be a bool")
    rgb = _to_rgb(image)
    if not is_sem:
        return rgb
    gray = to_gray(rgb)
    return np.stack([gray, _percentile(gray), _enhance(gray)], axis=-1)
