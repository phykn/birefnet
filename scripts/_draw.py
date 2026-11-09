import matplotlib.pyplot as plt
import numpy as np


def palette(num_classes: int) -> np.ndarray:
    if not 2 <= num_classes <= 256:
        raise ValueError("Preview supports 2 to 256 classes")
    colors = np.random.default_rng(0).integers(32, 240, (256, 3), dtype=np.uint8)
    colors[:4] = [(24, 32, 48), (174, 187, 199), (239, 89, 79), (64, 184, 220)]
    return colors


def colorize(labels: np.ndarray, num_classes: int, valid=None) -> np.ndarray:
    active = (
        np.ones(labels.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if ((labels[active] < 0) | (labels[active] >= num_classes)).any():
        raise ValueError("Mask contains invalid class IDs")
    colors = np.full((*labels.shape, 3), 100, dtype=np.uint8)
    colors[active] = palette(num_classes)[labels[active]]
    return colors


def overlay(image: np.ndarray, colors: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return np.rint(image.astype(np.float32) * 0.55 + colors * 0.45).astype(np.uint8)


def label_overlay(image: np.ndarray, label: np.ndarray, num_classes: int) -> np.ndarray:
    """Paint brush strokes over the image; unpainted pixels keep the image."""
    painted = label != 255
    base = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
    result = base.copy()
    result[painted] = colorize(label, num_classes, painted)[painted]
    return result


def figure(rows: list[list[np.ndarray]]):
    """Lay out one row of panels per sample."""
    if not rows:
        raise ValueError("Nothing to draw")
    width = max(len(row) for row in rows)
    fig, axes = plt.subplots(
        len(rows), width, figsize=(3 * width, 3 * len(rows)), squeeze=False
    )
    for axis_row, row in zip(axes, rows):
        for axis in axis_row:
            axis.axis("off")
        for axis, pixels in zip(axis_row, row):
            axis.imshow(
                pixels,
                cmap="gray" if pixels.ndim == 2 else None,
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
    fig.tight_layout()
    return fig
