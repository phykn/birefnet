import matplotlib.pyplot as plt
import numpy as np

PALETTE = np.random.default_rng(0).integers(32, 240, (256, 3), dtype=np.uint8)
PALETTE[:4] = [(24, 32, 48), (174, 187, 199), (239, 89, 79), (64, 184, 220)]


def colorize(labels):
    colors = PALETTE[labels]
    colors[labels == 255] = 100
    return colors


def overlay(image, colors):
    return np.rint(np.atleast_3d(image) * 0.55 + colors * 0.45).astype(np.uint8)


def overlay_label(image, label):
    painted = (label != 255)[..., None]
    return np.where(painted, colorize(label), np.atleast_3d(image))


def make_row(image, mask, label=None):
    colors = colorize(mask)
    row = [image, colors, overlay(image, colors)]
    if label is not None:
        row.append(overlay_label(image, label))
    return row


def show(rows):
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
    plt.show()
