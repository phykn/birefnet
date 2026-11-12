import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve

CLASSES = 5
DEFECTS_PER_IMAGE = (1, 3)
OBJECTS_PER_CLASS = (2, 5)
CIRCLE_RADIUS = (16, 40)
ELLIPSE_RADIUS = (60, 140)
LINE_WIDTH = (10, 16)
LINE_LENGTH = (160, 600)
BLOB_RADIUS = (30, 80)
BRIGHTNESS = (110, 200, 70, 170, 140)
TEXTURE_SIGMA = (4, 12)
LIGHT = 20
BLUR_SIGMA = 1.0
NOISE_SIGMA = 6
LABEL_FRACTION = (0.005, 0.03)
MAX_STROKES = 40
BACKGROUND_STROKE = 0.3
BRUSH_WIDTH = (5, 9)
NARROW_BRUSH = 3
SEGMENTS = (2, 4)
SEGMENT_LENGTH = (30, 150)
OVERFLOW = 0.3
OVERFLOW_PIXELS = (1, 3)
WRONG_STROKE = 0.05
EDGE_SHIFT = 6
ERASE = 0.1
SWAP = 0.1
FAKE_BLOBS = (0, 2)
FAKE_RADIUS = (12, 30)

PALETTE = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 200, 0)]


def generate(out, seed=0, train=300, test=100, size=1024):
    rng = np.random.default_rng(seed)
    flat = [channel for color in PALETTE for channel in color]
    palette = flat + [128] * (768 - len(flat))
    painted = np.zeros(CLASSES, dtype=np.int64)
    total = 0
    for split, count in (("train", train), ("test", test)):
        folders = {}
        for name in ("image", "mask", "label", "truth"):
            folders[name] = Path(out) / split / name
            folders[name].mkdir(parents=True, exist_ok=True)
        for index in range(count):
            sample = make_sample(rng, size)
            stem = f"{index:06d}.png"
            Image.fromarray(sample["image"], mode="L").save(folders["image"] / stem)
            for name in ("mask", "label", "truth"):
                classes = Image.fromarray(sample[name].astype(np.uint8), mode="P")
                classes.putpalette(palette)
                classes.save(folders[name] / stem)
            label = sample["label"]
            painted += np.bincount(label[label != 255], minlength=CLASSES)
            total += label.size
    return {"images": train + test, "label_ratio": (painted / total).tolist()}


def make_sample(rng, size):
    scale = size / 1024

    def length(low, high):
        return max(1, round(int(rng.integers(low, high + 1)) * scale))

    shapes, classes = place_objects(rng, size, length)
    truth = np.zeros((size, size), dtype=np.uint8)
    owner = np.full((size, size), -1, dtype=np.int64)
    for index, (shape, cls) in enumerate(zip(shapes, classes)):
        truth[shape] = cls
        owner[shape] = index
    label, stroked = paint_labels(rng, truth, owner, len(shapes), length)
    return {
        "image": render_image(rng, truth, size),
        "truth": truth,
        "label": label,
        "mask": make_flawed_mask(rng, truth, shapes, classes, stroked, label, length),
    }


def place_objects(rng, size, length):
    defects = rng.choice(
        np.arange(1, CLASSES),
        size=int(rng.integers(DEFECTS_PER_IMAGE[0], DEFECTS_PER_IMAGE[1] + 1)),
        replace=False,
    )
    shapes, classes = [], []
    for cls in defects:
        for _ in range(
            int(rng.integers(OBJECTS_PER_CLASS[0], OBJECTS_PER_CLASS[1] + 1))
        ):
            canvas = np.zeros((size, size), dtype=np.uint8)
            center = tuple(int(v) for v in rng.integers(0, size, 2))
            angle = float(rng.uniform(0, 360))
            if cls == 1:
                cv2.circle(canvas, center, length(*CIRCLE_RADIUS), 1, -1)
            elif cls == 2:
                axes = (length(*ELLIPSE_RADIUS), length(*ELLIPSE_RADIUS))
                cv2.ellipse(canvas, center, axes, angle, 0, 360, 1, -1)
            elif cls == 3:
                span = length(*LINE_LENGTH)
                theta = np.deg2rad(angle)
                end = (
                    int(center[0] + span * np.cos(theta)),
                    int(center[1] + span * np.sin(theta)),
                )
                cv2.line(canvas, center, end, 1, length(*LINE_WIDTH))
            else:
                radius = length(*BLOB_RADIUS)
                theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
                lobes, phase = int(rng.integers(3, 7)), float(rng.uniform(0, 2 * np.pi))
                stretch = rng.uniform(0.6, 1.0, 2)
                r = radius * (1 + 0.25 * np.sin(lobes * theta + phase))
                points = np.stack(
                    [
                        center[0] + r * stretch[0] * np.cos(theta),
                        center[1] + r * stretch[1] * np.sin(theta),
                    ],
                    axis=1,
                )
                cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
            shapes.append(canvas.astype(bool))
            classes.append(int(cls))
    return shapes, classes


def render_image(rng, truth, size):
    image = np.asarray(BRIGHTNESS, dtype=np.float32)[truth]
    texture = cv2.GaussianBlur(
        rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 2
    )
    texture /= texture.std() + 1e-6
    sigma = rng.uniform(*TEXTURE_SIGMA, CLASSES).astype(np.float32)
    image += texture * sigma[truth]
    coarse = rng.uniform(-1, 1, (4, 4)).astype(np.float32)
    image += (
        cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC).clip(-1, 1)
        * LIGHT
    )
    image = cv2.GaussianBlur(image, (0, 0), BLUR_SIGMA)
    image += rng.normal(0, NOISE_SIGMA, (size, size)).astype(np.float32)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def make_disk(radius):
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )


def paint_labels(rng, truth, owner, count, length):
    size = truth.shape[0]
    label = np.full(truth.shape, 255, dtype=np.uint8)
    goal = rng.uniform(*LABEL_FRACTION) * truth.size
    stroked = set()
    for _ in range(MAX_STROKES):
        if np.count_nonzero(label != 255) >= goal:
            break
        target = (
            -1
            if count == 0 or rng.random() < BACKGROUND_STROKE
            else int(rng.integers(count))
        )
        region = (owner == target).astype(np.uint8)
        if not region.any():
            continue
        width = length(*BRUSH_WIDTH)
        allowed = cv2.erode(region, make_disk(width // 2))
        if not allowed.any():
            width, allowed = length(NARROW_BRUSH, NARROW_BRUSH), region
        clip = allowed
        if rng.random() < OVERFLOW:
            clip = cv2.dilate(region, make_disk(length(*OVERFLOW_PIXELS)))
        ys, xs = np.nonzero(allowed)
        start = int(rng.integers(len(ys)))
        points = [(int(xs[start]), int(ys[start]))]
        for _ in range(int(rng.integers(SEGMENTS[0], SEGMENTS[1] + 1))):
            theta = rng.uniform(0, 2 * np.pi)
            step = length(*SEGMENT_LENGTH)
            x, y = points[-1]
            points.append(
                (
                    int(np.clip(x + step * np.cos(theta), 0, size - 1)),
                    int(np.clip(y + step * np.sin(theta), 0, size - 1)),
                )
            )
        canvas = np.zeros_like(region)
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, 1, width)
        stroke = (canvas & clip).astype(bool)
        value = int(truth[ys[start], xs[start]])
        if rng.random() < WRONG_STROKE:
            value = int(rng.choice([c for c in range(CLASSES) if c != value]))
        label[stroke] = value
        if target >= 0:
            stroked.add(target)
    return label, stroked


def make_flawed_mask(rng, truth, shapes, classes, stroked, label, length):
    mask = np.zeros_like(truth)
    for index, (shape, cls) in enumerate(zip(shapes, classes)):
        shift = int(rng.integers(-EDGE_SHIFT, EDGE_SHIFT + 1))
        region = shape.astype(np.uint8)
        if shift:
            radius = max(1, round(abs(shift) * truth.shape[0] / 1024))
            morph = cv2.dilate if shift > 0 else cv2.erode
            region = morph(region, make_disk(radius))
        if index not in stroked:
            draw = rng.random()
            if draw < ERASE:
                continue
            if draw < ERASE + SWAP:
                cls = int(rng.choice([c for c in range(1, CLASSES) if c != cls]))
        mask[region.astype(bool)] = cls
    ys, xs = np.nonzero(truth == 0)
    for _ in range(int(rng.integers(FAKE_BLOBS[0], FAKE_BLOBS[1] + 1))):
        if len(ys) == 0:
            break
        pick = int(rng.integers(len(ys)))
        canvas = np.zeros_like(mask)
        cv2.circle(canvas, (int(xs[pick]), int(ys[pick])), length(*FAKE_RADIUS), 1, -1)
        mask[canvas.astype(bool)] = int(rng.integers(1, CLASSES))
    painted = label != 255
    mask[painted] = label[painted]
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Write synthetic images, flawed masks and brush labels"
    )
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = generate(resolve(args.out), args.seed)
    print(f"images: {result['images']}")
    for cls, ratio in enumerate(result["label_ratio"]):
        print(f"class_{cls}_label_ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
