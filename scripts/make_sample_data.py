"""Generate four deterministic synthetic SEM-like pairs, not experimental data."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLORS = [(24, 32, 48), (174, 187, 199), (239, 89, 79), (64, 184, 220)]


def generate(index: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260926 + index)
    mask = np.zeros((size, size), np.uint8)
    solid = np.zeros_like(mask)
    particles = []
    for cy in (63, 191):
        for cx in (63, 191):
            center = (cx + int(rng.integers(-9, 10)), cy + int(rng.integers(-9, 10)))
            axes = (int(rng.integers(42, 55)), int(rng.integers(40, 56)))
            cv2.ellipse(solid, center, axes, float(rng.integers(0, 180)), 0, 360, 1, -1)
            particles.append(center)
    mask[solid == 1] = 1
    for cx, cy in particles:
        crack = np.zeros_like(mask)
        points = np.array([(cx-27, cy+8), (cx-9, cy-3), (cx+9, cy+6), (cx+27, cy-9)], np.int32)
        cv2.polylines(crack, [points], False, 1, 2 + index % 2)
        cv2.line(crack, (cx-9, cy-3), (cx-7, cy-21), 1, 2)
        mask[(crack == 1) & (solid == 1)] = 2
        cv2.ellipse(mask, (cx+8, cy+24), (8 + index, 6), 20, 0, 360, 3, -1)
    levels = np.array([32, 168, 39, 28], np.float32)
    image = levels[mask]
    texture = rng.normal(0, 8, mask.shape).astype(np.float32)
    distance = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
    image += np.minimum(distance, 25) * 1.1 + texture
    edge = cv2.morphologyEx(solid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    image += edge * 20
    image = cv2.GaussianBlur(image, (3, 3), 0.45)
    return np.clip(image, 0, 255).astype(np.uint8), mask


def main() -> None:
    image_dir, mask_dir = ROOT / "data/image", ROOT / "data/mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    palette = [value for color in COLORS for value in color] + [0] * (768 - 12)
    previews = []
    for index in range(4):
        name = f"sample_{index+1:02d}.png"
        paths = (image_dir / name, mask_dir / name)
        if any(path.exists() for path in paths):
            raise FileExistsError(f"Refusing to overwrite existing sample: {name}")
        image, mask = generate(index)
        Image.fromarray(image).save(paths[0])
        indexed = Image.fromarray(mask).convert("P")
        indexed.putpalette(palette)
        indexed.save(paths[1])
        assert set(np.unique(np.asarray(Image.open(paths[1])))) == {0, 1, 2, 3}
        previews.append(np.concatenate([np.repeat(image[..., None], 3, -1), np.asarray(indexed.convert("RGB"))], axis=1))
    Image.fromarray(np.concatenate(previews, axis=0)).save(ROOT / "data/preview.png")
    print("Created four 256x256 synthetic image/palette-mask pairs in data/")


if __name__ == "__main__":
    main()
