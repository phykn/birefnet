import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode.startswith("I;16"):
            values = np.asarray(image)
            scaled = values.astype(np.float64) / np.iinfo(values.dtype).max * 255.0
            return np.rint(scaled).astype(np.uint8)
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def pair_files(images, masks, labels=None) -> list[tuple[str, ...]]:
    folders = []
    for folder in [images, masks] + ([labels] if labels else []):
        files = {}
        for p in Path(folder).iterdir():
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.stem in files:
                raise ValueError(f"Duplicate stem {p.stem!r}: {files[p.stem]} vs {p}")
            files[p.stem] = str(p)
        folders.append(files)
    stems = set().union(*folders)
    missing = {
        f"missing_{name}": sorted(stems - set(folder))[:5]
        for name, folder in zip(("images", "masks", "labels"), folders)
        if stems - set(folder)
    }
    if missing:
        detail = ", ".join(f"{key}={value}" for key, value in missing.items())
        raise ValueError(f"File stems do not match: {detail}")
    return [tuple(folder[stem] for folder in folders) for stem in sorted(stems)]


def read_sample(paths: tuple[str, ...], classes):
    image = read_image(paths[0])
    sample = [image]
    for path in paths[1:]:
        with Image.open(path) as file:
            if file.mode not in {"P", "L", "I", "I;16", "1"}:
                raise ValueError(
                    f"Mask must contain class indices, got mode {file.mode}: {path}"
                )
            values = np.asarray(file, dtype=np.int64)
        invalid = (values != 255) & ((values < 0) | (values >= classes))
        if invalid.any():
            raise ValueError(
                f"{path}: invalid class indices {np.unique(values[invalid]).tolist()}"
            )
        if values.shape != image.shape:
            raise ValueError(f"{path} {values.shape} != image {image.shape}")
        sample.append(values)
    return sample
