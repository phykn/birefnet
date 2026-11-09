import cv2
import numpy as np
from ..prepare import fit_tensor, fit_mask, plan
from .augment import (
    blur,
    edge_mask,
    flip,
    gamma,
    illumination,
    jitter,
    noise,
    pick_focus,
    zoom,
)
from .io import read_sample


def hide_boundaries(mask: np.ndarray, radius: int) -> np.ndarray:
    """Ignore mask pixels within ``radius`` of another class; the image border is no boundary."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    labeled = mask != 255
    near = np.zeros(mask.shape, dtype=bool)
    for cls in np.unique(mask[labeled]):
        inside = (mask == cls).astype(np.uint8)
        near |= cv2.dilate(inside, kernel).astype(bool) & labeled & (mask != cls)
    return np.where(near, 255, mask)


class MaskDataset:
    def __init__(self, pairs, cfg):
        self.pairs, self.cfg = pairs, cfg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        cfg = self.cfg
        a = cfg.augment
        image, mask, *labels = read_sample(self.pairs[index], cfg.model.classes)
        focus = None
        if labels:
            if cfg.loss.mask_edge:
                mask = hide_boundaries(mask, cfg.loss.mask_edge)
            focus = pick_focus(labels[0])
        region, cut, side = zoom(image.shape[:2], a.zoom, cfg.train.crop, focus)
        image, mask = image[region].copy(), mask[region].copy()
        labels = [label[region] for label in labels]
        image = jitter(image, (a.brightness, a.contrast))
        image = gamma(image, a.gamma)
        image = illumination(image, a.light)
        image = blur(image, a.blur)
        image = noise(image, a.noise)
        if np.random.random() < a.hide_edges:
            image, mask, *labels = edge_mask(image, mask, *labels)
        fit = plan(*image.shape[:2], size=cfg.train.crop, scale=cfg.train.crop / side)
        tensor, _ = fit_tensor(image, is_sem=True, fit=fit)
        targets = [
            fit_mask(m, fit, padding=255).astype(np.int64) for m in (mask, *labels)
        ]
        cut = fit_mask(cut, fit)
        pixels, cut, *targets = flip(tensor.transpose(1, 2, 0), cut, *targets)
        sample = {
            "image": np.ascontiguousarray(pixels.transpose(2, 0, 1)),
            "mask": targets[0],
            "cut": cut[None],
        }
        if labels:
            sample["label"] = targets[1]
        return sample
