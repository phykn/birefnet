import numpy as np
from ..prepare import fit_tensor, fit_mask, plan
from .augment import blur, edge_mask, flip, gamma, illumination, jitter, noise, zoom
from .io import read_pair


class MaskDataset:
    def __init__(self, pairs, cfg):
        self.pairs, self.cfg = pairs, cfg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        cfg = self.cfg
        a = cfg.augment
        image, mask = read_pair(*self.pairs[index], cfg.model.classes)
        image, mask, cut, side = zoom(image, mask, a.zoom, cfg.train.crop)
        image = jitter(image, (a.brightness, a.contrast))
        image = gamma(image, a.gamma)
        image = illumination(image, a.light)
        image = blur(image, a.blur)
        image = noise(image, a.noise)
        if np.random.random() < a.hide_edges:
            image, mask = edge_mask(image, mask)
        fit = plan(*image.shape[:2], size=cfg.train.crop, scale=cfg.train.crop / side)
        tensor, _ = fit_tensor(image, is_sem=True, fit=fit)
        labels = fit_mask(mask, fit, padding=255).astype(np.int64)
        cut = fit_mask(cut, fit)
        pixels, labels, cut = flip(tensor.transpose(1, 2, 0), labels, cut)
        return {
            "image": np.ascontiguousarray(pixels.transpose(2, 0, 1)),
            "mask": labels,
            "cut": cut[None],
        }
