import random

import numpy as np
from .model_input import fit_image, fit_mask, plan_fit
from .augment import (
    apply_blur,
    apply_gamma,
    apply_noise,
    apply_shading,
    hide_edges,
    pick_crop,
    pick_painted_pixel,
    rotate_flip,
)
from .io import read_sample


class MaskDataset:
    def __init__(self, pairs, cfg):
        self.pairs, self.cfg = pairs, cfg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, *targets = read_sample(self.pairs[index], self.cfg.model.classes)
        image, targets, cut, side = self.crop(image, targets)
        image, targets = self.augment(image, targets)
        return self.to_tensor(image, targets, cut, side)

    def crop(self, image, targets):
        focus = None
        if len(targets) > 1:
            focus = pick_painted_pixel(targets[1])
        region, cut, side = pick_crop(
            image.shape[:2], self.cfg.augment.zoom, self.cfg.train.crop, focus
        )
        return image[region], [t[region] for t in targets], cut, side

    def augment(self, image, targets):
        a = self.cfg.augment
        if random.random() < a.gamma.p:
            image = apply_gamma(image, a.gamma.range)
        if random.random() < a.shading.p:
            image = apply_shading(image, a.shading.max)
        if random.random() < a.blur.p:
            image = apply_blur(image, a.blur.max_sigma)
        if random.random() < a.noise.p:
            image = apply_noise(image, a.noise.max_std)
        if random.random() < a.hide_edges.p:
            image, *targets = hide_edges(image, *targets)
        return image, targets

    def to_tensor(self, image, targets, cut, side):
        size = self.cfg.train.crop
        fit = plan_fit(*image.shape, size=size, scale=size / side)
        tensor, _ = fit_image(image, fit=fit)
        targets = [fit_mask(t, fit, padding=255) for t in targets]
        cut = fit_mask(cut, fit)
        pixels, cut, *targets = rotate_flip(tensor.transpose(1, 2, 0), cut, *targets)
        sample = {
            "image": np.ascontiguousarray(pixels.transpose(2, 0, 1)),
            "mask": targets[0],
            "cut": cut[None],
        }
        if len(targets) > 1:
            sample["label"] = targets[1]
        return sample
