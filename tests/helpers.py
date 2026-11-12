import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.config import load_config
from src.model import Output
from src.train.loss import TrainLoss
from src.train.schedule import cosine_schedule
from src.train.trainer import Trainer

COLORS = [(24, 32, 48), (174, 187, 199), (239, 89, 79), (64, 184, 220)]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(a, b, 1) for a, b in ((4, 4), (4, 8), (8, 16), (16, 32))
        )
        for idx, channels in enumerate([4, 8, 16, 32]):
            setattr(self, f"norm{idx}", nn.BatchNorm2d(channels))

    def forward(self, x):
        x = self.patch_embed(F.avg_pool2d(x, 4))
        outs = []
        for idx, layer in enumerate(self.layers):
            if idx:
                x = F.avg_pool2d(x, 2)
            x = getattr(self, f"norm{idx}")(layer(x))
            outs.append(x)
        return outs


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv2d(3, 4, 1)
        self.num_classes = 4

    def forward(self, x):
        return Output([self.decoder(x)])


def make_loss(classes=4, **loss):
    cfg = load_config()
    cfg.model.classes = classes
    for key, value in loss.items():
        setattr(cfg.loss, key, value)
    return TrainLoss(cfg)


def trainer(tmp_path, model=None):
    cfg = load_config()
    cfg.train.steps = 4
    cfg.train.warmup = 0
    cfg.train.accum = 1
    m = model or Tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.train.lr)
    sch = cosine_schedule(opt, 4, cfg.train.min_lr)
    batches = [
        {
            "image": torch.rand(1, 3, 16, 16),
            "mask": torch.zeros(1, 16, 16, dtype=torch.long),
        }
    ]
    return Trainer(m, batches, [], TrainLoss(cfg), opt, sch, tmp_path, cfg)


def make_sample(index, size=256):
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
        points = np.array(
            [(cx - 27, cy + 8), (cx - 9, cy - 3), (cx + 9, cy + 6), (cx + 27, cy - 9)],
            np.int32,
        )
        cv2.polylines(crack, [points], False, 1, 2 + index % 2)
        cv2.line(crack, (cx - 9, cy - 3), (cx - 7, cy - 21), 1, 2)
        mask[(crack == 1) & (solid == 1)] = 2
        cv2.ellipse(mask, (cx + 8, cy + 24), (8 + index, 6), 20, 0, 360, 3, -1)
    image = np.array([32, 168, 39, 28], np.float32)[mask]
    distance = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
    image += np.minimum(distance, 25) * 1.1 + rng.normal(0, 8, mask.shape)
    edge = cv2.morphologyEx(solid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    image = cv2.GaussianBlur(image + edge * 20, (3, 3), 0.45)
    return np.clip(image, 0, 255).astype(np.uint8), mask


def write(root):
    (root / "image").mkdir(parents=True)
    (root / "mask").mkdir()
    palette = [value for color in COLORS for value in color] + [0] * (768 - 12)
    for index in range(4):
        name = f"sample_{index + 1:02d}.png"
        image, mask = make_sample(index)
        Image.fromarray(image).save(root / "image" / name)
        indexed = Image.fromarray(mask).convert("P")
        indexed.putpalette(palette)
        indexed.save(root / "mask" / name)


def write_labels(root, paint):
    (root / "label").mkdir()
    for mask_path in sorted((root / "mask").iterdir()):
        mask = np.asarray(Image.open(mask_path))
        label = np.where(paint(mask), mask, 255).astype(np.uint8)
        Image.fromarray(label).save(root / "label" / mask_path.name)
