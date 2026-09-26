import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


def erode_valid(valid: torch.Tensor, radius: int) -> torch.Tensor:
    return 1.0 - F.max_pool2d(1.0 - valid, 2 * radius + 1, 1, radius)


def make_band(target: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 1:
        raise ValueError("boundary radius must be >= 1")
    padded = F.pad(target, (radius,) * 4, value=0.0)
    kernel = 2 * radius + 1
    return (F.max_pool2d(padded, kernel, 1) + F.max_pool2d(-padded, kernel, 1) > 0).to(target.dtype)


def prepare_target(logits, target, valid=None, ignore_index=255):
    if target.ndim != 3 or target.dtype != torch.long:
        raise ValueError("target must be a long tensor with shape B,H,W")
    if logits.ndim != 4 or logits.shape[0] != target.shape[0]:
        raise ValueError("logits must have shape B,C,H,W and match target batch")
    invalid = (target != ignore_index) & ((target < 0) | (target >= logits.shape[1]))
    if invalid.any():
        raise ValueError("target contains invalid class indices")
    if valid is None:
        valid = torch.ones_like(target[:, None], dtype=logits.dtype)
    if valid.shape != target[:, None].shape:
        raise ValueError("valid must have shape B,1,H,W matching target")
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target[:, None].float(), logits.shape[-2:], mode="nearest")[:, 0].long()
        valid = F.interpolate(valid.float(), logits.shape[-2:], mode="nearest")
    valid = valid.to(logits.dtype).clamp(0, 1) * (target[:, None] != ignore_index)
    safe = target.masked_fill(target == ignore_index, 0)
    onehot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    return target, valid, onehot


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, valid=None):
        if valid is None:
            valid = torch.ones_like(pred[:, :1])
        dims = (0, 2, 3)
        intersection = (pred * target * valid).sum(dims)
        total = ((pred + target) * valid).sum(dims)
        loss = 1.0 - (2 * intersection + self.eps) / (total + self.eps)
        return loss.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_region=1.0, lambda_boundary=0.5,
                 boundary_radius=3, num_classes=4, ignore_index=255):
        super().__init__()
        if num_classes < 2 or 0 <= ignore_index < num_classes:
            raise ValueError("invalid class count or ignore index")
        if boundary_radius < 1:
            raise ValueError("boundary_radius must be >= 1")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.region = DiceLoss()
        self.radius = boundary_radius
        self.lambda_cls = lambda_cls
        self.lambda_region = lambda_region
        self.lambda_boundary = lambda_boundary

    def compute(self, pred, target, valid=None, cut=None, include_boundary=True):
        # High-resolution reductions overflow float16 even when logits are finite.
        pred = pred.float()
        if pred.shape[1] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} logit channels, got {pred.shape[1]}")
        target, valid, onehot = prepare_target(pred, target, valid, self.ignore_index)
        ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction="none")[:, None]
        cls = masked_mean(ce, valid)
        region = self.region(pred.softmax(1), onehot, valid)
        boundary = pred.sum() * 0.0
        if include_boundary:
            active = make_band(onehot, self.radius).amax(1, keepdim=True) * erode_valid(valid, self.radius)
            if cut is not None:
                cut = F.interpolate(cut.float(), pred.shape[-2:], mode="nearest")
                blocked = F.max_pool2d(cut, 2 * self.radius + 1, 1, self.radius)
                active = active * (1 - blocked)
            boundary = masked_mean(ce, active)
        return {"cls_raw": cls, "region_raw": region, "boundary_raw": boundary,
                "cls": cls * self.lambda_cls, "region": region * self.lambda_region,
                "boundary": boundary * self.lambda_boundary}

    def forward(self, pred, target, valid=None, cut=None, include_boundary=True):
        parts = self.compute(pred, target, valid, cut, include_boundary)
        return parts["cls"] + parts["region"] + parts["boundary"]
