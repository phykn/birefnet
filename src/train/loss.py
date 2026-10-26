import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import gaussian_blur2d

from ..model.output import Output


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


def soften(onehot: torch.Tensor, valid: torch.Tensor, blur: float) -> torch.Tensor:
    """Blur one-hot labels so classes mix only near their edges.

    Masks drawn by hand, or resized with the image, are off by a pixel or two at
    class edges. Ignored pixels do not leak in: the blur is renormalized by valid
    pixels.
    """
    if blur <= 0:
        return onehot
    kernel = 2 * int(round(3 * blur)) + 1
    size, sigma = (kernel, kernel), (blur, blur)
    mass = gaussian_blur2d(valid, size, sigma, border_type="constant")
    onehot = gaussian_blur2d(onehot * valid, size, sigma, border_type="constant")
    return onehot / mass.clamp_min(1e-6)


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
                 boundary_radius=3, num_classes=4, ignore_index=255,
                 label_blur=0.0):
        super().__init__()
        if num_classes < 2 or 0 <= ignore_index < num_classes:
            raise ValueError("invalid class count or ignore index")
        if boundary_radius < 1:
            raise ValueError("boundary_radius must be >= 1")
        if not 0.0 <= label_blur <= 10.0:
            raise ValueError("label_blur must be in [0, 10] pixels")
        self.label_blur = float(label_blur)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.region = DiceLoss()
        self.radius = boundary_radius
        self.lambda_cls = lambda_cls
        self.lambda_region = lambda_region
        self.lambda_boundary = lambda_boundary

    def compute(self, pred, target, valid=None, cut=None, include_boundary=True,
                teacher=None, teacher_weight=0.0):
        # High-resolution reductions overflow float16 even when logits are finite.
        pred = pred.float()
        if pred.shape[1] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} logit channels, got {pred.shape[1]}")
        # Blur is in label pixels; coarser side outputs get a proportionally smaller one.
        scale = pred.shape[-1] / target.shape[-1]
        target, valid, onehot = prepare_target(pred, target, valid, self.ignore_index)
        soft = soften(onehot, valid, self.label_blur * scale)
        if teacher is not None and teacher_weight > 0:
            # Where the averaged model disagrees with the mask, the target leans its way.
            if teacher.shape[-2:] != pred.shape[-2:]:
                teacher = F.interpolate(teacher, pred.shape[-2:], mode="area")
            soft = soft * (1.0 - teacher_weight) + teacher.detach().float() * teacher_weight
        # Mask before CE/softmax: multiplying an ignored NaN by zero is still NaN.
        pred = pred.masked_fill(valid == 0, 0)
        ce = -(soft * pred.log_softmax(1)).sum(1, keepdim=True)
        cls = masked_mean(ce, valid)
        region = self.region(pred.softmax(1), soft, valid)
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


class TrainLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_region=1.0, lambda_boundary=0.5,
                 boundary_radius=3, lambda_aux=1.0,
                 num_classes=4, ignore_index=255,
                 label_blur=0.0):
        super().__init__()
        self.seg = SegmentationLoss(lambda_cls, lambda_region, lambda_boundary,
                                    boundary_radius, num_classes, ignore_index,
                                    label_blur)
        self.lambda_aux = lambda_aux
        self.ignore_index = ignore_index

    def _segment(self, logits, target, valid, cut=None, teacher=None, teacher_weight=0.0):
        totals = {}
        for index, pred in enumerate(logits):
            parts = self.seg.compute(pred, target, valid, cut,
                                     include_boundary=index == len(logits) - 1,
                                     teacher=teacher, teacher_weight=teacher_weight)
            for key, value in parts.items():
                scale = 1.0 if key.startswith("boundary") else 1.0 / len(logits)
                totals[key] = totals.get(key, value * 0.0) + value * scale
        totals["seg"] = totals["cls"] + totals["region"] + totals["boundary"]
        return totals

    @staticmethod
    def _guide(gdt, valid):
        preds, labels = gdt
        if not preds or len(preds) != len(labels):
            raise RuntimeError("GDT predictions and labels do not match")
        loss = valid.new_zeros(())
        for pred, label in zip(preds, labels):
            if pred.shape[1] != 1 or label.shape[1] != 1:
                raise ValueError("GDT predictions and labels must have one boundary channel")
            if pred.shape[-2:] != label.shape[-2:]:
                pred = F.interpolate(pred, label.shape[-2:], mode="bilinear", align_corners=False)
            mask = F.interpolate(valid, label.shape[-2:], mode="nearest")
            mask = erode_valid(mask, radius=2)
            pred = pred.masked_fill(mask == 0, 0)
            label = label.detach().masked_fill(mask == 0, 0)
            pixel_loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
            loss = loss + masked_mean(pixel_loss, mask)
        return loss / len(preds)

    def forward(self, out: Output, batch, teacher=None, teacher_weight=0.0):
        if not out.logits:
            raise ValueError("output must contain at least one segmentation logit")
        target = batch["mask"]
        valid = batch.get("valid", torch.ones_like(target[:, None], dtype=torch.float32))
        valid = valid * (target[:, None] != self.ignore_index)
        cut = batch.get("cut", torch.zeros_like(valid))
        size = target.shape[0]
        count = out.logits[-1].shape[0]
        if count != size:
            raise ValueError("output batch must match target batch")
        parts = self._segment(out.logits, target, valid, cut=cut,
                              teacher=teacher, teacher_weight=teacher_weight)
        aux_raw = self._guide(out.gdt, valid) if out.gdt is not None and self.lambda_aux else parts["seg"] * 0.0
        aux = self.lambda_aux * aux_raw
        loss = parts["seg"] + aux
        parts.update(loss=loss, aux_raw=aux_raw, aux=aux)
        return parts, loss
