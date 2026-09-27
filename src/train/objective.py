import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.output import Output
from .loss import SegmentationLoss, erode_valid, masked_mean


class TrainLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_region=1.0, lambda_boundary=0.5,
                 boundary_radius=3, lambda_aux=1.0,
                 num_classes=4, ignore_index=255):
        super().__init__()
        self.seg = SegmentationLoss(lambda_cls, lambda_region, lambda_boundary,
                                    boundary_radius, num_classes, ignore_index)
        self.lambda_aux = lambda_aux
        self.ignore_index = ignore_index

    def _segment(self, logits, target, valid, cut=None):
        totals = {}
        for index, pred in enumerate(logits):
            parts = self.seg.compute(pred, target, valid, cut,
                                     include_boundary=index == len(logits) - 1)
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
            pixel_loss = F.binary_cross_entropy_with_logits(pred, label.detach(), reduction="none")
            loss = loss + masked_mean(pixel_loss, mask)
        return loss / len(preds)

    def forward(self, out: Output, batch):
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
        zero = out.logits[-1].float().sum() * 0.0
        parts = self._segment(out.logits, target, valid, cut=cut)
        aux_raw = self._guide(out.gdt, valid) if out.gdt is not None and self.lambda_aux else zero
        aux = self.lambda_aux * aux_raw
        loss = parts["seg"] + aux
        parts.update(loss=loss, aux_raw=aux_raw, aux=aux)
        return parts, loss
