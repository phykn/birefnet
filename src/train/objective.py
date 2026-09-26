import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.output import Output
from .loss import SegmentationLoss, erode_valid, masked_mean


class TrainLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_region=1.0, lambda_boundary=0.5,
                 region_loss="dice", boundary_radius=3, lambda_aux=1.0,
                 teacher_confidence=0.95, min_gt_weight=0.25, lambda_teacher=0.1,
                 num_classes=4, ignore_index=255):
        super().__init__()
        if not 0.0 < teacher_confidence < 1.0:
            raise ValueError("teacher confidence must be in (0, 1)")
        if not 0.0 <= min_gt_weight <= 1.0:
            raise ValueError("minimum GT weight must be in [0, 1]")
        self.seg = SegmentationLoss(lambda_cls, lambda_region, lambda_boundary,
                                    region_loss, boundary_radius, num_classes, ignore_index)
        self.lambda_aux = lambda_aux
        self.teacher_confidence = teacher_confidence
        self.min_gt_weight = min_gt_weight
        self.lambda_teacher = lambda_teacher
        self.ignore_index = ignore_index

    def _segment(self, logits, target, valid, weight=None, cut=None):
        if not logits:
            raise ValueError("output must contain at least one segmentation logit")
        totals = {}
        for index, pred in enumerate(logits):
            parts = self.seg.compute(pred, target, valid, weight, cut,
                                     include_boundary=index == len(logits) - 1)
            for key, value in parts.items():
                scale = 1.0 if key.startswith("boundary") else 1.0 / len(logits)
                totals[key] = totals.get(key, value * 0.0) + value * scale
        totals["seg"] = totals["cls"] + totals["region"] + totals["boundary"]
        return totals

    def _weigh(self, teacher, target, scale):
        if teacher.shape[-2:] != target.shape[-2:]:
            teacher = F.interpolate(teacher, target.shape[-2:], mode="bilinear", align_corners=False)
        prob = teacher.detach().softmax(1)
        confidence, prediction = prob.max(1, keepdim=True)
        confidence = ((confidence - self.teacher_confidence) / (1 - self.teacher_confidence)).clamp(0, 1)
        conflict = confidence * (prediction != target[:, None]) * scale
        weight = 1 - (1 - self.min_gt_weight) * conflict
        return weight, confidence, prob

    @staticmethod
    def _distill(student, prob, conf, valid):
        if student.shape[-2:] != prob.shape[-2:]:
            prob = F.interpolate(prob, student.shape[-2:], mode="bilinear", align_corners=False)
            conf = F.interpolate(conf, student.shape[-2:], mode="area")
        if valid.shape[-2:] != student.shape[-2:]:
            valid = F.interpolate(valid, student.shape[-2:], mode="nearest")
        loss = -(prob * student.log_softmax(1)).sum(1, keepdim=True)
        return masked_mean(loss * conf, valid)

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

    def forward(self, out: Output, batch, teacher_logit=None, teacher_scale=0.0):
        target = batch["mask"]
        valid = batch.get("valid", torch.ones_like(target[:, None], dtype=torch.float32))
        valid = valid * (target[:, None] != self.ignore_index)
        cut = batch.get("cut", torch.zeros_like(valid))
        size = target.shape[0]
        count = out.logits[-1].shape[0]
        if count not in {size, 2 * size}:
            raise ValueError("output batch must contain one or two views per target")
        two_views = count == 2 * size
        zero = out.logits[-1].float().sum() * 0.0
        weight = torch.ones_like(valid)
        teacher_raw = zero
        if teacher_logit is not None and teacher_scale > 0:
            weight, conf, prob = self._weigh(teacher_logit, target, teacher_scale)
            student = out.logits[-1][size:] if two_views else out.logits[-1]
            teacher_raw = self._distill(student, prob, conf, valid)
        targets, valids, cuts, weights = target, valid, cut, weight
        if two_views:
            targets, valids, cuts, weights = [torch.cat([item, item], 0) for item in (target, valid, cut, weight)]
        parts = self._segment(out.logits, targets, valids, weights, cuts)
        aux_raw = self._guide(out.gdt, valids) if out.gdt is not None and self.lambda_aux else zero
        aux = self.lambda_aux * aux_raw
        teacher = self.lambda_teacher * teacher_scale * teacher_raw
        loss = parts["seg"] + aux + teacher
        parts.update(loss=loss, gt_weight=masked_mean(weight, valid), teacher_raw=teacher_raw,
                     teacher=teacher, aux_raw=aux_raw, aux=aux)
        return parts, loss
