import torch
import torch.nn.functional as F


def _active(target, valid, ignore_index):
    if target.ndim != 3:
        raise ValueError("target must have shape B,H,W")
    active = target != ignore_index
    if valid is not None:
        if valid.shape != target[:, None].shape:
            raise ValueError("valid must have shape B,1,H,W")
        active = active & (valid[:, 0] > 0.5)
    return active


def confusion_matrix(prediction, target, valid=None, num_classes=4, ignore_index=255):
    """Rows are true classes; columns are predicted classes. Padding is excluded."""
    if target.ndim == 2:
        target = target[None]
        if prediction.ndim == 2:
            prediction = prediction[None]
        elif prediction.ndim == 3:
            prediction = prediction[None]
        if valid is not None:
            valid = valid.reshape(1, 1, *target.shape[-2:])
    if prediction.ndim == 4:
        if prediction.shape[1] != num_classes:
            raise ValueError("logit channel count differs from num_classes")
        prediction = prediction.argmax(1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    active = _active(target, valid, ignore_index)
    truth, pred = target[active].long(), prediction[active].long()
    if ((truth < 0) | (truth >= num_classes) | (pred < 0) | (pred >= num_classes)).any():
        raise ValueError("invalid class index in active metric pixels")
    counts = torch.bincount(truth * num_classes + pred, minlength=num_classes ** 2)
    return counts.reshape(num_classes, num_classes)


def scores(confusion):
    """Undefined class ratios are NaN; macro means exclude undefined classes."""
    matrix = confusion.to(torch.float64)
    true = matrix.sum(1)
    pred = matrix.sum(0)
    hit = matrix.diag()
    nan = torch.full_like(hit, float("nan"))
    union = true + pred - hit
    iou = torch.where(union > 0, hit / union.clamp_min(1), nan)
    dice = torch.where(true + pred > 0, 2 * hit / (true + pred).clamp_min(1), nan)
    recall = torch.where(true > 0, hit / true.clamp_min(1), nan)
    return {"per_class_iou": iou, "per_class_dice": dice, "per_class_recall": recall,
            "miou": iou.nanmean(), "mdice": dice.nanmean(), "mean_recall": recall.nanmean(),
            "pixel_accuracy": hit.sum() / matrix.sum() if matrix.sum() > 0 else matrix.new_tensor(float("nan"))}


def iou_logits(logits, target, valid=None, ignore_index=255):
    return scores(confusion_matrix(logits, target, valid, logits.shape[1], ignore_index))["miou"]


def dice(logits, target, valid=None, ignore_index=255):
    return scores(confusion_matrix(logits, target, valid, logits.shape[1], ignore_index))["mdice"]


def brier(logits, target, valid=None, ignore_index=255):
    active = _active(target, valid, ignore_index)
    truth = F.one_hot(target.masked_fill(~active, 0).long(), logits.shape[1]).permute(0, 3, 1, 2)
    error = (logits.softmax(1) - truth).square().sum(1)
    return (error * active).sum() / active.sum().clamp_min(1)


def ece(logits, target, valid=None, bins=10, ignore_index=255):
    if bins <= 0:
        raise ValueError("bins must be positive")
    active = _active(target, valid, ignore_index)
    confidence, prediction = logits.softmax(1).max(1)
    confidence = confidence[active]
    correct = (prediction[active] == target[active]).to(logits.dtype)
    result = logits.sum() * 0.0
    if confidence.numel() == 0:
        return result
    for index in range(bins):
        in_bin = (confidence >= index / bins) & (confidence <= (index + 1) / bins if index == bins - 1 else confidence < (index + 1) / bins)
        if in_bin.any():
            result = result + in_bin.float().mean() * (confidence[in_bin].mean() - correct[in_bin].mean()).abs()
    return result
