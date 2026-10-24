import torch


def confusion_matrix(prediction, target, num_classes=4, ignore_index=255):
    """Rows are true classes; columns are predicted classes. Ignored pixels are left out."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    active = target != ignore_index
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
