import torch
from tqdm import tqdm

from .data.io import read_sample
from .predict import predict_logits


def confusion_matrix(prediction, target, num_classes=4, ignore_index=255):
    """Rows are true classes; columns are predicted classes. Ignored pixels are left out."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    active = target != ignore_index
    truth, pred = target[active].long(), prediction[active].long()
    if (
        (truth < 0) | (truth >= num_classes) | (pred < 0) | (pred >= num_classes)
    ).any():
        raise ValueError("invalid class index in active metric pixels")
    counts = torch.bincount(truth * num_classes + pred, minlength=num_classes**2)
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
    return {
        "miou": float(iou.nanmean()),
        **{f"class_{i}_iou": float(v) for i, v in enumerate(iou)},
        "pixel_accuracy": float(hit.sum() / matrix.sum())
        if matrix.sum()
        else float("nan"),
    }


def evaluate(model, pairs, tiles=(1,)):
    """Score image/mask pairs; image/mask/label triples score each image once against both."""
    if not pairs:
        raise ValueError("Evaluation needs at least one pair")
    classes = model.num_classes
    names = ["mask", "label"][: len(pairs[0]) - 1]
    matrices = [torch.zeros(classes, classes, dtype=torch.int64) for _ in names]
    model.eval()
    for paths in tqdm(pairs, desc="Evaluating", leave=False):
        image, *targets = read_sample(paths, classes)
        pred = torch.from_numpy(predict_logits(model, image, tiles=tiles).argmax(0))
        for matrix, target in zip(matrices, targets):
            matrix += confusion_matrix(pred, torch.from_numpy(target), classes)
    if len(names) == 1:
        return scores(matrices[0])
    return {
        f"{name}_{key}": value
        for name, matrix in zip(names, matrices)
        for key, value in scores(matrix).items()
    }
