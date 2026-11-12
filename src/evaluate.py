import torch
from tqdm import tqdm

from .data.io import read_sample
from .predict import predict


def count_confusion(prediction, target, classes):
    active = target != 255
    truth, pred = target[active].long(), prediction[active].long()
    counts = torch.bincount(truth * classes + pred, minlength=classes**2)
    return counts.reshape(classes, classes)


def compute_scores(confusion):
    matrix = confusion.to(torch.float64)
    true = matrix.sum(1)
    pred = matrix.sum(0)
    hit = matrix.diag()
    union = true + pred - hit
    iou = torch.where(union > 0, hit / union.clamp_min(1), float("nan"))
    total = matrix.sum()
    return {
        "miou": float(iou.nanmean()),
        **{f"class_{i}_iou": float(v) for i, v in enumerate(iou)},
        "pixel_accuracy": float(hit.sum() / total) if total else float("nan"),
    }


def evaluate(model, pairs, tiles=(1,)):
    if not pairs:
        raise ValueError("Evaluation needs at least one pair")
    classes = model.num_classes
    names = ["mask", "label"][: len(pairs[0]) - 1]
    matrices = [torch.zeros(classes, classes, dtype=torch.int64) for _ in names]
    for paths in tqdm(pairs, desc="Evaluating", leave=False):
        image, *targets = read_sample(paths, classes)
        pred = torch.from_numpy(predict(model, image, tiles=tiles)["classes"])
        for matrix, target in zip(matrices, targets):
            matrix += count_confusion(pred, torch.from_numpy(target), classes)
    if len(names) == 1:
        return compute_scores(matrices[0])
    return {
        f"{name}_{key}": value
        for name, matrix in zip(names, matrices)
        for key, value in compute_scores(matrix).items()
    }
