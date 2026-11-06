import torch
from tqdm import tqdm


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
    from ..data.io import read_pair
    from ..predict import predict_logits

    matrix = torch.zeros(model.num_classes, model.num_classes, dtype=torch.int64)
    if not pairs:
        raise ValueError("Evaluation needs at least one pair")
    model.eval()
    for image_path, mask_path in tqdm(pairs, desc="Evaluating", leave=False):
        image, mask = read_pair(image_path, mask_path, model.num_classes)
        pred = predict_logits(model, image, tiles=tiles).argmax(0)
        matrix += confusion_matrix(
            torch.from_numpy(pred), torch.from_numpy(mask), model.num_classes
        )
    return scores(matrix)
