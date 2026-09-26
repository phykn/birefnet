import torch
import torch.nn.functional as F

from src.train.metrics import brier, confusion_matrix, dice, ece, iou_logits, scores


def test_confusion_uses_all_classes_and_excludes_ignore_and_padding():
    target = torch.tensor([[[0, 1, 2], [3, 255, 2]]])
    pred = torch.tensor([[[0, 2, 2], [3, 0, 0]]])
    valid = torch.tensor([[[[1., 1., 1.], [1., 1., 0.]]]])
    matrix = confusion_matrix(pred, target, valid)
    expected = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert torch.equal(matrix, expected)
    result = scores(matrix)
    assert torch.allclose(result["per_class_iou"], torch.tensor([1., 0., .5, 1.], dtype=torch.float64))
    assert result["miou"] == .625
    assert result["pixel_accuracy"] == .75
    assert torch.equal(result["per_class_recall"], torch.tensor([1., 0., 1., 1.], dtype=torch.float64))


def test_perfect_logits_and_absent_classes():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * 40
    assert iou_logits(logits, target) == 1
    assert dice(logits, target) == 1
    result = scores(confusion_matrix(torch.zeros(1, 2, 2, dtype=torch.long), torch.zeros(1, 2, 2, dtype=torch.long)))
    assert result["miou"] == 1
    assert result["per_class_iou"][1:].isnan().all()


def test_calibration_uses_softmax_and_ignores_padding():
    target = torch.tensor([[[0, 1, 255]]])
    logits = torch.tensor([[[[40., 0., 0.]], [[0., 40., 40.]], [[0., 0., 0.]], [[0., 0., 0.]]]])
    assert brier(logits, target) < 1e-6
    assert ece(logits, target) < 1e-6


def test_all_ignored_metrics_are_undefined():
    result = scores(confusion_matrix(torch.zeros(1, 2, 2, dtype=torch.long), torch.full((1, 2, 2), 255)))
    assert result["miou"].isnan()
    assert result["pixel_accuracy"].isnan()


def test_native_shape_confusion_matches_batched():
    target = torch.tensor([[0, 1], [2, 3]])
    assert torch.equal(confusion_matrix(target, target), confusion_matrix(target[None], target[None]))
