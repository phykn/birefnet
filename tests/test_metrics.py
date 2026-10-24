import pytest
import torch

from src.train.metrics import confusion_matrix, scores


def test_confusion_uses_all_classes_and_excludes_ignored_pixels():
    target = torch.tensor([[0, 1, 2], [3, 255, 2]])
    pred = torch.tensor([[0, 2, 2], [3, 0, 0]])
    matrix = confusion_matrix(pred, target)
    expected = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
    assert torch.equal(matrix, expected)
    result = scores(matrix)
    assert result["miou"] == pytest.approx((1 / 2 + 0 + 1 / 3 + 1) / 4)
    assert result["pixel_accuracy"] == pytest.approx(0.6)
    assert torch.equal(result["per_class_recall"], torch.tensor([1., 0., .5, 1.], dtype=torch.float64))


def test_absent_classes_are_left_out_of_the_mean():
    result = scores(confusion_matrix(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)))
    assert result["miou"] == 1
    assert result["per_class_iou"][1:].isnan().all()


def test_all_ignored_metrics_are_undefined():
    result = scores(confusion_matrix(torch.zeros(2, 2, dtype=torch.long), torch.full((2, 2), 255)))
    assert result["miou"].isnan()
    assert result["pixel_accuracy"].isnan()


def test_rejects_mismatched_shapes_and_unknown_classes():
    with pytest.raises(ValueError, match="shapes"):
        confusion_matrix(torch.zeros(2, 2), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="class index"):
        confusion_matrix(torch.full((2, 2), 4), torch.zeros(2, 2, dtype=torch.long))
