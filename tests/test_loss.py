import pytest
import torch
import torch.nn.functional as F

from src.model.output import Output
from src.train.loss import DiceLoss, SegmentationLoss, soften
from src.train.objective import TrainLoss


def test_ce_matches_torch_and_dice_perfect():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * 30
    loss_fn = SegmentationLoss(lambda_region=0, lambda_boundary=0)
    parts = loss_fn.compute(logits, target)
    assert torch.allclose(parts["cls"], F.cross_entropy(logits, target))
    assert parts["region_raw"] < 1e-5


def test_loss_resizes_indices_and_backpropagates_cpu():
    logits = torch.randn(2, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    target[:, :2] = 255
    loss = SegmentationLoss()(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_padding_ignore_do_not_affect_loss_or_gradient():
    target = torch.randint(0, 4, (1, 8, 8))
    target[:, :, 6:] = 255
    valid = torch.ones(1, 1, 8, 8)
    valid[:, :, :2] = 0
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    reference = SegmentationLoss()(logits, target, valid)
    changed = logits.detach().clone()
    changed[:, :, :, 6:] = 100
    changed[:, :, :2] = -100
    assert torch.allclose(SegmentationLoss()(changed, target, valid), reference)
    reference.backward()
    assert not logits.grad[:, :, :, 6:].any()
    assert not logits.grad[:, :, :2].any()


def test_all_ignored_loss_is_differentiable_zero():
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    loss = SegmentationLoss()(logits, torch.full((1, 8, 8), 255, dtype=torch.long))
    assert loss == 0
    loss.backward()
    assert not logits.grad.any()


def test_invalid_label_and_binary_channel_are_rejected():
    with pytest.raises(ValueError, match="channels"):
        SegmentationLoss()(torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="invalid class"):
        SegmentationLoss()(torch.zeros(1, 4, 2, 2), torch.full((1, 2, 2), 4, dtype=torch.long))


def test_classwise_dice_counts_each_class():
    target = torch.eye(4).reshape(1, 4, 1, 4)
    assert DiceLoss()(target, target) == 0
    assert DiceLoss()(target.flip(1), target) > .99


def test_deep_supervision_boundary_is_not_diluted():
    loss_fn = TrainLoss(lambda_boundary=1)
    preds = [torch.zeros(1, 4, size, size) for size in (8, 16, 32)]
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 2
    valid = torch.ones(1, 1, 32, 32)
    parts = loss_fn._segment(preds, target, valid)
    assert torch.allclose(parts["boundary"], loss_fn.seg.compute(preds[-1], target, valid)["boundary"])


def test_single_view_multiscale_and_binary_gdt_cpu_backward():
    preds = [torch.randn(2, 4, size, size, requires_grad=True) for size in (4, 8)]
    edge = torch.randn(2, 1, 8, 8, requires_grad=True)
    label = torch.rand(2, 1, 8, 8, requires_grad=True)
    batch = {"mask": torch.randint(0, 4, (2, 8, 8)), "valid": torch.ones(2, 1, 8, 8)}
    parts, loss = TrainLoss()(Output(preds, ([edge], [label])), batch)
    assert torch.isfinite(loss) and parts["aux"] > 0
    loss.backward()
    assert all(pred.grad is not None for pred in preds)
    assert edge.grad is not None and label.grad is None


def test_single_view_without_gdt_still_deep_supervises():
    preds = [torch.randn(1, 4, size, size, requires_grad=True) for size in (4, 8)]
    parts, loss = TrainLoss()(Output(preds), {"mask": torch.randint(0, 4, (1, 8, 8))})
    loss.backward()
    assert parts["aux"] == 0
    assert all(pred.grad is not None for pred in preds)


def test_empty_logits_raise_value_error():
    with pytest.raises(ValueError, match="at least one segmentation logit"):
        TrainLoss()(Output([]), {"mask": torch.zeros(1, 8, 8, dtype=torch.long)})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_nonfinite_ignored_logits_do_not_affect_loss_or_gradients(bad, all_ignored):
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[..., 4:] = 255
    valid = torch.ones(1, 1, 8, 8)
    valid[..., :2] = 0
    if all_ignored:
        target.fill_(255)
    inactive = (target[:, None] == 255) | (valid == 0)
    clean = torch.zeros(1, 4, 8, 8, requires_grad=True)
    dirty = clean.detach().masked_fill(inactive, bad).requires_grad_()
    criterion = SegmentationLoss()

    expected = criterion(clean, target, valid)
    actual = criterion(dirty, target, valid)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(dirty.grad, clean.grad)
    assert not dirty.grad.masked_select(inactive).any()

    parts, total = TrainLoss()(Output([dirty]), {"mask": target, "valid": valid})
    torch.testing.assert_close(total, expected)
    assert parts["aux"] == 0


def test_gdt_ignores_nonfinite_predictions_and_labels_outside_valid_region():
    valid = torch.zeros(1, 1, 12, 12)
    valid[..., 1:11, 1:11] = 1
    pred = torch.zeros_like(valid, requires_grad=True)
    label = torch.ones_like(valid)
    dirty_pred = pred.detach().clone()
    dirty_label = label.clone()
    dirty_pred[..., :3, :] = float("nan")
    dirty_label[..., :3, :] = float("nan")
    dirty_pred.requires_grad_()

    expected = TrainLoss._guide(([pred], [label]), valid)
    actual = TrainLoss._guide(([dirty_pred], [dirty_label]), valid)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(dirty_pred.grad, pred.grad)


def test_soft_labels_off_match_plain_cross_entropy():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    parts = SegmentationLoss(num_classes=4).compute(pred, target, include_boundary=False)
    torch.testing.assert_close(parts["cls_raw"], F.cross_entropy(pred, target))


def test_label_blur_mixes_classes_only_near_edges():
    onehot = torch.zeros(1, 2, 20, 20)
    onehot[:, 0, :, :10] = 1
    onehot[:, 1, :, 10:] = 1
    soft = soften(onehot, torch.ones(1, 1, 20, 20), blur=1.0)
    torch.testing.assert_close(soft.sum(1), torch.ones(1, 20, 20))
    assert torch.all(soft[0, 0, :, :6] > 0.999) and torch.all(soft[0, 1, :, 14:] > 0.999)
    assert 0.6 < soft[0, 0, 5, 9] < 0.9 and 0.6 < soft[0, 1, 5, 10] < 0.9


def test_ignored_pixels_do_not_leak_into_soft_labels():
    onehot = torch.zeros(1, 2, 20, 20)
    onehot[:, 1] = 1
    valid = torch.ones(1, 1, 20, 20)
    valid[..., :10] = 0
    soft = soften(onehot, valid, blur=2.0)
    torch.testing.assert_close(soft[0, 1, :, 10:], torch.ones(20, 10))


def test_teacher_pulls_targets_toward_its_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    pred = torch.zeros(1, 2, 4, 4)
    pred[:, 1] = 5.0
    loss = SegmentationLoss(num_classes=2)
    plain = loss.compute(pred, target, include_boundary=False)["cls_raw"]
    # The teacher agrees with the prediction, not the mask; a 1024-to-side-output
    # resize is done by area averaging.
    teacher = torch.zeros(1, 2, 8, 8)
    teacher[:, 1] = 1.0
    taught = loss.compute(pred, target, include_boundary=False,
                          teacher=teacher, teacher_weight=0.3)["cls_raw"]
    assert taught < plain
    unchanged = loss.compute(pred, target, include_boundary=False,
                             teacher=teacher, teacher_weight=0.0)["cls_raw"]
    torch.testing.assert_close(unchanged, plain)


@pytest.mark.parametrize("name, value", [("label_blur", -1.0), ("label_blur", 11.0)])
def test_label_blur_is_checked(name, value):
    with pytest.raises(ValueError, match=name):
        SegmentationLoss(**{name: value})
