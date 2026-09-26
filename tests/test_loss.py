import pytest
import torch
import torch.nn.functional as F

from src.model.output import Output
from src.train.loss import DiceLoss, SegmentationLoss
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


def test_teacher_softmax_only_downweights_confident_conflicting_labels():
    teacher = torch.tensor([[[[0., 40.]], [[40., 0.]], [[0., 0.]], [[0., 0.]]]])
    target = torch.zeros(1, 1, 2, dtype=torch.long)
    weight, conf, prob = TrainLoss()._weigh(teacher, target, 1.)
    assert torch.allclose(weight, torch.tensor([[[[.25, 1.]]]]))
    assert torch.allclose(prob.sum(1), torch.ones_like(target, dtype=torch.float32))
    assert (conf > .99).all()


def test_two_view_multiscale_and_binary_gdt_cpu_backward():
    preds = [torch.randn(4, 4, size, size, requires_grad=True) for size in (4, 8)]
    edge = torch.randn(4, 1, 8, 8, requires_grad=True)
    label = torch.rand(4, 1, 8, 8, requires_grad=True)
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
