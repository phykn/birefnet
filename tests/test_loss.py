import pytest
import torch
import torch.nn.functional as F

from src.model import Output
from tests.helpers import make_loss


def test_ce_matches_torch_and_dice_perfect():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * 30
    parts, _ = make_loss(gce_q=0)(Output([logits]), {"mask": target})
    torch.testing.assert_close(parts["ce"], F.cross_entropy(logits, target))
    assert parts["dice"] < 1e-5
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    target[:, 0] = 255
    parts, _ = make_loss(gce_q=0)(Output([pred]), {"mask": target})
    expected = F.cross_entropy(pred, target, ignore_index=255)
    torch.testing.assert_close(parts["ce"], expected)


def test_gce_pulls_less_on_confidently_wrong_mask_pixels():
    target = torch.tensor([[[0, 0]]])
    pred = torch.tensor([[[[0.0, -6.0]], [[0.0, 6.0]]]]).requires_grad_()
    grads = []
    for q in (0, 0.7):
        pred.grad = None
        loss = make_loss(classes=2, dice=0, gce_q=q)
        loss(Output([pred]), {"mask": target})[1].backward()
        grads.append(pred.grad[0, :, 0].abs().sum(0))
    ce, gce = grads
    assert gce[1] / gce[0] < ce[1] / ce[0]


def test_loss_resizes_indices_and_backpropagates():
    logits = torch.randn(2, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    target[:, :2] = 255
    _, loss = make_loss()(Output([logits]), {"mask": target})
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_deep_supervision_boundary_is_not_diluted():
    loss = make_loss(boundary=1)
    preds = [torch.zeros(1, 4, size, size) for size in (8, 16, 32)]
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 2
    deep = loss(Output(preds), {"mask": target})[0]["boundary"]
    last = loss(Output(preds[-1:]), {"mask": target})[0]["boundary"]
    assert deep > 0
    torch.testing.assert_close(deep, last)


def test_multiscale_and_gdt_backward():
    preds = [torch.randn(2, 4, size, size, requires_grad=True) for size in (4, 8)]
    edge = torch.randn(2, 1, 8, 8, requires_grad=True)
    label = torch.rand(2, 1, 8, 8, requires_grad=True)
    batch = {"mask": torch.randint(0, 4, (2, 8, 8))}
    parts, loss = make_loss(aux=1.0)(Output(preds, ([edge], [label])), batch)
    assert torch.isfinite(loss) and parts["aux"] > 0
    loss.backward()
    assert all(pred.grad is not None for pred in preds)
    assert edge.grad is not None and label.grad is None
    parts, _ = make_loss()(Output(preds), batch)
    assert parts["aux"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_nonfinite_ignored_logits_do_not_affect_loss_or_gradients(bad, all_ignored):
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[..., 4:] = 255
    if all_ignored:
        target.fill_(255)
    inactive = target[:, None] == 255
    clean = torch.zeros(1, 4, 8, 8, requires_grad=True)
    dirty = clean.detach().masked_fill(inactive, bad).requires_grad_()
    loss = make_loss(boundary=1)
    _, expected = loss(Output([clean]), {"mask": target})
    _, actual = loss(Output([dirty]), {"mask": target})
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(dirty.grad, clean.grad)
    assert not dirty.grad.masked_select(inactive).any()
    if all_ignored:
        assert expected == 0


def test_gdt_ignores_nonfinite_values_outside_valid_region():
    valid = torch.zeros(1, 1, 12, 12)
    valid[..., 1:11, 1:11] = 1
    pred = torch.zeros_like(valid, requires_grad=True)
    label = torch.ones_like(valid)
    dirty_pred = pred.detach().clone()
    dirty_label = label.clone()
    dirty_pred[..., :3, :] = float("nan")
    dirty_label[..., :3, :] = float("nan")
    dirty_pred.requires_grad_()
    loss = make_loss(aux=1)
    expected = loss.compute_guidance_loss(([pred], [label]), valid)
    actual = loss.compute_guidance_loss(([dirty_pred], [dirty_label]), valid)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(dirty_pred.grad, pred.grad)


def test_label_blur_changes_loss_only_near_class_edges():
    pred = torch.randn(1, 4, 20, 20)
    target = torch.ones(1, 20, 20, dtype=torch.long)
    target[..., :5] = 255
    plain, blurred = make_loss(), make_loss(label_blur=2.0)
    flat = {"mask": target}
    out = Output([pred])
    torch.testing.assert_close(blurred(out, flat)[0]["ce"], plain(out, flat)[0]["ce"])
    edged = {"mask": target.clone()}
    edged["mask"][..., 12:] = 3
    assert blurred(out, edged)[0]["ce"] != plain(out, edged)[0]["ce"]


def test_teacher_pulls_targets_toward_its_prediction():
    batch = {"mask": torch.zeros(1, 8, 8, dtype=torch.long)}
    out = Output([torch.zeros(1, 2, 4, 4)])
    out.logits[0][:, 1] = 5.0
    teacher = torch.zeros(1, 2, 8, 8)
    teacher[:, 1] = 1.0
    loss = make_loss(classes=2)
    plain = loss(out, batch)[0]["ce"]
    taught = loss(out, batch, teacher=teacher, teacher_weight=0.3)[0]["ce"]
    unchanged = loss(out, batch, teacher=teacher, teacher_weight=0.0)[0]["ce"]
    assert taught < plain
    torch.testing.assert_close(unchanged, plain)


def test_band_excludes_ignore_and_crop_edges():
    pred = torch.zeros(1, 4, 16, 16, requires_grad=True)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, 8:] = 255
    loss = make_loss(boundary=1)
    assert loss(Output([pred]), {"mask": target})[0]["boundary"] == 0
    uniform = torch.full_like(target, 2)
    assert loss(Output([pred]), {"mask": uniform})[0]["boundary"] == 0
    target[:, :, 8:] = 1
    assert loss(Output([pred]), {"mask": target})[0]["boundary"] > 0
    cut = torch.zeros(1, 1, 16, 16)
    cut[:, :, :, 7:9] = 1
    assert loss(Output([pred]), {"mask": target, "cut": cut})[0]["boundary"] == 0


def label_ce(logits, mask, label, **loss):
    batch = {"mask": mask, "label": label}
    return make_loss(**loss)(Output([logits]), batch)[0]["label_ce"]


def test_label_term_does_not_scale_with_unpainted_pixels():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10, 50)
    label = torch.full((1, 10, 50), 255)
    label[0, 5, :10] = 2
    mask = torch.zeros(1, 10, 50, dtype=torch.long)
    wide = torch.cat([logits, torch.randn(1, 4, 10, 50)], dim=3)
    wide_label = torch.cat([label, torch.full_like(label, 255)], dim=2)
    wide_mask = torch.zeros(1, 10, 100, dtype=torch.long)
    small = label_ce(logits, mask, label)
    large = label_ce(wide, wide_mask, wide_label)
    torch.testing.assert_close(small, large)


def test_one_wrong_label_pixel_is_bounded():
    logits = torch.zeros(1, 4, 10, 100)
    logits[0, 0, 3, 3] = 5.0
    label = torch.full((1, 10, 100), 255)
    label[0, 3, 3] = 1
    pixel = -torch.log_softmax(logits[0, :, 3, 3], 0)[1]
    actual = label_ce(logits, torch.zeros_like(label), label, label_smooth=0)
    torch.testing.assert_close(actual, pixel / 5)


def test_mask_is_ignored_on_label_pixels():
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 16, 16)
    mask = torch.randint(0, 4, (2, 16, 16))
    label = torch.full_like(mask, 255)
    label[:, 4:8, 4:8] = 3
    loss = make_loss(mask_weight=0.3)
    base_parts, base = loss(Output([logits]), {"mask": mask, "label": label})
    changed = mask.clone()
    changed[:, 4:8, 4:8] = 0
    _, other = loss(Output([logits]), {"mask": changed, "label": label})
    torch.testing.assert_close(base, other)
    changed[:, 4:8, 4:8] = 255
    parts, _ = loss(Output([logits]), {"mask": changed, "label": label})
    torch.testing.assert_close(parts["label_ce"], base_parts["label_ce"])


def test_label_smoothing_keeps_some_loss():
    logits = torch.full((1, 5, 8, 8), -20.0)
    logits[:, 2] = 20.0
    label = torch.full((1, 8, 8), 255)
    label[0, :4] = 2
    assert label_ce(logits, torch.full_like(label, 2), label, classes=5) > 0
