import torch
from src.model import Output
from src.train.loss import TrainLoss
from src.evaluate import compute_scores, count_confusion


def test_ce_and_ignore():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    target[:, 0] = 255
    _, loss = TrainLoss(1, 0)(Output([logits]), {"mask": target})
    torch.testing.assert_close(
        loss, torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    )
    loss.backward()
    assert not logits.grad[:, :, 0].any()
    x = logits.detach().clone()
    x[:, :, 0] = float("nan")
    x.requires_grad_()
    _, ignored = TrainLoss()(Output([x]), {"mask": torch.full_like(target, 255)})
    assert ignored == 0
    ignored.backward()
    assert not x.grad.any()


def test_metrics():
    matrix = torch.tensor([[4, 1], [2, 3]])
    s = compute_scores(matrix)
    assert s["class_0_iou"] == 4 / 7 and s["class_1_iou"] == 0.5
    assert s["miou"] == (4 / 7 + 0.5) / 2 and s["pixel_accuracy"] == 0.7
    assert count_confusion(torch.tensor([0, 1]), torch.tensor([0, 255]), 2).sum() == 1


def fixed_batch():
    g = torch.Generator().manual_seed(0)
    logits = [torch.randn(2, 4, 16, 16, generator=g), torch.randn(2, 4, 32, 32, generator=g)]
    mask = torch.randint(0, 4, (2, 32, 32), generator=g)
    mask[0, :3] = 255
    return logits, mask


def test_loss_without_labels_is_unchanged():
    # Values computed before label support was added.
    logits, mask = fixed_batch()
    _, plain = TrainLoss()(Output(logits), {"mask": mask})
    _, extras = TrainLoss(lambda_boundary=0.5, label_blur=1.0)(Output(logits), {"mask": mask})
    assert abs(float(plain) - 2.4720215797424316) <= 1e-6
    assert abs(float(extras) - 3.3340988159179688) <= 1e-6


def label_loss(logits, mask, label, **kwargs):
    parts, loss = TrainLoss(**kwargs)(Output([logits]), {"mask": mask, "label": label})
    return parts, loss


def test_label_term_does_not_scale_with_unpainted_pixels():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10, 50)
    label = torch.full((1, 10, 50), 255)
    label[0, 5, :10] = 2
    mask = torch.zeros(1, 10, 50, dtype=torch.long)
    wide = torch.cat([logits, torch.randn(1, 4, 10, 50)], dim=3)
    wide_label = torch.cat([label, torch.full_like(label, 255)], dim=2)
    wide_mask = torch.zeros(1, 10, 100, dtype=torch.long)
    small = label_loss(logits, mask, label, label_smooth=0.05)[0]["label_ce"]
    large = label_loss(wide, wide_mask, wide_label, label_smooth=0.05)[0]["label_ce"]
    assert abs(float(small) - float(large)) <= 1e-6


def test_one_wrong_label_pixel_is_bounded():
    logits = torch.zeros(1, 4, 10, 100)
    logits[0, 0, 3, 3] = 5.0
    label = torch.full((1, 10, 100), 255)
    label[0, 3, 3] = 1
    pixel = -torch.log_softmax(logits[0, :, 3, 3], 0)[1]
    parts, _ = label_loss(logits, torch.zeros_like(label), label)
    assert abs(float(parts["label_ce"]) - float(pixel) / 5) <= 1e-6


def test_mask_is_ignored_on_label_pixels():
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 16, 16)
    mask = torch.randint(0, 4, (2, 16, 16))
    label = torch.full_like(mask, 255)
    label[:, 4:8, 4:8] = 3
    _, base = label_loss(logits, mask, label, mask_weight=0.3)
    changed = mask.clone()
    changed[:, 4:8, 4:8] = 0
    _, other = label_loss(logits, changed, label, mask_weight=0.3)
    assert abs(float(base) - float(other)) <= 1e-6
    changed[:, 4:8, 4:8] = 255
    parts, _ = label_loss(logits, changed, label, mask_weight=0.3)
    base_parts, _ = label_loss(logits, mask, label, mask_weight=0.3)
    assert abs(float(parts["label_ce"]) - float(base_parts["label_ce"])) <= 1e-6


def test_label_smoothing_keeps_some_loss():
    logits = torch.full((1, 5, 8, 8), -20.0)
    logits[:, 2] = 20.0
    label = torch.full((1, 8, 8), 255)
    label[0, :4] = 2
    parts, _ = label_loss(logits, torch.full_like(label, 2), label, label_smooth=0.05, num_classes=5)
    assert parts["label_ce"] > 0
