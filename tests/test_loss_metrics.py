import torch
from src.model import Output
from src.train.loss import TrainLoss
from src.train.metrics import confusion_matrix, scores


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
    s = scores(matrix)
    assert s["class_0_iou"] == 4 / 7 and s["class_1_iou"] == 0.5
    assert s["miou"] == (4 / 7 + 0.5) / 2 and s["pixel_accuracy"] == 0.7
    assert confusion_matrix(torch.tensor([0, 1]), torch.tensor([0, 255]), 2).sum() == 1
