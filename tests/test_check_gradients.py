import pytest
import torch
from torch import nn

from scripts.check_gradients import audit
from src.model.output import Output
from src.train.objective import TrainLoss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 4, 1)
        self.frozen = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, inputs):
        self.views = inputs.shape[0]
        return Output([self.head(inputs) * self.frozen])


def batch():
    return {
        "image": torch.randn(1, 3, 8, 8),
        "mask": torch.randint(0, 4, (1, 8, 8)),
        "valid": torch.ones(1, 1, 8, 8),
    }


def test_audit_single_view_reaches_gradients_and_releases_hooks():
    model = TinyModel().eval()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    report = audit(model, batch(), TrainLoss())
    assert report["ok"]
    assert report["reached_tensors"] == 2
    assert report["frozen_tensors"] == 1
    assert model.views == 1
    assert not model.training
    assert report["missing"] == report["nonfinite"] == report["frozen_grad"] == []
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name])
        assert param.grad is None
        assert not param._post_accumulate_grad_hooks


def test_audit_identifies_disconnected_trainable_parameter():
    model = TinyModel()
    model.unused = nn.Parameter(torch.ones(()))
    report = audit(model, batch(), TrainLoss())
    assert not report["ok"]
    assert report["missing"] == ["unused"]


def test_audit_zero_gradients_are_not_failures():
    model = TinyModel()

    def criterion(out, batch):
        loss = out.logits[-1].sum() * 0
        return {}, loss

    report = audit(model, batch(), criterion)
    assert report["ok"]
    assert set(report["zero"]) == {"head.weight", "head.bias"}


def test_audit_identifies_nonfinite_gradients_with_finite_loss():
    model = TinyModel()
    handle = model.head.weight.register_hook(lambda grad: grad * float("nan"))
    try:
        report = audit(model, batch(), TrainLoss())
    finally:
        handle.remove()
    assert report["finite_loss"]
    assert not report["ok"]
    assert report["nonfinite"] == ["head.weight"]


def test_audit_cleans_up_after_forward_error():
    model = TinyModel().eval()

    def criterion(out, batch):
        raise ValueError("bad criterion")

    with pytest.raises(ValueError, match="bad criterion"):
        audit(model, batch(), criterion)
    assert not model.training
    assert all(not param._post_accumulate_grad_hooks for param in model.head.parameters())
