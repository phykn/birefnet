import pytest
import torch
import torch.nn as nn

from src.train.ema import EMA


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.frozen = nn.Parameter(torch.tensor(2.0), requires_grad=False)


def test_ema_averages_trainable_weights_only():
    model = Model()
    ema = EMA(model, decay=0.5)
    assert set(ema.params) == {"weight"}
    with torch.no_grad():
        model.weight.fill_(3)
    ema.update()
    assert ema.params["weight"] == 2
    with ema.applied():
        assert model.weight == 2
    assert model.weight == 3 and model.frozen == 2


def test_ema_resume_matches_uninterrupted_updates():
    model = Model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3)
    ema.update()
    restored = EMA(model, decay=ema.decay)
    restored.load_state_dict(ema.state_dict())
    with torch.no_grad():
        model.weight.fill_(5)
    ema.update()
    restored.update()
    torch.testing.assert_close(restored.params["weight"], ema.params["weight"])


def test_ema_rejects_mismatched_state():
    ema = EMA(Model(), decay=0.5)
    with pytest.raises(RuntimeError):
        ema.load_state_dict({"weight": torch.zeros(2)})
