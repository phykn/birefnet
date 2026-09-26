import torch
import torch.nn as nn

from src.train.ema import EMA


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.frozen = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.register_buffer("running", torch.tensor(1.0))
        self.register_buffer("count", torch.tensor(0))


def test_ema_averages_trainable_weights_and_float_buffers():
    model = Model()
    ema = EMA(model, decay=0.5)
    assert set(ema.params) == {"weight", "running", "count"}
    snapshot = ema.state_dict()
    with torch.no_grad():
        model.weight.fill_(3)
        model.running.fill_(5)
        model.count.fill_(7)
    ema.update(model)
    assert ema.params["weight"] == 2
    assert ema.params["running"] == 3
    assert ema.params["count"] == 7
    assert snapshot["weight"] == 1
    assert model.weight == 3 and model.frozen == 2


def test_ema_resume_matches_uninterrupted_updates():
    model = Model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3)
    ema.update(model)
    restored = EMA(model, decay=ema.decay)
    restored.load_state_dict(ema.state_dict())
    with torch.no_grad():
        model.weight.fill_(5)
    ema.update(model)
    restored.update(model)
    for name, value in ema.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_ema_checkpoint_excludes_nonpersistent_buffers():
    model = Model()
    model.register_buffer("cache", torch.tensor(3.0), persistent=False)
    ema = EMA(model)
    state = model.state_dict()
    state.update(ema.state_dict())
    assert "cache" not in state
    model.load_state_dict(state, strict=True)
