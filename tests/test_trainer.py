import pytest
import torch

from src.model import Output
from tests.helpers import trainer


def test_resume(tmp_path):
    t = trainer(tmp_path)
    t.step()
    t.best_miou = 0.5
    t.save()
    expected = {k: v.clone() for k, v in t.model.state_dict().items()}
    restored = trainer(tmp_path)
    restored.resume(tmp_path / "weights/last.train.pth")
    assert restored.global_step == 1 and restored.best_miou == 0.5
    assert restored.scheduler.state_dict() == t.scheduler.state_dict()
    assert restored.optimizer.param_groups[0]["lr"] == t.optimizer.param_groups[0]["lr"]
    for k, v in restored.model.state_dict().items():
        torch.testing.assert_close(v, expected[k])
    for k, v in restored.ema.params.items():
        torch.testing.assert_close(v, t.ema.params[k])
    restored.step()
    assert restored.global_step == 2
    state = torch.load(tmp_path / "weights/last.pth")
    for k, v in state["model"].items():
        torch.testing.assert_close(v, t.ema.params[k])


def test_nonfinite(tmp_path):
    t = trainer(tmp_path)
    t.train_loader[0]["image"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="step 1"):
        t.step()


class HalvingScaler:
    value = 2.0

    def scale(self, x):
        return x

    def unscale_(self, opt):
        pass

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.value

    def step(self, opt):
        pass

    def update(self):
        self.value /= 2


def test_skipped_step(tmp_path):
    t = trainer(tmp_path)
    t.scaler = HalvingScaler()
    before = {k: v.clone() for k, v in t.ema.params.items()}
    schedule = t.scheduler.last_epoch
    _, updated = t.step()
    assert not updated and t.global_step == 0 and t.scheduler.last_epoch == schedule
    for k, v in t.ema.params.items():
        torch.testing.assert_close(v, before[k])


def test_skipped_limit(tmp_path, monkeypatch):
    t = trainer(tmp_path)
    monkeypatch.setattr(t, "step", lambda: ({"loss": 1.0}, False))
    t.cfg.train.steps = 1
    with pytest.raises(FloatingPointError, match="16 consecutive"):
        t.train()


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Conv2d(3, 4, 1)
        self.encodes = 0
        self.calls = 0

    def encode(self, x):
        self.encodes += 1
        return x

    def forward(self, x, encoded=None):
        self.calls += 1
        return Output([self.decoder(self.encode(x) if encoded is None else encoded)])


def test_teacher_ramp_and_shared_features(tmp_path):
    m = CountingModel()
    t = trainer(tmp_path, m)
    t.cfg.loss.teacher = 0.3
    t.cfg.loss.teacher_start = 10
    shares = []
    for step in (0, 10, 15, 20, 100):
        t.global_step = step
        shares.append(t.teacher_share())
    assert shares == pytest.approx([0, 0, 0.15, 0.3, 0.3])
    with torch.no_grad():
        t.ema.params["decoder.weight"].zero_()
        t.ema.params["decoder.bias"].copy_(torch.tensor([0.0, 9.0, 0.0, 0.0]))
    x = torch.randn(1, 3, 16, 16)
    p = t.predict_teacher(x, m.encode(x))
    assert (p.argmax(1) == 1).all()
    assert m.training and m.decoder.bias.abs().max() < 9
    m.calls = m.encodes = 0
    seen = []
    criterion = t.criterion

    def capture(out, batch, **kw):
        seen.append(kw)
        return criterion(out, batch, **kw)

    t.criterion = capture
    t.global_step = 20
    t.step()
    assert m.calls == 2 and m.encodes == 1
    assert seen[0]["teacher_weight"] == pytest.approx(0.3)
    assert not seen[0]["teacher"].requires_grad
    m.calls = m.encodes = 0
    t.cfg.loss.teacher = 0
    t.step()
    assert m.calls == 1 and m.encodes == 1
