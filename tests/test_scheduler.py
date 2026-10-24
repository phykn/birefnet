import pytest
import torch

from src.train.schedule import cosine_schedule


def _make(lr=1.0, min_lr=0.0, warmup=5, steps=20):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=lr)
    return opt, cosine_schedule(opt, steps=steps, min_lr=min_lr, warmup=warmup)


def _lr(opt):
    return opt.param_groups[0]["lr"]


def _advance(opt, sched, count=1):
    for _ in range(count):
        opt.step()
        sched.step()


def test_warmup_starts_at_min_lr():
    opt, _ = _make(min_lr=0.01)
    assert _lr(opt) == pytest.approx(0.01)


def test_without_warmup_starts_at_lr():
    opt, sched = _make(min_lr=0.01, warmup=0)
    assert _lr(opt) == 1.0
    _advance(opt, sched)
    assert 0.01 < _lr(opt) < 1.0


def test_warmup_rises_monotonically_to_lr():
    opt, sched = _make(warmup=5)
    prev = _lr(opt)
    for _ in range(5):
        _advance(opt, sched)
        assert _lr(opt) >= prev
        prev = _lr(opt)
    assert _lr(opt) == pytest.approx(1.0)


def test_cosine_ends_and_stays_at_min_lr():
    opt, sched = _make(min_lr=0.01, warmup=5, steps=20)
    _advance(opt, sched, 20)
    assert _lr(opt) == pytest.approx(0.01)
    _advance(opt, sched)
    assert _lr(opt) == pytest.approx(0.01)


def test_resume_continues_the_same_curve():
    opt, sched = _make()
    _advance(opt, sched, 7)
    restored_opt, restored = _make()
    restored_opt.load_state_dict(opt.state_dict())
    restored.load_state_dict(sched.state_dict())
    _advance(opt, sched)
    _advance(restored_opt, restored)
    assert restored.get_last_lr() == sched.get_last_lr() == [_lr(restored_opt)]


@pytest.mark.parametrize(("warmup", "steps"), [(-1, 20), (20, 20), (0, 0), (0.5, 20), (0, 1.5)])
def test_rejects_invalid_lengths(warmup, steps):
    with pytest.raises(ValueError):
        _make(warmup=warmup, steps=steps)


@pytest.mark.parametrize(("lr", "min_lr"), [(1.0, float("nan")), (float("inf"), 0.0), (1.0, -0.1), (0.1, 0.2), (0.0, 0.0)])
def test_rejects_invalid_learning_rates(lr, min_lr):
    with pytest.raises(ValueError, match="learning rates"):
        _make(lr=lr, min_lr=min_lr)
