import pytest
import torch

from src.train.schedule import CosineSchedule

def test_shrinking_cycles_keep_a_decay_step():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = CosineSchedule(
        optimizer, first_cycle_steps=2, cycle_mult=0.1,
        max_lr=0.1, min_lr=0.001,
    )

    for _ in range(6):
        optimizer.step()
        scheduler.step()
        assert scheduler.cur_cycle_steps >= 1
        assert 0.001 <= optimizer.param_groups[0]["lr"] <= 0.1


def test_shrinking_cycles_keep_warmup_and_decay():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = CosineSchedule(
        optimizer, first_cycle_steps=3, cycle_mult=0.1,
        max_lr=0.1, min_lr=0.001, warmup_steps=1,
    )

    rates = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        assert scheduler.cur_cycle_steps > scheduler.warmup_steps
        rates.append(optimizer.param_groups[0]["lr"])
    assert 0.1 in rates[3:]

def _make(max_lr=1.0, min_lr=0.0, warmup=5, total=20):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.0)
    sched = CosineSchedule(
        optimizer=opt,
        first_cycle_steps=total,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup,
    )
    return opt, sched


def _lr(opt):
    return opt.param_groups[0]["lr"]


def test_starts_at_min_lr():
    opt, _ = _make()
    assert _lr(opt) == 0.0


def test_without_warmup_starts_at_max_lr():
    opt, sched = _make(max_lr=1.0, min_lr=0.01, warmup=0, total=20)
    assert _lr(opt) == 1.0
    sched.step()
    assert 0.01 < _lr(opt) < 1.0


def test_last_lr_matches_optimizer_before_and_after_resume():
    opt, sched = _make()
    assert sched.get_last_lr() == [_lr(opt)]
    for _ in range(7):
        sched.step()
        assert sched.get_last_lr() == [_lr(opt)]
    restored_opt, restored = _make()
    restored_opt.load_state_dict(opt.state_dict())
    restored.load_state_dict(sched.state_dict())
    sched.step()
    restored.step()
    assert restored.get_last_lr() == sched.get_last_lr() == [_lr(restored_opt)]


def test_warmup_reaches_max_lr_after_warmup_steps():
    opt, sched = _make(max_lr=1.0, warmup=5, total=20)
    for _ in range(5):
        sched.step()
    assert abs(_lr(opt) - 1.0) < 1e-6


def test_warmup_is_monotonic_increasing():
    opt, sched = _make(max_lr=1.0, warmup=5, total=20)
    prev = _lr(opt)
    for _ in range(5):
        sched.step()
        cur = _lr(opt)
        assert cur >= prev
        prev = cur


def test_cosine_decays_to_min_lr_at_end_of_cycle():
    opt, sched = _make(max_lr=1.0, min_lr=0.01, warmup=5, total=20)
    for _ in range(20):
        sched.step()
    assert abs(_lr(opt) - 0.01) < 1e-6


def test_parameter_groups_keep_their_lr_scale():
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    opt = torch.optim.SGD(
        [
            {"params": [params[0]], "max_lr": 1.0, "min_lr": 0.1},
            {"params": [params[1]], "max_lr": 0.5, "min_lr": 0.05},
        ]
    )
    sched = CosineSchedule(
        optimizer=opt,
        first_cycle_steps=20,
        max_lr=1.0,
        min_lr=0.1,
        warmup_steps=5,
    )
    for _ in range(20):
        lrs = [group["lr"] for group in opt.param_groups]
        assert abs(lrs[1] - 0.5 * lrs[0]) < 1e-9
        sched.step()


@pytest.mark.parametrize(
    ("warmup", "total"),
    [(-1, 20), (20, 20), (0, 0)],
)
def test_rejects_invalid_cycle_lengths(warmup, total):
    with pytest.raises(ValueError):
        _make(warmup=warmup, total=total)


@pytest.mark.parametrize("options", [
    {"first_cycle_steps": 1.5}, {"first_cycle_steps": float("nan")},
    {"warmup_steps": 0.5}, {"warmup_steps": float("nan")},
    {"cycle_mult": float("nan")}, {"cycle_mult": float("inf")},
    {"gamma": float("nan")}, {"gamma": float("inf")},
    {"min_lr": float("nan")}, {"max_lr": float("inf")},
    {"min_lr": -0.1}, {"max_lr": -0.1},
    {"min_lr": 0.2, "max_lr": 0.1},
])
def test_rejects_invalid_schedule_before_changing_optimizer(options):
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=0.05)
    kwargs = {"first_cycle_steps": 10, **options}

    with pytest.raises(ValueError):
        CosineSchedule(optimizer, **kwargs)

    assert optimizer.param_groups[0]["lr"] == 0.05


def test_validates_learning_rate_overrides_per_group():
    params = [torch.nn.Parameter(torch.ones(())) for _ in range(2)]
    optimizer = torch.optim.SGD([
        {"params": [params[0]], "min_lr": 0.01},
        {"params": [params[1]], "min_lr": float("nan")},
    ], lr=0.05)

    with pytest.raises(ValueError, match="learning rates"):
        CosineSchedule(optimizer, first_cycle_steps=10, max_lr=0.1)

    assert [group["lr"] for group in optimizer.param_groups] == [0.05, 0.05]


@pytest.mark.parametrize("warmup", [0, 1])
@pytest.mark.parametrize("gamma", [0.0, 0.1])
def test_restart_decay_respects_minimum_learning_rate(warmup, gamma):
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = CosineSchedule(
        optimizer, first_cycle_steps=4, warmup_steps=warmup,
        max_lr=0.1, min_lr=0.01, gamma=gamma,
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert all(0.01 <= lr <= 0.1 for lr in rates)
    assert rates[-1] == pytest.approx(0.01)
