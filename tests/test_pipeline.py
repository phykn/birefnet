import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.config import ROOT, Config, save_config
from src.model import Output
from src.model.checkpoint import read_checkpoint, load, load_trained
from src.predict import predict_logits
from src.train.trainer import Trainer
from src.train.loss import TrainLoss
from src.train.schedule import cosine_schedule
from src.data.io import pairs, read_image
from tests.sample_data import write


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Conv2d(3, 4, 1)
        self.num_classes = 4

    def forward(self, x):
        return Output([self.decoder(x)])

    def list_trainable(self):
        return list(self.parameters())


def trainer(tmp_path, model=None):
    cfg = Config()
    cfg.train.steps = 4
    cfg.train.warmup = 0
    cfg.train.accum = 1
    m = model or Tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.train.lr)
    sch = cosine_schedule(opt, 4, cfg.train.min_lr)
    batches = [
        {
            "image": torch.rand(1, 3, 16, 16),
            "mask": torch.zeros(1, 16, 16, dtype=torch.long),
        }
    ]
    return Trainer(m, batches, [], TrainLoss(), opt, sch, tmp_path, cfg)


def test_resume(tmp_path):
    t = trainer(tmp_path)
    t.step()
    t.best_region = 0.5
    t.save()
    expected = {k: v.clone() for k, v in t.model.state_dict().items()}
    restored = trainer(tmp_path)
    restored.load_resume(tmp_path / "weights/last.train.pth")
    assert restored.global_step == 1 and restored.best_region == 0.5
    assert restored.scheduler.state_dict() == t.scheduler.state_dict()
    assert restored.optimizer.param_groups[0]["lr"] == t.optimizer.param_groups[0]["lr"]
    for k, v in restored.model.state_dict().items():
        torch.testing.assert_close(v, expected[k])
    for k, v in restored.ema.params.items():
        torch.testing.assert_close(v, t.ema.params[k])
    restored.step()
    assert restored.global_step == 2
    state = read_checkpoint(tmp_path / "weights/last.pth")
    for k, v in state["model"].items():
        torch.testing.assert_close(v, t.ema.params[k])


def test_nonfinite(tmp_path):
    t = trainer(tmp_path)
    t.train_loader[0]["image"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="step 1"):
        t.step()


def test_skipped_step(tmp_path):
    t = trainer(tmp_path)

    class Scaler:
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

    t.scaler = Scaler()
    before = {k: v.clone() for k, v in t.ema.params.items()}
    schedule = t.scheduler.last_epoch
    _, updated = t.step()
    assert not updated and t.global_step == 0 and t.scheduler.last_epoch == schedule
    for k, v in t.ema.params.items():
        torch.testing.assert_close(v, before[k])


def test_legacy(tmp_path):
    p = tmp_path / "v2.pth"
    torch.save({"format": "birefnet-multiclass-v2"}, p)
    with pytest.raises(ValueError, match="750d7d9"):
        read_checkpoint(p)


def test_tiles_shape_and_batch():
    m = Tiny().eval()
    x = np.arange(35 * 47, dtype=np.uint8).reshape(35, 47)
    for tiles in ((1,), (1, 3)):
        a = predict_logits(m, x, size=64, tiles=tiles, tile_batch=1)
        b = predict_logits(m, x, size=64, tiles=tiles, tile_batch=3)
        assert a.shape == (4, 35, 47)
        np.testing.assert_allclose(a, b, atol=1.0e-6, rtol=0)


def test_typo_before_model(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("augment:\n  lights: 1\n")
    result = subprocess.run(
        [sys.executable, "run_train.py", "--config", str(p)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode and "augment.lights" in result.stderr


@pytest.mark.skipif(
    not (ROOT / "weight/BiRefNet-general-epoch_244.pth").exists(),
    reason="base unavailable",
)
def test_real_train_predict_evaluate(tmp_path):
    from src.run import train

    data = tmp_path / "data"
    write(data)
    cfg = Config()
    cfg.data.image_dir = str(data / "image")
    cfg.data.mask_dir = str(data / "mask")
    cfg.train.steps = 2
    cfg.train.warmup = 0
    cfg.train.accum = 1
    cfg.train.crop = 64
    cfg.train.val_every = 2
    cfg.train.save_every = 2
    cfg.loader.batch_size = 1
    cfg.loader.num_workers = 0
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    target = train(config=path, runs=tmp_path / "run", device=torch.device("cpu"))
    assert {p.name for p in (target / "weights").iterdir()} == {
        "last.pth",
        "last.train.pth",
        "best_miou.pth",
    }
    for p in (target / "weights").iterdir():
        assert read_checkpoint(p)["format"] == "birefnet-multiclass-v3"
    log = EventAccumulator(str(target / "logs")).Reload()
    assert len(log.Scalars("train/loss")) == 2 and len(log.Scalars("valid/miou")) == 1
    copied = tmp_path / "detached.pth"
    shutil.copy(target / "weights/best_miou.pth", copied)
    image = read_image(data / "image/sample_01.png")
    m = load(copied, torch.device("cpu"))
    a = predict_logits(m, image, size=64).argmax(0)
    load_trained(m, read_checkpoint(target / "weights/best_miou.pth")["model"])
    np.testing.assert_array_equal(a, predict_logits(m, image, size=64).argmax(0))
    del m
    env = __import__("os").environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["OMP_NUM_THREADS"] = "2"
    # Use a small real image for UI checks; inference still uses its fixed 1024 canvas.
    cmd = [
        sys.executable,
        "scripts/03_predict.py",
        "--weight",
        str(copied),
        "--image",
        str(data / "image/sample_01.png"),
        "--mask",
        str(data / "mask/sample_01.png"),
        "--tiles",
        "1",
        "3",
        "--class-id",
        "1",
        "--device",
        "cpu",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/04_evaluate.py",
            "--weight",
            str(copied),
            "--images",
            str(data / "image"),
            "--masks",
            str(data / "mask"),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "pairs: 4" in result.stdout and "pixel_accuracy:" in result.stdout


def test_early_rejections(tmp_path, monkeypatch):
    from src import run

    def forbidden(cfg):
        raise AssertionError("model must not be built")

    monkeypatch.setattr(run, "build", forbidden)
    with pytest.raises(ValueError, match="combined"):
        run.train(resume="unused", config="unused")
    data = tmp_path / "data"
    write(data)
    cfg = Config()
    cfg.data.image_dir = str(data / "image")
    cfg.data.mask_dir = str(data / "mask")
    target = tmp_path / "run"
    (target / "weights").mkdir(parents=True)
    save_config(cfg, target / "config.yaml")
    from src.data import split

    split.save(split.make(pairs(data / "image", data / "mask"), 0.25), target)
    (data / "image/sample_01.png").unlink()
    (data / "mask/sample_01.png").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        run.train(resume=target / "weights/last.train.pth")


def test_skipped_limit(tmp_path, monkeypatch):
    t = trainer(tmp_path)
    monkeypatch.setattr(t, "step", lambda: ({"loss": 1.0}, False))
    with pytest.raises(FloatingPointError, match="16 consecutive"):
        t.train(1)
