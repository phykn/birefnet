import importlib
import matplotlib
import shutil
import sys
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.config import load_config, save_config
from src.model import Output
from src.model.checkpoint import read_checkpoint, load, load_trained
from src.predict import predict
from src.data.io import pair_files, read_image
from tests.sample_data import write
from tests.train_utils import Tiny, trainer


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
        a = predict(m, x, size=64, tiles=tiles, tile_batch=1)["logits"]
        b = predict(m, x, size=64, tiles=tiles, tile_batch=3)["logits"]
        assert a.shape == (4, 35, 47)
        np.testing.assert_allclose(a, b, atol=1.0e-6, rtol=0)


def test_typo_before_model(tmp_path, monkeypatch):
    import run_train

    p = tmp_path / "bad.yaml"
    p.write_text("augment:\n  lights: 1\n")
    monkeypatch.setattr(sys, "argv", ["run_train.py", "--config", str(p)])
    with pytest.raises(ValueError, match="augment.lights"):
        run_train.main()


def use_tiny(monkeypatch):
    from src import run
    from src.model import checkpoint

    monkeypatch.setattr(run, "build", lambda cfg: Tiny())
    monkeypatch.setattr(checkpoint, "build", lambda cfg: Tiny())


def run_script(monkeypatch, name, *args):
    monkeypatch.setattr(sys, "argv", [name, *args])
    importlib.import_module(f"scripts.{name}").main()


def test_train_predict_evaluate(tmp_path, monkeypatch, capsys):
    from src.run import train

    use_tiny(monkeypatch)
    data = tmp_path / "data"
    write(data)
    cfg = load_config()
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
    a = predict(m, image, size=64)["classes"]
    load_trained(m, read_checkpoint(target / "weights/best_miou.pth")["model"])
    np.testing.assert_array_equal(a, predict(m, image, size=64)["classes"])
    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    run_script(
        monkeypatch, "03_predict",
        "--weight", str(copied),
        "--image", str(data / "image/sample_01.png"),
        "--mask", str(data / "mask/sample_01.png"),
        "--tiles", "1", "2",
        "--class-id", "1",
        "--device", "cpu",
    )
    run_script(
        monkeypatch, "04_evaluate",
        "--weight", str(copied),
        "--images", str(data / "image"),
        "--masks", str(data / "mask"),
        "--device", "cpu",
    )
    out = capsys.readouterr().out
    assert "pairs: 4" in out and "pixel_accuracy:" in out


def test_early_rejections(tmp_path, monkeypatch):
    from src import run

    def forbidden(cfg):
        raise AssertionError("model must not be built")

    monkeypatch.setattr(run, "build", forbidden)
    data = tmp_path / "data"
    write(data)
    cfg = load_config()
    cfg.data.image_dir = str(data / "image")
    cfg.data.mask_dir = str(data / "mask")
    target = tmp_path / "run"
    (target / "weights").mkdir(parents=True)
    save_config(cfg, target / "config.yaml")
    from src.data import split

    pairs = pair_files(data / "image", data / "mask")
    split.write_split_csv(split.split_pairs(pairs, 0.25), target)
    (data / "image/sample_01.png").unlink()
    (data / "mask/sample_01.png").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        run.train(resume=target / "weights/last.train.pth")


def test_train_and_resume_paths_use_project_root(tmp_path, monkeypatch):
    from src import run

    root = tmp_path / "project"
    write(root / "data")
    cfg = load_config()
    cfg.data.image_dir = "data/image"
    cfg.data.mask_dir = "data/mask"
    cfg.train.steps = cfg.train.val_every = cfg.train.save_every = 1
    cfg.train.warmup = 0
    cfg.train.accum = 1
    cfg.train.crop = 64
    cfg.loader.batch_size = 1
    cfg.loader.num_workers = 0
    save_config(cfg, root / "config.yaml")
    monkeypatch.setattr("src.config.ROOT", root)
    monkeypatch.setattr(run, "build", lambda cfg: Tiny())
    monkeypatch.chdir(tmp_path)
    target = run.train(runs="run", device=torch.device("cpu"))
    assert target.parent == root / "run"
    checkpoint = target / "weights/last.train.pth"
    assert run.train(
        resume=checkpoint.relative_to(root), device=torch.device("cpu")
    ) == target
    assert read_checkpoint(checkpoint)["step"] == 1


def test_skipped_limit(tmp_path, monkeypatch):
    t = trainer(tmp_path)
    monkeypatch.setattr(t, "step", lambda: ({"loss": 1.0}, False))
    t.cfg.train.steps = 1
    with pytest.raises(FloatingPointError, match="16 consecutive"):
        t.train()


def test_train_with_labels(tmp_path, monkeypatch):
    from PIL import Image
    from src.run import train

    use_tiny(monkeypatch)

    data = tmp_path / "data"
    write(data)
    (data / "label").mkdir()
    for mask_path in (data / "mask").iterdir():
        mask = np.asarray(Image.open(mask_path))
        label = np.full(mask.shape, 255, np.uint8)
        label[100:110, 100:140] = mask[100:110, 100:140]
        Image.fromarray(label).save(data / "label" / mask_path.name)
    cfg = load_config()
    cfg.data.image_dir = str(data / "image")
    cfg.data.mask_dir = str(data / "mask")
    cfg.data.label_dir = str(data / "label")
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
    assert (target / "train.csv").read_text().startswith("image,mask,label")
    log = EventAccumulator(str(target / "logs")).Reload()
    tags = set(log.Tags()["scalars"])
    assert {"train/label_ce", "train/ce", "train/dice"} <= tags
    for name in ("label", "mask"):
        assert {f"valid/{name}_miou", f"valid/{name}_pixel_accuracy"} <= tags
        assert {f"valid/{name}_class_{i}_iou" for i in range(4)} <= tags
    assert "valid/miou" not in tags
    metrics = read_checkpoint(target / "weights/best_miou.pth")["metrics"]
    assert "label_miou" in metrics and "mask_miou" in metrics


class HeadTiny(Tiny):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.conv_out1 = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))

    def forward(self, x):
        return Output([self.decoder.conv_out1(x)])


def test_class_heads_get_a_larger_lr(tmp_path, monkeypatch):
    from src import run
    from src.model import checkpoint

    monkeypatch.setattr(run, "build", lambda cfg: HeadTiny())
    monkeypatch.setattr(checkpoint, "build", lambda cfg: HeadTiny())
    data = tmp_path / "data"
    write(data)
    cfg = load_config()
    cfg.data.image_dir = str(data / "image")
    cfg.data.mask_dir = str(data / "mask")
    cfg.train.steps = cfg.train.val_every = cfg.train.save_every = 1
    cfg.train.warmup = 0
    cfg.train.crop = 64
    cfg.train.accum = 1
    cfg.loader.batch_size = 1
    cfg.loader.num_workers = 0
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    target = run.train(config=path, runs=tmp_path / "run", device=torch.device("cpu"))
    state = read_checkpoint(target / "weights/last.train.pth")
    groups = state["optimizer"]["param_groups"]
    assert len(groups[0]["params"]) == 0 and len(groups[1]["params"]) == 2
    assert groups[1]["initial_lr"] == cfg.train.lr * cfg.train.head_lr_scale
