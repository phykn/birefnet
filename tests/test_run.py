from argparse import Namespace

import pytest
from omegaconf import OmegaConf

import run_train
from src import run
from src.config import ROOT, load_config
from src.run import create_run_dir, load_run
from src.data.split import save as save_splits


def _make_run(tmp_path):
    run_dir = tmp_path / "run" / "sample"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    checkpoint = weights_dir / "last.train.pth"
    checkpoint.touch()
    cfg = OmegaConf.create({"marker": "saved"})
    OmegaConf.save(cfg, run_dir / "config.yaml")
    splits = {
        "train_image": ["train.png"],
        "train_mask": ["train.png"],
        "valid_image": ["valid.png"],
        "valid_mask": ["valid.png"],
    }
    save_splits(splits, run_dir)
    return run_dir, checkpoint, splits


def test_default_run_combines_train_and_model_config():
    cfg, checkpoint, run_dir = load_run()
    train = OmegaConf.load(ROOT / "config/train.yaml")
    assert cfg.train == train.train
    assert cfg.birefnet == load_config().birefnet
    assert checkpoint is None
    assert run_dir is None


def test_resume_rejects_config_override():
    with pytest.raises(ValueError, match="saved config"):
        load_run("last.train.pth", "train.yaml")


def test_create_run_dir_never_reuses_existing_run(tmp_path):
    first = create_run_dir(tmp_path)
    second = create_run_dir(tmp_path)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_load_run_uses_saved_config(tmp_path):
    run_dir, checkpoint, _ = _make_run(tmp_path)

    cfg, actual_checkpoint, actual_run_dir = load_run(str(checkpoint))

    assert cfg.marker == "saved"
    assert actual_checkpoint == checkpoint.resolve()
    assert actual_run_dir == run_dir.resolve()


def test_load_run_requires_saved_config(tmp_path):
    run_dir, checkpoint, _ = _make_run(tmp_path)
    (run_dir / "config.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="Run config not found"):
        load_run(str(checkpoint))


def test_main_resumes_in_existing_run(monkeypatch, tmp_path):
    run_dir, checkpoint, splits = _make_run(tmp_path)
    cfg = OmegaConf.create(
        {
            "marker": "saved",
            "train": {"steps": 7, "val_freq": 2, "save_freq": 3},
        }
    )
    OmegaConf.save(cfg, run_dir / "config.yaml")
    calls = {}

    class Base:
        stats = {"total": 10, "trainable": 2}
        def to(self, device):
            return self

    class Trainer:
        save_dir = str(run_dir)

        def load_resume(self, path):
            calls["checkpoint"] = path

        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setattr(
        run_train,
        "parse_args",
        lambda: Namespace(resume=str(checkpoint), config=None),
    )
    def build_model(actual, load_pretrained):
        calls["load_pretrained"] = load_pretrained
        return Base()

    monkeypatch.setattr(run, "build_model", build_model)

    def build_data(actual, saved):
        calls["config"] = actual
        calls["splits"] = saved
        return "train", "valid", splits

    def build_trainer(**kwargs):
        calls["save_dir"] = kwargs["save_dir"]
        return Trainer()

    monkeypatch.setattr(run, "build_data", build_data)
    monkeypatch.setattr(run, "build_trainer", build_trainer)
    original = {name: (run_dir / name).read_bytes()
                for name in ("config.yaml", "train.csv", "valid.csv")}

    run_train.main()

    assert calls["config"].marker == "saved"
    assert calls["load_pretrained"] is False
    assert calls["splits"] == splits
    assert calls["save_dir"] == run_dir.resolve()
    assert calls["checkpoint"] == str(checkpoint.resolve())
    assert calls["train"] == {"steps": 7, "val_freq": 2, "save_freq": 3}
    for name, content in original.items():
        assert (run_dir / name).read_bytes() == content


def test_new_run_saves_config_and_splits_before_training(monkeypatch, tmp_path):
    from src.data.split import load as load_splits

    cfg = load_config(ROOT / "config/train.yaml")
    path = tmp_path / "custom.yaml"
    cfg.train.steps = 3
    OmegaConf.save(cfg, path)
    splits = {"train_image": ["train.png"], "train_mask": ["train.png"],
              "valid_image": ["valid.png"], "valid_mask": ["valid.png"]}
    calls = {}

    class Model:
        stats = {"total": 10, "trainable": 2}

        def to(self, device):
            return self

    class Trainer:
        def train(self, **kwargs):
            target = calls["save_dir"]
            assert OmegaConf.load(target / "config.yaml") == cfg
            assert load_splits(target) == splits
            calls["train"] = kwargs

        def load_resume(self, path):
            raise AssertionError("New runs must not restore a training checkpoint")

    def build_model(actual, load_pretrained):
        assert actual == cfg
        assert load_pretrained is True
        return Model()

    def build_data(actual, saved):
        assert saved is None
        return "train", "valid", splits

    def build_trainer(**kwargs):
        calls["save_dir"] = kwargs["save_dir"]
        return Trainer()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "build_model", build_model)
    monkeypatch.setattr(run, "build_data", build_data)
    monkeypatch.setattr(run, "build_trainer", build_trainer)
    monkeypatch.setattr(run_train, "parse_args", lambda: Namespace(resume=None, config=path))
    run_train.main()
    assert calls["save_dir"].resolve().parent == tmp_path / "run"
    assert calls["train"] == {"steps": 3, "val_freq": cfg.train.val_freq,
                              "save_freq": cfg.train.save_freq}


def test_missing_saved_split_fails_before_building_model(monkeypatch, tmp_path):
    run_dir, checkpoint, _ = _make_run(tmp_path)
    (run_dir / "valid.csv").unlink()

    def fail(*args, **kwargs):
        raise AssertionError("Must check saved split files before allocating the model")

    monkeypatch.setattr(run, "build_model", fail)
    with pytest.raises(FileNotFoundError):
        run.train(checkpoint)
