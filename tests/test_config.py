import pytest
from omegaconf import OmegaConf

from src.config import ROOT, load_config, load_run


def test_explicit_config_overrides_shared_model_defaults(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"birefnet": {"num_classes": 5}, "train": {"steps": 9}}, path)
    cfg = load_config(path)
    assert cfg.birefnet.num_classes == 5
    assert cfg.birefnet.weight == load_config().birefnet.weight
    assert cfg.train.steps == 9


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


def test_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
