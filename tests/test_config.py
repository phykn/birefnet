import pytest
from omegaconf import OmegaConf

from src.config import load_config
from src.run import load_run


def test_explicit_config_overrides_shared_model_defaults(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"model": {"classes": 5}, "train": {"steps": 9}}, path)
    cfg = load_config(path)
    assert cfg.model.classes == 5
    assert cfg.model.weight == load_config().model.weight
    assert cfg.train.steps == 9


def test_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_sem_requires_boolean(tmp_path, value):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"sem": value}}, path)
    with pytest.raises(ValueError, match="sem must be a bool"):
        load_config(path)


def test_default_config_groups_settings():
    cfg, _, _ = load_run()
    assert set(cfg.augment) == {"zoom", "hide_edges", "brightness", "contrast", "gamma", "light", "blur", "noise"}
    assert cfg.train.ema == 0.995
    assert cfg.loss.teacher == 0.3
