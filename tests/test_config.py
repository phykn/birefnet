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


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_is_sem_requires_boolean(tmp_path, value):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"is_sem": value}}, path)
    with pytest.raises(ValueError, match="is_sem must be a bool"):
        load_config(path)


@pytest.mark.parametrize("mode, expected", [("rgb", False), ("sem_features", True)])
def test_legacy_input_config_is_migrated(tmp_path, mode, expected):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"mode": mode}}, path)
    cfg = load_config(path)
    assert cfg.data.is_sem is expected
    assert "mode" not in cfg.data


def test_removed_representations_are_not_silently_treated_as_rgb(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"mode": "gray_features"}}, path)
    with pytest.raises(RuntimeError, match="preprocess"):
        load_config(path)


def test_saved_crop_settings_keep_values_after_rename(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"global_prob": 0.15}}, path)
    cfg = load_config(path)
    assert cfg.data.crop_prob == pytest.approx(0.85)
    assert "global_prob" not in cfg.data


def test_conflicting_crop_setting_names_are_rejected(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"global_prob": 0.1, "crop_prob": 0.2}}, path)
    with pytest.raises(ValueError, match="crop_prob"):
        load_config(path)


@pytest.mark.parametrize("size", [512, 2048, "1024", 1024.0])
def test_config_rejects_nonstandard_input_size(tmp_path, size):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"size": size}}, path)
    with pytest.raises(ValueError, match="fixed at 1024"):
        load_config(path)


def test_legacy_standard_size_is_removed_from_config(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"size": 1024}}, path)
    assert "size" not in load_config(path).data


@pytest.mark.parametrize("size", [0, -1, 1025, "256", 256.0, True, None])
def test_invalid_min_crop_size_is_rejected(tmp_path, size):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"min_crop_size": size}}, path)
    with pytest.raises(ValueError, match="min_crop_size"):
        load_config(path)


@pytest.mark.parametrize("size", [1, 256, 512, 1024])
def test_min_crop_size_is_configurable(tmp_path, size):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"min_crop_size": size}}, path)
    assert load_config(path).data.min_crop_size == size



def test_fixed_training_options_are_not_exposed():
    cfg, _, _ = load_run()
    assert not {"mode", "freeze_bn", "backbone_stages", "backbone_lr_scale"} & set(cfg.train)
    assert "teacher" not in cfg and "grad_checkpoint" not in cfg.birefnet
    assert set(cfg.augment) == {"brightness", "contrast", "masking_prob"}
    assert cfg.train.ema_decay == 0.99


def test_old_augmentation_uses_strong_limits_without_two_views(tmp_path):
    path = tmp_path / "train.yaml"
    OmegaConf.save({"augment": {
        "weak": {"brightness": 0.1, "contrast": 0.2},
        "strong": {"brightness": 0.2, "contrast": 0.4}, "edge_prob": 1.0,
    }}, path)
    assert dict(load_config(path).augment) == {"brightness": 0.2, "contrast": 0.4, "masking_prob": 1.0}
