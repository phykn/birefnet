import pytest
import torch
from src.config import load_config, resolve
from src.model.checkpoint import (
    build,
    load_base,
    load_trained,
    trained_state,
    SEGMENTATION_HEADS,
)


def test_base_and_trained_keys():
    cfg = load_config()
    if not resolve(cfg.model.weight).exists():
        pytest.skip("base unavailable")
    m = build(cfg)
    base = torch.load(resolve(cfg.model.weight), weights_only=True, map_location="cpu")
    current = m.state_dict()
    replaced = []
    for key, value in base.items():
        if key.rsplit(".", 1)[0] in SEGMENTATION_HEADS:
            assert value.shape[0] == 1 and current[key].shape[0] == cfg.model.classes
            replaced.append(key)
        else:
            torch.testing.assert_close(current[key], value, atol=0, rtol=0)
    assert len(replaced) == 8
    key = "bb.patch_embed.proj.weight"
    broken = dict(base)
    broken.pop(key)
    with pytest.raises(RuntimeError, match=key):
        load_base(m, broken)
    broken = dict(base)
    broken[key] = base[key][:1]
    with pytest.raises(RuntimeError, match=key):
        load_base(m, broken)
    trained = trained_state(current)
    trained.pop(next(iter(trained)))
    with pytest.raises(RuntimeError):
        load_trained(m, trained)
