from pathlib import Path

import pytest
import torch

from src.model.checkpoint import SEGMENTATION_HEADS, load_base
from src.model import BiRefNet
from test_model import tiny_model

ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = ROOT / "weight" / "BiRefNet-general-epoch_244.pth"


@pytest.mark.skipif(not BASE_CHECKPOINT.exists(), reason="base checkpoint is absent")
def test_pinned_base_checkpoint_only_replaces_segmentation_heads():
    with torch.device("meta"):
        model = BiRefNet()
    state = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    expected = model.state_dict()
    assert state.keys() == expected.keys()
    changed = {key for key in state if state[key].shape != expected[key].shape}
    assert changed == {f"{head}.{param}" for head in SEGMENTATION_HEADS for param in ("weight", "bias")}
    for key in changed:
        assert state[key].shape == (1, *expected[key].shape[1:])


def test_base_load_preserves_shared_weights_and_reinitializes_heads(tiny_model):
    state = {key: value.clone() for key, value in tiny_model.state_dict().items()}
    heads = {f"{head}.{param}" for head in SEGMENTATION_HEADS for param in ("weight", "bias")}
    initial = {key: state[key].clone() for key in heads}
    for key in state:
        state[key] = state[key][:1] if key in heads else torch.full_like(state[key], 0.25)
    load_base(tiny_model, state)
    for key, value in tiny_model.state_dict().items():
        assert torch.equal(value, initial[key] if key in heads else state[key])


def test_base_load_rejects_unrelated_shape_mismatch(tiny_model):
    state = dict(tiny_model.state_dict())
    state["bb.patch_embed.weight"] = torch.empty(1)
    with pytest.raises(RuntimeError, match="bb.patch_embed.weight"):
        load_base(tiny_model, state)


def test_base_load_rejects_missing_head(tiny_model):
    state = dict(tiny_model.state_dict())
    del state["decoder.conv_out1.0.weight"]
    with pytest.raises(RuntimeError, match="keys"):
        load_base(tiny_model, state)
