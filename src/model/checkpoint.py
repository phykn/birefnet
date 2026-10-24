from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..prepare.spec import PreprocessSpec

FORMAT = "birefnet-multiclass-v2"
# Only these parts train. The backbone comes from the base weights and depth from
# Depth Anything, so checkpoints keep just these and stay about a tenth the size.
TRAINED = ("squeeze_module.", "decoder.")
SEGMENTATION_HEADS = (
    "decoder.conv_out1.0",
    "decoder.conv_ms_spvn_2",
    "decoder.conv_ms_spvn_3",
    "decoder.conv_ms_spvn_4",
)


def load_base(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("Base checkpoint must be a flat tensor state_dict")
    # The depth estimator and its decoder branch are not part of the base weights:
    # the estimator comes with its own weights and the branch starts at zero.
    added = ("depth.", "decoder.dpt_blks.")
    expected = {key: value for key, value in model.state_dict().items() if not key.startswith(added)}
    if state.keys() != expected.keys():
        raise RuntimeError("Base checkpoint keys do not match BiRefNet")
    replaced = {}
    for key, value in state.items():
        shape = expected[key].shape
        if value.shape == shape:
            continue
        is_head = key.rsplit(".", 1)[0] in SEGMENTATION_HEADS
        if not is_head or value.shape != (1, *shape[1:]) or shape[0] != model.num_classes:
            raise RuntimeError(f"Incompatible base checkpoint tensor: {key}: {tuple(value.shape)} != {tuple(shape)}")
        replaced[key] = expected[key]
    missing, unexpected = model.load_state_dict({**state, **replaced}, strict=False)
    if unexpected or any(not key.startswith(added) for key in missing):
        raise RuntimeError("Base checkpoint keys do not match BiRefNet")


def trained_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if key.startswith(TRAINED)}


def load_trained(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a checkpoint into a model that already holds its base weights."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(key.startswith(TRAINED) for key in missing):
        raise RuntimeError("Checkpoint does not match the trained part of BiRefNet")


def read_checkpoint(path: str | Path, num_classes: int) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise RuntimeError(f"Expected a {FORMAT} full model checkpoint")
    if state.get("num_classes") != num_classes:
        raise RuntimeError("Checkpoint num_classes does not match configuration")
    return state


def pack_model(
    model: nn.Module, preprocess: PreprocessSpec, ignore_index: int = 255
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "model": {key: value.detach().cpu() for key, value in trained_state(model.state_dict()).items()},
        "num_classes": model.num_classes,
        "ignore_index": ignore_index,
        "preprocess": preprocess.to_meta(),
    }
