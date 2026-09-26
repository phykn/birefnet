from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..prepare.spec import PreprocessSpec

FORMAT = "birefnet-multiclass-v1"
SEGMENTATION_HEADS = (
    "decoder.conv_out1.0",
    "decoder.conv_ms_spvn_2",
    "decoder.conv_ms_spvn_3",
    "decoder.conv_ms_spvn_4",
)


def load_base(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("Base checkpoint must be a flat tensor state_dict")
    expected = model.state_dict()
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
    model.load_state_dict({**state, **replaced}, strict=True)


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
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "num_classes": model.num_classes,
        "ignore_index": ignore_index,
        "preprocess": preprocess.to_meta(),
    }
