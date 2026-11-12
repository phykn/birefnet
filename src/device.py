from contextlib import nullcontext

import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_amp_dtype(device):
    if device.type == "cpu":
        return None
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return torch.bfloat16
    return torch.float16


def autocast(device, dtype):
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        return torch.amp.autocast(device.type, dtype=dtype, enabled=dtype is not None)
