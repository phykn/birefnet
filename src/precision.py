from contextlib import nullcontext

import torch


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return torch.bfloat16
    return torch.float16


def autocast(device: torch.device, dtype: torch.dtype, enabled: bool = True):
    context = torch.cuda.device(device) if enabled and device.type == "cuda" else nullcontext()
    # Autocast checks BF16 support while its context is constructed.
    with context:
        return torch.amp.autocast(device.type, dtype=dtype, enabled=enabled)
