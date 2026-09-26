"""Compare an explicit upstream checkout and base checkpoint without downloads."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import BiRefNet


def compare(actual, expected) -> float:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return float((actual - expected).abs().max())
    if expected is None:
        assert actual is None
        return 0.0
    assert type(actual) is type(expected)
    assert len(actual) == len(expected)
    return max(compare(a, b) for a, b in zip(actual, expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"Upstream: {revision}", flush=True)
    sys.path.insert(0, str(source))
    from models.birefnet import BiRefNet as UpstreamBiRefNet

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        upstream = UpstreamBiRefNet(bb_pretrained=False)
        local = BiRefNet(num_classes=1)
    for model in (upstream, local):
        model.load_state_dict(state, strict=True, assign=True)
        model.to(args.device)
    del state
    print("Both models strictly loaded the same checkpoint.", flush=True)

    with torch.inference_mode():
        for training, shape in [
            (False, (1, 3, 64, 96)),
            (False, (1, 3, 128, 128)),
            (True, (2, 3, 64, 64)),
        ]:
            for model in (upstream, local):
                model.train(training)
                model.bb.eval()
                # Use the same BatchNorm statistics for both implementations.
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            torch.manual_seed(7)
            x = torch.randn(shape, device=args.device)
            torch.manual_seed(11)
            expected = upstream(x)
            torch.manual_seed(11)
            actual = local(x)
            actual = [[list(actual.gdt), actual.logits], [None]] if training else actual.logits
            error = compare(actual, expected)
            print(
                f"{'train' if training else 'eval'} {shape}: "
                f"max_abs_error={error:.8g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
