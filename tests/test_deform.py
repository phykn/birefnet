import pytest
import torch
from torchvision.ops import deform_conv2d

from src.model.decoder.deform import DeformableConv2d, grid_deform_conv2d


@pytest.mark.parametrize("kernel, padding, stride", [(3, 1, 1), (1, 0, 1), (3, 1, 2), (5, 2, 1)])
def test_grid_deform_conv_matches_torchvision(kernel, padding, stride):
    torch.manual_seed(0)
    out_h = (9 + 2 * padding - kernel) // stride + 1
    out_w = (11 + 2 * padding - kernel) // stride + 1
    inputs = (
        torch.randn(2, 4, 9, 11, dtype=torch.float64),
        # Large offsets also sample outside the image.
        torch.randn(2, 2 * kernel * kernel, out_h, out_w, dtype=torch.float64) * 3,
        torch.randn(5, 4, kernel, kernel, dtype=torch.float64),
        torch.randn(5, dtype=torch.float64),
        torch.rand(2, kernel * kernel, out_h, out_w, dtype=torch.float64),
    )
    for value in inputs:
        value.requires_grad_(True)
    x, offset, weight, bias, mask = inputs
    expected = deform_conv2d(x, offset, weight, bias, stride=stride, padding=padding, mask=mask)
    actual = grid_deform_conv2d(x, offset, weight, bias, stride=stride, padding=padding, mask=mask)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad(actual.square().sum(), inputs),
                         torch.autograd.grad(expected.square().sum(), inputs)):
        torch.testing.assert_close(got, want)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_deformable_conv_trains_on_mps_without_cpu_fallback():
    torch.manual_seed(0)
    layer = DeformableConv2d(4, 5)
    with torch.no_grad():
        layer.offset_conv.weight.normal_(0, 0.1)
    x = torch.randn(1, 4, 16, 16)
    expected = layer(x)
    layer.to("mps")
    actual = layer(x.to("mps"))
    actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    assert layer.offset_conv.weight.grad.abs().sum() > 0
