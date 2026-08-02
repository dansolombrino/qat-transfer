"""CPU-only tests for the vendored RepQ-ViT adapter.

These tests require no checkpoint, dataset, or GPU. They validate the official
quantizers, current-timm attention adaptation, LayerNorm scale-folding
identity, the in-place public API, and explicit head skipping.

Run: uv run --active python code/test/repqvit.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from timm.models.swin_transformer import SwinTransformer, WindowAttention
from timm.models.vision_transformer import Attention, VisionTransformer

from src.repqvit import (
    LogSqrt2Quantizer,
    QuantLinear,
    UniformQuantizer,
    apply_repqvit_,
)
from src.repqvit.quant_model import (
    _inject_attention_matmuls_,
    _reparameterize_pair_,
)


def test_quantizers() -> None:
    torch.manual_seed(0)
    cases = (
        (torch.randn(7, 11), (7, 1)),
        (torch.randn(2, 5, 7), (1, 1, 7)),
        (torch.randn(7, 3, 2, 2), (7, 1, 1, 1)),
    )
    for x, expected_shape in cases:
        quantizer = UniformQuantizer(n_bits=3, channel_wise=True)
        y = quantizer(x)
        assert y.shape == x.shape
        assert quantizer.delta.shape == expected_shape
        assert torch.isfinite(y).all()

    constant = torch.full((3, 4), 1.25)
    constant_y = UniformQuantizer(n_bits=3, channel_wise=True)(constant)
    assert torch.equal(constant_y, constant)

    probabilities = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    log_y = LogSqrt2Quantizer(n_bits=3)(probabilities)
    assert log_y.shape == probabilities.shape
    assert torch.isfinite(log_y).all() and torch.all(log_y >= 0)
    print("PASS  official uniform/log-sqrt(2) quantizers")


def test_current_timm_attention_adapters() -> None:
    torch.manual_seed(1)

    vit_attn = Attention(dim=32, num_heads=4, qkv_bias=True).eval()
    vit_attn.fused_attn = False
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        expected = vit_attn(x)
    injected = _inject_attention_matmuls_(vit_attn)
    assert injected == [""]
    with torch.no_grad():
        actual = vit_attn(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    swin_attn = WindowAttention(
        dim=32,
        num_heads=4,
        window_size=(2, 2),
        qkv_bias=True,
    ).eval()
    swin_attn.fused_attn = False
    x_window = torch.randn(3, 4, 32)
    with torch.no_grad():
        expected_window = swin_attn(x_window)
    injected_window = _inject_attention_matmuls_(swin_attn)
    assert injected_window == [""]
    with torch.no_grad():
        actual_window = swin_attn(x_window)
    assert torch.allclose(actual_window, expected_window, atol=1e-6, rtol=1e-5)
    print("PASS  current timm ViT/Swin attention adapters preserve FP outputs")


def test_layernorm_reparameterization_identity() -> None:
    torch.manual_seed(2)
    norm = nn.LayerNorm(8)
    linear = QuantLinear(
        8,
        5,
        input_quant_params={"n_bits": 3, "channel_wise": True},
        weight_quant_params={"n_bits": 3, "channel_wise": True},
    )
    x = torch.randn(6, 10, 8)
    with torch.no_grad():
        normalized = norm(x)
        expected = linear(normalized)
        linear.input_quantizer(normalized)
        _reparameterize_pair_(norm, linear)
        actual = linear(norm(x))

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert not linear.input_quantizer.channel_wise
    assert linear.input_quantizer.delta.ndim == 0
    assert not linear.weight_quantizer.inited
    print("PASS  LayerNorm-to-Linear scale folding preserves the FP function")


class TinyImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VisionTransformer(
            img_size=16,
            patch_size=4,
            in_chans=3,
            num_classes=3,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def test_apply_repqvit_end_to_end() -> None:
    torch.manual_seed(3)
    model = TinyImageClassifier().eval()
    original_head = copy.deepcopy(model.model.head.state_dict())
    calibration = torch.randn(2, 3, 16, 16)

    quantized = apply_repqvit_(
        model=model,
        calib_data=calibration,
        w_bits=3,
        a_bits=3,
        skip_modules=frozenset({"head"}),
    )

    assert "model.head" not in quantized
    assert isinstance(model.model.head, nn.Linear)
    for key, value in original_head.items():
        assert torch.equal(model.model.head.state_dict()[key], value)
    assert any(name.endswith("attn.matmul2") for name in quantized)
    assert any(isinstance(module, QuantLinear) for module in model.modules())

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 16, 16))
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    print("PASS  apply_repqvit_ end-to-end (W3/A3, calibration, skip=head)")


class TinySwinImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SwinTransformer(
            img_size=16,
            patch_size=4,
            in_chans=3,
            num_classes=3,
            embed_dim=16,
            depths=(1, 1),
            num_heads=(2, 4),
            window_size=2,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def test_apply_repqvit_swin_end_to_end() -> None:
    torch.manual_seed(4)
    model = TinySwinImageClassifier().eval()
    calibration = torch.randn(2, 3, 16, 16)
    quantized = apply_repqvit_(
        model=model,
        calib_data=calibration,
        w_bits=3,
        a_bits=3,
        skip_modules=frozenset({"head"}),
    )
    assert any(name.endswith("downsample.reduction") for name in quantized)
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 16, 16))
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    print("PASS  apply_repqvit_ end-to-end on current timm Swin NHWC paths")


def main() -> None:
    test_quantizers()
    test_current_timm_attention_adapters()
    test_layernorm_reparameterization_identity()
    test_apply_repqvit_end_to_end()
    test_apply_repqvit_swin_end_to_end()
    print("\nALL REPQ-VIT TESTS PASSED")


if __name__ == "__main__":
    main()
