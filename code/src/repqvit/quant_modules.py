"""Quantized modules from official RepQ-ViT, adapted for current PyTorch.

Derived from ``classification/quant/quant_modules.py`` in the official
RepQ-ViT release (Apache-2.0).  Modifications preserve module device/dtype and
bias semantics, avoid mutable defaults, and support current timm Swin NHWC
PatchMerging activations.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .quantizer import LogSqrt2Quantizer, UniformQuantizer


class QuantConv2d(nn.Conv2d):
    """RepQ-ViT quantized Conv2d (patch-embedding input stays at 8 bits)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        input_quant_params: dict | None = None,
        weight_quant_params: dict | None = None,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            device=device,
            dtype=dtype,
        )
        input_params = deepcopy(input_quant_params or {})
        input_params["n_bits"] = 8
        self.input_quantizer = UniformQuantizer(**input_params)
        self.weight_quantizer = UniformQuantizer(**(weight_quant_params or {}))
        self.use_input_quant = False
        self.use_weight_quant = False

    def set_quant_state(self, input_quant: bool = False, weight_quant: bool = False) -> None:
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_quant:
            x = self.input_quantizer(x)
        weight = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantLinear(nn.Linear):
    """RepQ-ViT quantized Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_quant_params: dict | None = None,
        weight_quant_params: dict | None = None,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.input_quantizer = UniformQuantizer(**(input_quant_params or {}))
        self.weight_quantizer = UniformQuantizer(**(weight_quant_params or {}))
        self.use_input_quant = False
        self.use_weight_quant = False

    def set_quant_state(self, input_quant: bool = False, weight_quant: bool = False) -> None:
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_quant:
            if self.input_quantizer.channel_wise and x.ndim == 4:
                # Current timm Swin passes NHWC tensors through MLP fc1 and
                # PatchMerging.reduction. RepQ-ViT's channel axis remains the
                # last dimension, so flatten only the two spatial dimensions.
                batch, height, width, channels = x.shape
                x = self.input_quantizer(x.reshape(batch, height * width, channels))
                x = x.reshape(batch, height, width, channels)
            else:
                x = self.input_quantizer(x)
        weight = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        return F.linear(x, weight, self.bias)


class QuantMatMul(nn.Module):
    """RepQ-ViT quantized attention matrix multiplication."""

    def __init__(self, input_quant_params: dict | None = None):
        super().__init__()
        params_a = deepcopy(input_quant_params or {})
        log_quant = bool(params_a.pop("log_quant", False))
        self.quantizer_A = (
            LogSqrt2Quantizer(**params_a)
            if log_quant
            else UniformQuantizer(**params_a)
        )
        self.quantizer_B = UniformQuantizer(**params_a)
        self.use_input_quant = False

    def set_quant_state(self, input_quant: bool = False, weight_quant: bool = False) -> None:
        del weight_quant
        self.use_input_quant = input_quant

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.use_input_quant:
            a = self.quantizer_A(a)
            b = self.quantizer_B(b)
        return a @ b
