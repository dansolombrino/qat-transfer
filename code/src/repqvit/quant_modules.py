import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .quantizer import UniformQuantizer, LogSqrt2Quantizer


class QuantConv2d(nn.Conv2d):
    """Quantized Conv2d — used for patch-embedding layers."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, input_quant_params={}, weight_quant_params={}):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias)
        input_quant_params_conv = deepcopy(input_quant_params)
        input_quant_params_conv['n_bits'] = 8
        self.input_quantizer = UniformQuantizer(**input_quant_params_conv)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super().__repr__()
        return "(" + s + f" input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_quant:
            x = self.input_quantizer(x)
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    """Quantized Linear — used for Q/K/V, projections, and MLP layers."""

    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super().__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super().__repr__()
        return "(" + s + f" input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_quant:
            x = self.input_quantizer(x)
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        return F.linear(x, w, self.bias)


class QuantMatMul(nn.Module):
    """
    Quantized matrix multiplication.
    matmul1 (pre-softmax, QK^T): uniform quantizer for both operands.
    matmul2 (post-softmax, AV):  log-sqrt(2) quantizer for A (attention weights),
                                  uniform for V.
    """

    def __init__(self, input_quant_params={}):
        super().__init__()
        params_a = deepcopy(input_quant_params)
        if 'log_quant' in params_a:
            params_a.pop('log_quant')
            self.quantizer_A = LogSqrt2Quantizer(**params_a)
        else:
            self.quantizer_A = UniformQuantizer(**params_a)
        params_b = deepcopy(input_quant_params)
        params_b.pop('log_quant', None)
        self.quantizer_B = UniformQuantizer(**params_b)
        self.use_input_quant = False

    def __repr__(self):
        s = super().__repr__()
        return "(" + s + f" input_quant={self.use_input_quant})"

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        return A @ B
