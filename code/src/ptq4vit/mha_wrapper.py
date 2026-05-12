"""FusedQKVQuantLinear — wraps nn.MultiheadAttention's fused in_proj_weight for PTQ4ViT."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_linear import PTQSLBatchingQuantLinear


class FusedQKVQuantLinear(PTQSLBatchingQuantLinear):
    """Treats nn.MultiheadAttention's fused in_proj_weight as a PTQ4ViT-compatible linear.

    Shares in_proj_weight/in_proj_bias from the parent MHA (no copy).
    Compatible with PTQ4ViT's calibration hooks via inheritance from MinMaxQuantLinear.
    """

    def __init__(self, mha: nn.MultiheadAttention, **ptqsl_kwargs):
        embed_dim = mha.embed_dim
        has_bias = mha.in_proj_bias is not None
        super().__init__(embed_dim, 3 * embed_dim, bias=has_bias, **ptqsl_kwargs)
        # Share weight/bias with the parent MHA — do NOT copy
        del self.weight
        self.weight = mha.in_proj_weight
        if has_bias:
            del self.bias
            self.bias = mha.in_proj_bias
        else:
            del self.bias
            self.bias = None

    def forward(self, x):
        """Mode-aware forward: raw, calibration_step1/2, quant_forward."""
        if self.mode == 'raw':
            out = F.linear(x, self.weight, self.bias)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out
