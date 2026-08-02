"""RepQ-ViT post-training quantization adapted for this project.

The implementation is derived from the official ICCV 2023 RepQ-ViT
classification release (https://github.com/zkkli/RepQ-ViT), licensed under
Apache-2.0.  See ``LICENSE`` and the per-file modification notices.
"""

from .quant_model import apply_repqvit_
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul
from .quantizer import LogSqrt2Quantizer, UniformQuantizer

__all__ = [
    "LogSqrt2Quantizer",
    "QuantConv2d",
    "QuantLinear",
    "QuantMatMul",
    "UniformQuantizer",
    "apply_repqvit_",
]
