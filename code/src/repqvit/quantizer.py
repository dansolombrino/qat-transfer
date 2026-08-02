"""Official RepQ-ViT quantizers, adapted and numerically hardened.

Derived from ``classification/quant/quantizer.py`` in the official RepQ-ViT
release (Apache-2.0).  Modifications: vectorized per-channel calibration,
non-persistent PyTorch buffers for calibrated parameters, explicit validation,
and finite behavior for constant/zero tensors.
"""

from __future__ import annotations

from math import ceil, floor, sqrt

import torch
from torch import nn


_PERCENTILES = (0.999, 0.9999, 0.99999)


def _quantile_from_sorted(sorted_x: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    """Linear-interpolation quantile of an already-sorted tensor.

    ``torch.quantile`` rejects inputs with more than 2**24 elements, which every
    real calibration batch exceeds (a 32x197x3072 activation is 19.4M values), so
    percentile calibration cannot call it.  Sorting once per calibration and
    indexing is exact and reproduces torch.quantile's default ``linear``
    interpolation, so calibrated scales are unchanged on the small tensors where
    both are usable.
    """

    n = sorted_x.shape[dim]
    position = q * (n - 1)
    lower_index = int(floor(position))
    upper_index = int(ceil(position))
    fraction = position - lower_index
    lower = sorted_x.select(dim, lower_index)
    upper = sorted_x.select(dim, upper_index)
    return lower + (upper - lower) * fraction


def _as_channel_rows(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return channel-major rows and the broadcast shape for qparams."""

    if x.ndim == 2:  # weights: (out, in)
        return x.detach().reshape(x.shape[0], -1), (x.shape[0], 1)
    if x.ndim == 3:  # activations: (batch, tokens, channels)
        return x.detach().permute(2, 0, 1).reshape(x.shape[-1], -1), (
            1,
            1,
            x.shape[-1],
        )
    if x.ndim == 4:  # convolution weights: (out, in, kh, kw)
        return x.detach().reshape(x.shape[0], -1), (x.shape[0], 1, 1, 1)
    raise NotImplementedError(
        f"RepQ-ViT channel-wise quantization does not support shape {tuple(x.shape)}"
    )


def _safe_affine_qparams(
    lower: torch.Tensor,
    upper: torch.Tensor,
    n_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build affine qparams while preserving constant tensors exactly."""

    width = upper - lower
    eps = torch.finfo(lower.dtype).eps
    constant = width.abs() <= eps
    delta = torch.where(constant, torch.ones_like(width), width / (n_levels - 1))
    zero_point = torch.where(constant, -lower, (-lower / delta).round())
    return delta, zero_point


def _fake_quantize_affine(
    x: torch.Tensor,
    delta: torch.Tensor,
    zero_point: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    x_int = torch.round(x / delta) + zero_point
    x_quant = torch.clamp(x_int, 0, n_levels - 1)
    return (x_quant - zero_point) * delta


class UniformQuantizer(nn.Module):
    """Asymmetric uniform quantizer used for RepQ-ViT weights/activations."""

    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super().__init__()
        if not 2 <= n_bits <= 8:
            raise ValueError(f"RepQ-ViT supports 2-8 bits, got {n_bits}")
        self.n_bits = int(n_bits)
        self.n_levels = 2**self.n_bits
        self.channel_wise = bool(channel_wise)
        self.inited = False
        self.register_buffer("delta", None, persistent=False)
        self.register_buffer("zero_point", None, persistent=False)

    def extra_repr(self) -> str:
        return f"bits={self.n_bits}, inited={self.inited}, channel_wise={self.channel_wise}"

    def reset(self) -> None:
        self.delta = None
        self.zero_point = None
        self.inited = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inited:
            self.delta, self.zero_point = self.init_quantization_scale(
                x, channel_wise=self.channel_wise
            )
            self.inited = True
        return _fake_quantize_affine(x, self.delta, self.zero_point, self.n_levels)

    def init_quantization_scale(
        self,
        x: torch.Tensor,
        channel_wise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_floating_point():
            raise TypeError(f"Expected floating-point tensor, got {x.dtype}")

        if channel_wise:
            rows, broadcast_shape = _as_channel_rows(x)
            best_score = torch.full(
                (rows.shape[0],), float("inf"), device=x.device, dtype=x.dtype
            )
            best_delta = torch.ones_like(best_score)
            best_zero = torch.zeros_like(best_score)

            sorted_rows, _ = torch.sort(rows, dim=1)
            for pct in _PERCENTILES:
                upper = _quantile_from_sorted(sorted_rows, pct, dim=1)
                lower = _quantile_from_sorted(sorted_rows, 1.0 - pct, dim=1)
                delta, zero = _safe_affine_qparams(lower, upper, self.n_levels)
                quant = _fake_quantize_affine(
                    rows, delta[:, None], zero[:, None], self.n_levels
                )
                score = (rows - quant).square().mean(dim=1)
                improve = score < best_score
                best_score = torch.where(improve, score, best_score)
                best_delta = torch.where(improve, delta, best_delta)
                best_zero = torch.where(improve, zero, best_zero)

            return best_delta.reshape(broadcast_shape), best_zero.reshape(
                broadcast_shape
            )

        flat = x.detach().reshape(-1)
        best_score = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        best_delta = torch.ones((), device=x.device, dtype=x.dtype)
        best_zero = torch.zeros((), device=x.device, dtype=x.dtype)
        sorted_flat, _ = torch.sort(flat)
        for pct in _PERCENTILES:
            upper = _quantile_from_sorted(sorted_flat, pct, dim=0)
            lower = _quantile_from_sorted(sorted_flat, 1.0 - pct, dim=0)
            delta, zero = _safe_affine_qparams(lower, upper, self.n_levels)
            quant = _fake_quantize_affine(flat, delta, zero, self.n_levels)
            score = (flat - quant).square().mean()
            if score < best_score:
                best_score = score
                best_delta = delta
                best_zero = zero
        return best_delta, best_zero


class LogSqrt2Quantizer(nn.Module):
    """Log-sqrt(2) quantizer for RepQ-ViT post-Softmax activations."""

    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super().__init__()
        if not 2 <= n_bits <= 8:
            raise ValueError(f"RepQ-ViT supports 2-8 bits, got {n_bits}")
        if channel_wise:
            raise ValueError("LogSqrt2Quantizer is tensor-wise in RepQ-ViT")
        self.n_bits = int(n_bits)
        self.n_levels = 2**self.n_bits
        self.inited = False
        self.register_buffer("delta", None, persistent=False)

    def extra_repr(self) -> str:
        return f"bits={self.n_bits}, inited={self.inited}"

    def reset(self) -> None:
        self.delta = None
        self.inited = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inited:
            self.delta = self.init_quantization_scale(x)
            self.inited = True
        return self.quantize(x, self.delta)

    def init_quantization_scale(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(x < 0):
            raise ValueError("Post-Softmax RepQ-ViT quantizer received negative values")
        flat = x.detach().reshape(-1)
        best_score = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        best_delta = torch.ones((), device=x.device, dtype=x.dtype)
        sorted_flat, _ = torch.sort(flat)
        for pct in _PERCENTILES:
            candidate = _quantile_from_sorted(sorted_flat, pct, dim=0)
            candidate = torch.where(
                candidate > 0, candidate, torch.ones_like(candidate)
            )
            quant = self.quantize(flat, candidate)
            score = (flat - quant).square().mean()
            if score < best_score:
                best_score = score
                best_delta = candidate
        return best_delta

    def quantize(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        positive = x > 0
        safe_ratio = torch.where(positive, x / delta, torch.ones_like(x))
        x_int = torch.round(-safe_ratio.log2() * 2)
        underflow = (~positive) | (x_int >= self.n_levels)
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_scale = (x_quant.remainder(2)) * (sqrt(2.0) - 1.0) + 1.0
        result = 2.0 ** (-torch.ceil(x_quant / 2.0)) * odd_scale * delta
        return torch.where(underflow, torch.zeros_like(result), result)
