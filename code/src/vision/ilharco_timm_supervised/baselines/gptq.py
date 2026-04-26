"""
GPTQ: Accurate Post-Training Quantization via Hessian-based error compensation.

Adapted from the reference implementation at references/gptq/ (Frantar et al., 2023).
Simplified for vision models: only nn.Linear support, no groupsize/actorder.
"""

import logging
import math
import time

import torch
import torch.nn as nn

from typing import List

log = logging.getLogger(__name__)


# =============================================================================
# Per-column symmetric fake-quantize (used inside the GPTQ loop)
# =============================================================================


def _fake_quantize_column_symmetric(
    w: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Fake-quantize a single column vector w using pre-computed per-row scales.

    Args:
        w: column vector, shape [out_features]
        scale: per-row scale, shape [out_features] (channelwise)
               or scalar tensor (tensorwise)
        bits: quantization bit-width

    Returns:
        Fake-quantized column, shape [out_features]
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    int_w = (w / scale).round().clamp(qmin, qmax)
    return int_w * scale


# =============================================================================
# GPTQ class — per-layer Hessian accumulation and column-wise quantization
# =============================================================================


class GPTQ:
    """
    GPTQ quantizer for a single nn.Linear layer.

    Usage:
        gptq = GPTQ(layer)
        # ... run calibration batches, calling gptq.add_batch() via hook ...
        gptq.quantize(bits, granularity, sym)
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]     # out_features
        self.columns = W.shape[1]  # in_features
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Accumulate Hessian from a batch of layer inputs.

        Called as a forward hook: inp is the input tensor to the layer.
        For nn.Linear, inp has shape (batch, *, in_features).
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def quantize(
        self,
        bits: int,
        granularity: str,
        sym: bool,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ):
        """
        Run GPTQ column-wise quantization with error compensation.

        Modifies self.layer.weight.data in-place.

        Args:
            bits: target bit-width
            granularity: "tensor" or "channel"
            sym: if True, use symmetric quantization; if False, raise NotImplementedError
            blocksize: number of columns to process per block
            percdamp: Hessian damping as fraction of diagonal mean
        """
        if not sym:
            raise NotImplementedError(
                "Asymmetric quantization is not yet supported in GPTQ. "
                "Use sym=True for symmetric quantization."
            )

        W = self.layer.weight.data.clone().float()
        tick = time.time()

        # Compute per-row (channelwise) or global (tensorwise) scales upfront
        if granularity == "channel":
            qmax = 2 ** (bits - 1) - 1
            abs_max = W.abs().amax(dim=1).clamp(min=1e-8)
            scale = abs_max / qmax  # shape [out_features]
        elif granularity == "tensor":
            qmax = 2 ** (bits - 1) - 1
            abs_max = W.abs().max().clamp(min=1e-8)
            scale = abs_max / qmax  # scalar
        else:
            raise ValueError(
                f"granularity expected to be 'tensor' or 'channel', got '{granularity}'"
            )

        H = self.H
        del self.H

        # Handle dead columns (zero diagonal in Hessian)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Damp Hessian and compute Cholesky inverse
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = _fake_quantize_column_symmetric(w, scale, bits)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        elapsed = time.time() - tick
        total_loss = torch.sum(Losses).item()
        log.info(
            "GPTQ layer %s: time=%.2fs, error=%.6f",
            getattr(self, '_layer_name', '?'), elapsed, total_loss,
        )

        self.layer.weight.data = Q.to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# =============================================================================
# Model-level GPTQ entry point (drop-in alternative to apply_ptq_)
# =============================================================================


def _find_linear_layers(
    model: nn.Module,
    skip_modules: frozenset,
    prefix: str = "",
) -> dict:
    """
    Recursively find all nn.Linear layers, skipping modules in skip_modules.

    Returns dict mapping fully-qualified name → nn.Linear module.
    """
    result = {}
    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{prefix}{name}"
        if isinstance(module, nn.Linear):
            result[full] = module
        else:
            result.update(
                _find_linear_layers(module, skip_modules, prefix=full + ".")
            )
    return result


def apply_gptq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    sym: bool,
    skip_modules: frozenset[str],
    calibration_loader,
    num_calibration_samples: int = 128,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> List[str]:
    """
    Apply GPTQ post-training quantization in-place.

    Uses calibration data to compute per-layer Hessians, then performs
    column-wise weight quantization with error compensation.

    Same interface as apply_ptq_ (returns list of quantized layer names),
    but requires a calibration_loader and num_calibration_samples.

    Args:
        model: the model to quantize (mutated in-place)
        bits: target bit-width
        granularity: "tensor" or "channel"
        sym: if True, use symmetric quantization
        skip_modules: frozenset of child module names to skip
        calibration_loader: DataLoader yielding (images, labels) or dicts
        num_calibration_samples: number of samples to use for calibration
        blocksize: GPTQ block size for column processing
        percdamp: Hessian damping factor

    Returns:
        List of fully-qualified names of quantized nn.Linear layers.
    """
    device = next(model.parameters()).device

    # Step 1: Find all target Linear layers
    linear_layers = _find_linear_layers(model, skip_modules)
    if not linear_layers:
        return []

    # Step 2: Create GPTQ instances and register hooks
    gptq_instances = {}
    hooks = []

    for name, layer in linear_layers.items():
        gptq_obj = GPTQ(layer)
        gptq_obj._layer_name = name
        gptq_instances[name] = gptq_obj

        def make_hook(gptq_obj):
            def hook_fn(module, inp, out):
                gptq_obj.add_batch(inp[0].data, out.data)
            return hook_fn

        handle = layer.register_forward_hook(make_hook(gptq_obj))
        hooks.append(handle)

    # Step 3: Run calibration forward passes
    model.eval()
    samples_seen = 0

    with torch.no_grad():
        for batch in calibration_loader:
            if samples_seen >= num_calibration_samples:
                break

            if isinstance(batch, dict):
                images = batch['images'].to(device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)

            model(images)
            samples_seen += images.shape[0]

    # Step 4: Remove hooks
    for handle in hooks:
        handle.remove()

    # Step 5: Run GPTQ quantization on each layer
    for name, gptq_obj in gptq_instances.items():
        gptq_obj.quantize(
            bits=bits,
            granularity=granularity,
            sym=sym,
            blocksize=blocksize,
            percdamp=percdamp,
        )
        gptq_obj.free()

    return list(linear_layers.keys())
