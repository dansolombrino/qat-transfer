"""GPTQ weight quantization, as an error-minimizing alternative to the RTN in `quantization.py`.

`apply_ptq_` rounds each weight to the nearest representable value independently. GPTQ instead
minimizes the *layer output* error ||WX - W_q X||^2 on calibration data, quantizing column by
column and pushing each column's rounding error onto the not-yet-quantized columns.

This exists to answer one question: the ranking-fragility results (F23/F24) were all measured
under RTN, and a referee will ask whether they survive a quantizer that explicitly minimizes the
error that causes rank swaps. Same bit-widths and same granularities as `quantization.py`, so the
two are directly comparable.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pretrained
Transformers" (ICLR 2023).
"""
from typing import Dict, List, Optional

import torch
from torch import nn

from src.quantization import _parse_group_size


def _fake_quant_columns(
    w: torch.Tensor, bits: int, scale: torch.Tensor
) -> torch.Tensor:
    """Round-trip a column block through the integer grid at a fixed scale. w: [out, n]."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    return (w / scale).round().clamp(qmin, qmax) * scale


def _column_scale(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric absmax scale per output row, over the given columns. Returns [out, 1]."""
    qmax = 2 ** (bits - 1) - 1
    return (w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax)


def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int,
    granularity: str,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """Return a fake-quantized copy of `weight` [out, in] using GPTQ against `hessian` [in, in].

    Granularity matches `quantization.quantize_tensor`: "channel" uses one scale per output row
    (computed once, over all columns); "group_<N>" recomputes the scale at each N-column boundary,
    which is what makes group_128 strictly finer.
    """
    W = weight.detach().clone().float()
    H = hessian.detach().clone().float()
    n_out, n_in = W.shape

    # Columns the calibration data never excited carry no information; zero them and neutralise
    # their Hessian entry so the Cholesky stays well-conditioned.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H)).clamp(min=1e-8)
    H[range(n_in), range(n_in)] += damp

    # Hinv upper-Cholesky factor; Hinv[i, i:] is the row used to redistribute column i's error.
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    if granularity == "channel":
        group = None
        scale = _column_scale(W, bits)                      # one scale per row, fixed
    elif granularity.startswith("group_"):
        group = _parse_group_size(granularity)
        if n_in % group != 0:
            raise ValueError(f"group size {group} does not divide in_features {n_in}")
        scale = None                                        # recomputed per group below
    else:
        raise ValueError(f"gptq supports 'channel' or 'group_<int>', got '{granularity}'")

    Q = torch.zeros_like(W)
    for start in range(0, n_in, blocksize):
        end = min(start + blocksize, n_in)
        W_blk = W[:, start:end].clone()
        Q_blk = torch.zeros_like(W_blk)
        E_blk = torch.zeros_like(W_blk)
        Hinv_blk = Hinv[start:end, start:end]

        for j in range(end - start):
            col = start + j
            w = W_blk[:, j]
            d = Hinv_blk[j, j]

            if group is not None and col % group == 0:
                # Scale for this group, from the current (error-compensated) columns.
                g_end = min(col + group, n_in)
                if g_end <= end:
                    src = W_blk[:, j:j + (g_end - col)]
                else:
                    src = torch.cat([W_blk[:, j:], W[:, end:g_end]], dim=1)
                scale = _column_scale(src, bits)

            q = _fake_quant_columns(w.unsqueeze(1), bits, scale).squeeze(1)
            Q_blk[:, j] = q

            err = (w - q) / d
            # Push this column's error onto the remaining columns of the block.
            W_blk[:, j:] -= err.unsqueeze(1) * Hinv_blk[j, j:].unsqueeze(0)
            E_blk[:, j] = err

        Q[:, start:end] = Q_blk
        # ...and onto every column after the block.
        if end < n_in:
            W[:, end:] -= E_blk @ Hinv[start:end, end:]

    return Q.to(weight.dtype)


class _InputCollector:
    """Accumulates H = 2 X^T X for one Linear, streaming so calibration inputs are never stored."""

    def __init__(self, n_in: int, device: torch.device):
        self.H = torch.zeros((n_in, n_in), dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).float()
        self.n += x.shape[0]
        self.H += 2.0 * (x.t() @ x)

    def finalize(self) -> torch.Tensor:
        return self.H / max(self.n, 1)


def _target_linears(
    model: nn.Module, skip_modules: frozenset[str], _prefix: str = ""
) -> List[tuple]:
    """(name, module) for every nn.Linear not under a skipped name — same traversal as apply_ptq_."""
    out = []
    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.Linear):
            out.append((full, module))
        else:
            out.extend(_target_linears(module, skip_modules, _prefix=full + "."))
    return out


@torch.no_grad()
def apply_gptq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    calib_batches: List,
    forward_fn,
    blocksize: int = 128,
    percdamp: float = 0.01,
    verbose: bool = True,
) -> List[str]:
    """Quantize every eligible nn.Linear in-place with GPTQ. Returns the names quantized.

    Layers are processed in forward order and each layer's Hessian is collected from the model as
    it stands, so layer L sees inputs produced by the *already quantized* layers before it — the
    sequential propagation the GPTQ paper specifies, not one-shot on FP activations.

    `forward_fn(model, batch)` runs one calibration batch; `calib_batches` is a list of whatever
    that function accepts.
    """
    targets = _target_linears(model, skip_modules)
    if not targets:
        return []

    # Establish forward order by observing which layer fires first.
    order: List[str] = []
    handles = []
    for name, mod in targets:
        handles.append(mod.register_forward_hook(
            lambda m, i, o, n=name: order.append(n) if n not in order else None))
    forward_fn(model, calib_batches[0])
    for h in handles:
        h.remove()
    by_name = dict(targets)
    ordered = [(n, by_name[n]) for n in order if n in by_name]
    ordered += [(n, m) for n, m in targets if n not in order]   # never-fired layers, last

    quantized: List[str] = []
    for idx, (name, mod) in enumerate(ordered):
        collector = _InputCollector(mod.in_features, mod.weight.device)
        h = mod.register_forward_pre_hook(lambda m, inp, c=collector: c.add(inp[0]))
        for batch in calib_batches:
            forward_fn(model, batch)
        h.remove()

        H = collector.finalize()
        if H.abs().sum() == 0:
            if verbose:
                print(f"  [gptq] {name}: no calibration signal, leaving FP")
            continue

        mod.weight.copy_(gptq_quantize_weight(
            mod.weight, H, bits, granularity, blocksize=blocksize, percdamp=percdamp))
        quantized.append(name)
        del collector, H
        torch.cuda.empty_cache()
        if verbose and (idx % 10 == 0 or idx == len(ordered) - 1):
            print(f"  [gptq] {idx + 1}/{len(ordered)} layers", flush=True)

    return quantized
