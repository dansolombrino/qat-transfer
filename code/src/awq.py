"""AWQ (activation-aware weight quantization), as a second error-minimizing baseline.

GPTQ compensates rounding error by pushing it onto not-yet-quantized columns. AWQ takes a
different route: it observes that a small fraction of weight channels matter far more than the
rest, and protects them by *rescaling* rather than by compensating. For a per-input-channel scale
`s`, `W X = (W diag(s)) (diag(s)^-1 X)`, so quantizing `W diag(s)` and folding `s` back gives the
salient channels effectively more of the integer grid. `s = s_x^alpha` with `s_x` the mean
activation magnitude per input channel, and alpha grid-searched per layer.

Included because the two methods differ in *mechanism*, not just in strength — a result that holds
for GPTQ should not be assumed to hold for AWQ. Same bit-widths and granularities as
`quantization.py`, so all three are directly comparable.

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration" (MLSys 2024).
"""
from typing import List

import torch
from torch import nn

from src.quantization import fake_quantize_tensor


def _layer_error(W: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """||(W - W_hat) X^T||_F^2 = trace(dW H dW^T), with H = X^T X. Avoids storing activations."""
    dW = (W - W_hat).float()
    return torch.einsum("oi,ij,oj->", dW, H, dW)


@torch.no_grad()
def awq_quantize_weight(
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    hessian: torch.Tensor,
    bits: int,
    granularity: str,
    n_grid: int = 20,
) -> torch.Tensor:
    """Fake-quantize `weight` [out, in] with the best per-layer alpha in [0, 1].

    `act_scale` is mean |x| per input channel; `hessian` is X^T X for the error objective.
    alpha = 0 reduces exactly to RTN, so AWQ can never be worse than RTN on this objective.
    """
    W = weight.detach().float()
    sx = act_scale.detach().float().clamp(min=1e-8)

    best, best_W, best_alpha = None, None, 0.0
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = sx.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()          # keep the scale centred, avoids drift
        W_scaled = W * s.unsqueeze(0)
        Q = fake_quantize_tensor(W_scaled, bits, granularity)
        W_hat = Q / s.unsqueeze(0)
        err = _layer_error(W, W_hat, hessian)
        if best is None or err < best:
            best, best_W, best_alpha = err, W_hat, alpha
    return best_W.to(weight.dtype), best_alpha


class _ActCollector:
    """Streams mean |x| per input channel and H = X^T X, without retaining activations."""

    def __init__(self, n_in: int, device: torch.device):
        self.absum = torch.zeros(n_in, dtype=torch.float32, device=device)
        self.H = torch.zeros((n_in, n_in), dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).float()
        self.absum += x.abs().sum(0)
        self.H += x.t() @ x
        self.n += x.shape[0]

    def finalize(self):
        n = max(self.n, 1)
        return self.absum / n, self.H / n


def _target_linears(model: nn.Module, skip_modules: frozenset, _prefix: str = "") -> List[tuple]:
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
def apply_awq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset,
    calib_batches: List,
    forward_fn,
    only: frozenset | None = None,
    n_grid: int = 20,
    verbose: bool = True,
) -> List[str]:
    """Quantize every eligible nn.Linear in-place with AWQ. Returns the names quantized.

    Layers are processed in forward order with sequential propagation, matching `apply_gptq_`, so
    the two differ only in the quantization rule and the comparison is clean.
    """
    targets = _target_linears(model, skip_modules)
    if only is not None:
        # `skip_modules` is matched against LOCAL child names; a per-layer bit allocation is
        # keyed by full dotted paths, so it cannot be expressed through it. Filter explicitly.
        targets = [(n, m) for n, m in targets if n in only]
    if not targets:
        return []

    order: List[str] = []
    handles = [m.register_forward_hook(
        lambda mod, i, o, n=n: order.append(n) if n not in order else None)
        for n, m in targets]
    forward_fn(model, calib_batches[0])
    for h in handles:
        h.remove()
    by_name = dict(targets)
    ordered = [(n, by_name[n]) for n in order if n in by_name]
    ordered += [(n, m) for n, m in targets if n not in order]

    quantized, alphas = [], []
    for idx, (name, mod) in enumerate(ordered):
        col = _ActCollector(mod.in_features, mod.weight.device)
        h = mod.register_forward_pre_hook(lambda m, inp, c=col: c.add(inp[0]))
        for batch in calib_batches:
            forward_fn(model, batch)
        h.remove()

        sx, H = col.finalize()
        if sx.abs().sum() == 0:
            if verbose:
                print(f"  [awq] {name}: no calibration signal, leaving FP")
            continue
        W_hat, alpha = awq_quantize_weight(mod.weight, sx, H, bits, granularity, n_grid=n_grid)
        mod.weight.copy_(W_hat)
        quantized.append(name)
        alphas.append(alpha)
        del col, H, sx
        torch.cuda.empty_cache()
        if verbose and (idx % 20 == 0 or idx == len(ordered) - 1):
            print(f"  [awq] {idx + 1}/{len(ordered)} layers (alpha median "
                  f"{torch.tensor(alphas).median():.2f})", flush=True)

    if verbose and alphas:
        a = torch.tensor(alphas)
        print(f"  [awq] alpha: median {a.median():.2f}, mean {a.mean():.2f}, "
              f"{(a == 0).float().mean() * 100:.0f}% of layers fell back to RTN (alpha=0)")
    return quantized
