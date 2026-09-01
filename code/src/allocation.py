"""Bit allocation driven by gap sensitivity rather than reconstruction error.

Existing weight-only PTQ methods choose *how* to round (GPTQ compensates against the activation
covariance, AWQ rescales outlier channels) but spend their bit budget uniformly across layers, or
allocate it to minimise per-layer reconstruction error. Corollary 5 says neither is the operative
quantity: whether the top-1 survives depends on the perturbation of the single scalar
`z_(1) - z_(2)`, and bounding that through a norm on the whole score vector costs a factor of two.

This module measures, per layer, how much quantizing *that layer alone* perturbs the top-1/top-2
gap, and allocates finer quantization granularity to the layers that matter by that measure. The
budget is held fixed: granularity is traded between layers so the average number of scales per
weight is unchanged, hence so is the memory and the inference cost.

Nothing here trains anything. It is an allocation policy on top of an existing quantizer.
"""
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from src.quantization import fake_quantize_tensor


def _linears(model: nn.Module, skip: frozenset, prefix: str = "") -> List[Tuple[str, nn.Linear]]:
    out = []
    for name, mod in model.named_children():
        if name in skip:
            continue
        full = f"{prefix}{name}"
        if isinstance(mod, nn.Linear):
            out.append((full, mod))
        else:
            out.extend(_linears(mod, skip, full + "."))
    return out


@torch.no_grad()
def gap_sensitivity(model: nn.Module, skip: frozenset, bits: int, granularity: str,
                    score_fn: Callable[[], np.ndarray], verbose: bool = True,
                    tail_frac: float | None = None):
    """Per-layer sensitivity of the top-1/top-2 gap to quantizing that layer alone.

    `score_fn()` returns a (n, C) array of scores for the calibration batch under the model's
    current weights. For each layer we quantize it, recompute, and record the median absolute
    change in the top-1/top-2 gap of the *unquantized* ranking -- the quantity Cor. 5 identifies.
    """
    base = score_fn()
    order = np.argsort(-base, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    rows = np.arange(len(base))
    base_gap = base[rows, top1] - base[rows, top2]
    # Optional tail restriction: only inputs in the lowest-gap quantile inform the estimate.
    # These are the inputs whose fate an allocation can actually change; on tasks where most
    # inputs sit far above the flip threshold, the plain median measures bulk gap motion among
    # inputs that were never at risk. Returned as a second dict so one sweep yields both.
    tail_rows = None
    if tail_frac is not None:
        tail_rows = base_gap <= np.quantile(base_gap, tail_frac)

    sens: Dict[str, float] = {}
    sens_tail: Dict[str, float] = {}
    layers = _linears(model, skip)
    for i, (name, mod) in enumerate(layers):
        orig = mod.weight.detach().clone()
        mod.weight.copy_(fake_quantize_tensor(orig, bits, granularity))
        s = score_fn()
        gap = s[rows, top1] - s[rows, top2]          # same pair, by identity
        d = np.abs(gap - base_gap)
        sens[name] = float(np.median(d))
        if tail_rows is not None:
            sens_tail[name] = float(np.median(d[tail_rows]))
        mod.weight.copy_(orig)                        # restore before the next layer
        if verbose and (i % 20 == 0 or i == len(layers) - 1):
            print(f"  [sens] {i+1}/{len(layers)}", flush=True)
    if tail_frac is not None:
        return sens, sens_tail
    return sens


@torch.no_grad()
def mse_sensitivity(model: nn.Module, skip: frozenset, bits: int,
                    granularity: str) -> Dict[str, float]:
    """Per-layer reconstruction error -- the objective existing allocation schemes use."""
    out = {}
    for name, mod in _linears(model, skip):
        w = mod.weight.detach()
        out[name] = float((w - fake_quantize_tensor(w, bits, granularity)).pow(2).mean())
    return out


def allocate(sens: Dict[str, float], sizes: List[int], numel: Dict[str, int],
             base_group: int) -> Dict[str, int]:
    """Assign a group size per layer, holding the scale budget equal to uniform `base_group`.

    A group size of g stores one scale per g weights, so the scale cost of a layer is
    numel/g. Budget is sum(numel)/base_group. Layers are sorted by sensitivity per weight and
    the finest granularity is spent on the most sensitive until the budget is exhausted.
    """
    budget = sum(numel.values()) / base_group
    ranked = sorted(sens, key=lambda k: sens[k] / max(numel[k], 1), reverse=True)
    alloc = {k: max(sizes) for k in sens}             # start everyone at the cheapest
    cost = sum(numel[k] / max(sizes) for k in sens)
    for g in sorted(sizes):                            # finest first
        for k in ranked:
            if alloc[k] <= g:
                continue
            delta = numel[k] / g - numel[k] / alloc[k]
            if cost + delta <= budget:
                cost += delta
                alloc[k] = g
    return alloc


@torch.no_grad()
def apply_allocation_(model: nn.Module, alloc: Dict[str, int], bits: int) -> int:
    """Quantize each layer at its assigned group size."""
    n = 0
    for name, mod in _linears(model, frozenset()):
        if name not in alloc:
            continue
        g = alloc[name]
        gran = "channel" if g <= 0 else f"group_{g}"
        mod.weight.copy_(fake_quantize_tensor(mod.weight, bits, gran))
        n += 1
    return n


def allocate_bits(sens: Dict[str, float], numel: Dict[str, int], bit_choices: List[int],
                  avg_bits: float) -> Dict[str, int]:
    """Assign a bit-width per layer holding the average bits per weight fixed.

    Group size turned out to be a poor lever: coarsening a layer costs far more than refining
    another gains, so any redistribution loses to uniform. Bit-width is the standard
    mixed-precision lever and is closer to linear in damage. Layers are ranked by sensitivity per
    weight and the highest bit-width is spent on the most sensitive until the budget runs out.
    """
    total = sum(numel.values())
    budget = total * avg_bits
    lo = min(bit_choices)
    alloc = {k: lo for k in sens}
    cost = total * lo
    ranked = sorted(sens, key=lambda k: sens[k] / max(numel[k], 1), reverse=True)
    for b in sorted(bit_choices, reverse=True):
        for k in ranked:
            if alloc[k] >= b:
                continue
            delta = numel[k] * (b - alloc[k])
            if cost + delta <= budget:
                cost += delta
                alloc[k] = b
    return alloc


@torch.no_grad()
def apply_bit_allocation_(model: nn.Module, alloc: Dict[str, int], granularity: str) -> int:
    """Quantize each layer at its assigned bit-width, all at the same granularity."""
    n = 0
    for name, mod in _linears(model, frozenset()):
        if name not in alloc:
            continue
        mod.weight.copy_(fake_quantize_tensor(mod.weight, alloc[name], granularity))
        n += 1
    return n


def hawq_sensitivity(st_model, inner: nn.Module, skip: frozenset, bits: int, granularity: str,
                     calib_queries, calib_docs, batch_size: int = 16,
                     verbose: bool = True) -> Dict[str, float]:
    """HAWQ-V2-style curvature sensitivity: Fisher-trace estimate times quantization noise.

    sens(l) = (1/n_l) * sum_i ||grad_l L_i||^2  *  ||Q(W_l) - W_l||_F^2

    The Fisher trace approximates the Hessian trace of the task loss (the standard
    empirical-Fisher surrogate HAWQ implementations use); the loss is in-batch InfoNCE between
    calibration queries and their paired documents, so no labels are needed. This is the
    curvature-weighted output-error criterion of the mixed-precision literature, included as a
    baseline: like reconstruction error it targets output error, not ranking gaps.
    """
    device = next(inner.parameters()).device
    layers = dict(_linears(inner, skip))
    fisher = {n: 0.0 for n in layers}
    was_training = st_model.training
    st_model.eval()
    for p in inner.parameters():
        p.requires_grad_(True)
    n_batches = 0
    for a in range(0, min(len(calib_queries), len(calib_docs)), batch_size):
        q = calib_queries[a:a + batch_size]
        d = calib_docs[a:a + batch_size]
        if len(q) < 2 or len(q) != len(d):
            continue
        fq = st_model.tokenize(q)
        fd = st_model.tokenize(d)
        fq = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fq.items()}
        fd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fd.items()}
        eq = torch.nn.functional.normalize(st_model(fq)["sentence_embedding"], dim=-1)
        ed = torch.nn.functional.normalize(st_model(fd)["sentence_embedding"], dim=-1)
        logits = (eq @ ed.T) / 0.05
        labels = torch.arange(len(q), device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        inner.zero_grad(set_to_none=True)
        loss.backward()
        for n, mod in layers.items():
            if mod.weight.grad is not None:
                fisher[n] += float(mod.weight.grad.detach().pow(2).sum())
        n_batches += 1
        if verbose and n_batches % 4 == 0:
            print(f"  [hawq] batch {n_batches}", flush=True)
    inner.zero_grad(set_to_none=True)
    for p in inner.parameters():
        p.requires_grad_(False)
    if was_training:
        st_model.train()

    out: Dict[str, float] = {}
    with torch.no_grad():
        for n, mod in layers.items():
            w = mod.weight
            noise = float((w - fake_quantize_tensor(w, bits, granularity)).pow(2).sum())
            out[n] = (fisher[n] / max(w.numel(), 1)) * noise
    return out


def hawq_sensitivity_clsf(inner: nn.Module, skip: frozenset, bits: int, granularity: str,
                          batches, forward_fn, verbose: bool = True) -> Dict[str, float]:
    """HAWQ-V2-style sensitivity for classifiers: Fisher trace of the cross-entropy task loss
    times quantization noise. `forward_fn(batch)` returns (logits, labels). This is the faithful
    HAWQ setting -- curvature of the actual task loss the method was designed around."""
    layers = dict(_linears(inner, skip))
    fisher = {n: 0.0 for n in layers}
    for p in inner.parameters():
        p.requires_grad_(True)
    nb = 0
    for batch in batches:
        logits, labels = forward_fn(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        inner.zero_grad(set_to_none=True)
        loss.backward()
        for n, mod in layers.items():
            if mod.weight.grad is not None:
                fisher[n] += float(mod.weight.grad.detach().pow(2).sum())
        nb += 1
        if verbose and nb % 4 == 0:
            print(f"  [hawq] batch {nb}", flush=True)
    inner.zero_grad(set_to_none=True)
    for p in inner.parameters():
        p.requires_grad_(False)
    out: Dict[str, float] = {}
    with torch.no_grad():
        for n, mod in layers.items():
            w = mod.weight
            noise = float((w - fake_quantize_tensor(w, bits, granularity)).pow(2).sum())
            out[n] = (fisher[n] / max(w.numel(), 1)) * noise
    return out


def activation_stats(model: nn.Module, skip: frozenset, forward_fn: Callable[[], object]):
    """Per-input-channel activation moments for every target linear, from one forward pass.

    Returns {layer: (E[x^2], E|x|)} as tensors of shape (in_features,). These are the statistics
    the activation-aware quantizers are built on: GPTQ's Hessian is 2 X^T X, so its diagonal is
    n * E[x^2], and AWQ's salience is a function of E|x| per channel.
    """
    layers = dict(_linears(model, skip))
    acc = {n: None for n in layers}
    cnt = {n: 0 for n in layers}

    def mk(name):
        def hook(_m, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            sq = x.pow(2).sum(0)
            ab = x.abs().sum(0)
            if acc[name] is None:
                acc[name] = [sq, ab]
            else:
                acc[name][0] += sq
                acc[name][1] += ab
            cnt[name] += x.shape[0]
        return hook

    handles = [layers[n].register_forward_hook(mk(n)) for n in layers]
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for h in handles:
            h.remove()
    out = {}
    for n in layers:
        if acc[n] is None or cnt[n] == 0:
            continue
        out[n] = (acc[n][0] / cnt[n], acc[n][1] / cnt[n])
    return out


def hessian_sensitivity(model: nn.Module, skip: frozenset, bits: int, granularity: str,
                        stats: Dict[str, tuple]) -> Dict[str, float]:
    """GPTQ/OBQ-style: the layerwise proxy loss those methods minimise, used to rank layers.

    sens(l) = || X (W - Q(W))^T ||_F^2 / n  =  sum_i E[x_i^2] * || dW[:, i] ||^2

    This is reconstruction error measured in *activation* space rather than weight space, i.e.
    output error weighted by the diagonal of the layer Hessian. It is the natural allocation
    criterion implied by GPTQ, and unlike HAWQ it needs no backward pass.
    """
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, mod in _linears(model, skip):
            if name not in stats:
                continue
            w = mod.weight.detach()
            dw = (w - fake_quantize_tensor(w, bits, granularity)).float()
            h_diag = stats[name][0].to(dw.device)
            out[name] = float((dw.pow(2).sum(0) * h_diag).sum())
    return out


def awq_salience_sensitivity(model: nn.Module, skip: frozenset, bits: int, granularity: str,
                             stats: Dict[str, tuple]) -> Dict[str, float]:
    """AWQ-style: quantization error weighted by per-channel activation magnitude.

    AWQ's premise is that channels with large activations are salient and must be protected.
    Lifted to a layer-level allocation criterion, that is sum_i E|x_i| * ||dW[:, i]||^2.
    """
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, mod in _linears(model, skip):
            if name not in stats:
                continue
            w = mod.weight.detach()
            dw = (w - fake_quantize_tensor(w, bits, granularity)).float()
            a = stats[name][1].to(dw.device)
            out[name] = float((dw.pow(2).sum(0) * a).sum())
    return out


def relative_error_sensitivity(model: nn.Module, skip: frozenset, bits: int,
                               granularity: str) -> Dict[str, float]:
    """Scale-free weight-space error: ||dW||^2 / ||W||^2, the SQNR view. Data-free."""
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, mod in _linears(model, skip):
            w = mod.weight.detach().float()
            dw = w - fake_quantize_tensor(mod.weight.detach(), bits, granularity).float()
            out[name] = float(dw.pow(2).sum() / max(float(w.pow(2).sum()), 1e-12))
    return out


def position_sensitivity(model: nn.Module, skip: frozenset) -> Dict[str, float]:
    """The deployment folk rule: keep the first and last blocks at higher precision.

    Scores layers by distance from the middle of the stack, so the extremes rank highest.
    Data-free and weight-free; included to test whether the ranking signal is anything more
    than depth.
    """
    import re
    names = [n for n, _ in _linears(model, skip)]
    idx = {}
    for n in names:
        m = re.search(r"\.(\d+)\.", n)
        idx[n] = int(m.group(1)) if m else 0
    if not idx:
        return {}
    lo, hi = min(idx.values()), max(idx.values())
    mid = (lo + hi) / 2.0
    span = max(hi - lo, 1) / 2.0
    return {n: abs(idx[n] - mid) / span for n in names}


def fisher_only_sensitivity(fisher_sens: Dict[str, float], model: nn.Module, skip: frozenset,
                            bits: int, granularity: str) -> Dict[str, float]:
    """HAWQ's curvature term with the quantization-noise factor divided out.

    Isolates whether the curvature route's signal comes from the Hessian estimate itself or
    from the noise weighting it is multiplied by.
    """
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, mod in _linears(model, skip):
            if name not in fisher_sens:
                continue
            w = mod.weight
            noise = float((w - fake_quantize_tensor(w, bits, granularity)).pow(2).sum())
            out[name] = fisher_sens[name] / max(noise, 1e-30)
    return out


def act_norm_sensitivity(stats: Dict[str, tuple]) -> Dict[str, float]:
    """Activation energy alone: sum_i E[x_i^2], ignoring the weights entirely.

    Tests whether data-awareness by itself is enough to rank layers, or whether the criterion
    has to combine it with what quantization actually does to those weights.
    """
    return {n: float(v[0].sum()) for n, v in stats.items()}
