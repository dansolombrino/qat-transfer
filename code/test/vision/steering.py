"""CPU smoke test for src/vision/steering.py (no checkpoints, no GPU, no timm).

Validates the activation-space steering primitives used by 003_qat_transfer_activ:
  * select_tap_modules finds per-block taps and rejects unimplemented strategies,
  * ActivationMeanCapture returns [tokens, dim] per-token means that match a
    manual reference and are invariant to batching,
  * reduce_steering's mean/per_token identity (per-token is a superset of mean),
  * ActivationInjector adds exactly alpha * reduce(S) to the tapped output.

Run: uv run --active python code/test/vision/steering.py
"""

import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from src.vision.steering import (
    select_tap_modules,
    ActivationMeanCapture,
    ActivationInjector,
    reduce_steering,
)


# Toy modules that mimic the timm naming the per_block tap expects:
# classifier.model.blocks.{i}, each block emitting a [B, N, D] tensor.
class ToyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):  # [B, N, D] -> [B, N, D], per-sample independent
        return self.lin(x)


class ToyBackbone(nn.Module):
    def __init__(self, dim, depth, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, 0])


class ToyClassifier(nn.Module):
    def __init__(self, dim=8, depth=3, num_classes=5):
        super().__init__()
        self.model = ToyBackbone(dim, depth, num_classes)

    def forward(self, x):
        return self.model(x)


def test_select_tap_modules():
    clf = ToyClassifier(dim=8, depth=3)
    taps = select_tap_modules(clf, "per_block")
    names = list(taps.keys())
    assert names == ["model.blocks.0", "model.blocks.1", "model.blocks.2"], names
    for name, mod in taps.items():
        assert isinstance(mod, ToyBlock)

    for strat in ("per_linear", "per_attn_mlp"):
        try:
            select_tap_modules(clf, strat)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{strat} should raise NotImplementedError")

    try:
        select_tap_modules(clf, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown strategy should raise ValueError")

    print("PASS  select_tap_modules (per_block names, stub + unknown rejection)")


def test_capture_matches_manual_mean():
    torch.manual_seed(0)
    dim, depth, N = 8, 3, 4
    clf = ToyClassifier(dim=dim, depth=depth).eval()
    taps = select_tap_modules(clf, "per_block")

    batches = [torch.randn(b, N, dim) for b in (3, 5, 2)]
    full = torch.cat(batches, dim=0)

    # Manual reference: per-token mean of each block output over a single pass.
    ref = {}
    handles = []
    for name, mod in taps.items():
        handles.append(mod.register_forward_hook(
            lambda m, i, o, n=name: ref.__setitem__(n, o.detach())
        ))
    with torch.no_grad():
        clf(full)
    for h in handles:
        h.remove()
    ref_mean = {n: ref[n].mean(dim=0) for n in ref}

    # Capture accumulates across mini-batches.
    with ActivationMeanCapture(taps) as cap:
        with torch.no_grad():
            for b in batches:
                clf(b)
    got = cap.result()

    assert cap.num_samples == full.shape[0], (cap.num_samples, full.shape[0])
    for name in taps:
        assert got[name].shape == (N, dim), got[name].shape
        assert torch.allclose(got[name], ref_mean[name], atol=1e-5), name

    print("PASS  ActivationMeanCapture (shape, batching-invariant per-token mean)")


def test_reduce_identity():
    torch.manual_seed(1)
    N, dim = 4, 8
    v = torch.randn(N, dim)
    assert torch.equal(reduce_steering(v, "per_token"), v)
    assert torch.allclose(reduce_steering(v, "mean"), v.mean(dim=0))
    # per-token is a superset of mean: mean-over-tokens recovers the [dim] vector.
    assert torch.allclose(reduce_steering(v, "per_token").mean(dim=0),
                          reduce_steering(v, "mean"), atol=1e-6)
    print("PASS  reduce_steering (mean/per_token + superset identity)")


def test_injector_adds_alpha_times_steering():
    torch.manual_seed(2)
    dim, N, B = 8, 4, 6
    alpha = 2.0
    block = ToyBlock(dim).eval()
    taps = OrderedDict([("blk", block)])
    x = torch.randn(B, N, dim)

    with torch.no_grad():
        base = block(x)

    steer = {"blk": torch.randn(N, dim)}
    for reduce in ("mean", "per_token"):
        with ActivationInjector(taps, steer, alpha=alpha, token_reduce=reduce):
            with torch.no_grad():
                y = block(x)  # hook modifies the returned output in place
        expected = base + alpha * reduce_steering(steer["blk"], reduce)
        assert y.shape == base.shape
        assert torch.allclose(y, expected, atol=1e-5), reduce

    # per_token token-count mismatch must raise.
    bad = {"blk": torch.randn(N + 1, dim)}
    try:
        with ActivationInjector(taps, bad, alpha=1.0, token_reduce="per_token"):
            block(x)
    except ValueError:
        pass
    else:
        raise AssertionError("per_token token mismatch should raise ValueError")

    print("PASS  ActivationInjector (adds alpha*reduce(S); per_token shape guard)")


def main():
    test_select_tap_modules()
    test_capture_matches_manual_mean()
    test_reduce_identity()
    test_injector_adds_alpha_times_steering()
    print("\nALL STEERING SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
