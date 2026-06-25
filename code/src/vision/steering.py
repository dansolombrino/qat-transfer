"""Activation-space steering utilities shared by 003_qat_transfer_activ.

Single source of truth for:
  * which modules are "tapped" for a given ``steering_strategy`` (so the tap
    names used as cache keys, hook targets, and result fields never drift), and
  * the forward-hook machinery to (a) accumulate per-token activation means
    while running a model over a dataset (donor side, precompute), and
    (b) inject a scaled steering vector into a model's activations during
    inference (receiver side, transfer).

Tap names are the fully-qualified module names from ``classifier.named_modules()``
(e.g. ``"model.blocks.0"``), so both scripts must build hooks against the same
``classifier`` root for the names to line up.

A steering vector is stored per tap as a per-token ``[tokens, dim]`` tensor (the
reduce-agnostic superset). The ``token_reduce`` setting is applied at injection:
``mean`` collapses it to ``[dim]``; ``per_token`` keeps it.
"""

import re
from collections import OrderedDict

import torch
from torch import nn


STEERING_STRATEGIES = ("per_block", "per_linear", "per_attn_mlp")
TOKEN_REDUCES = ("per_token", "mean")

# Per-block residual-stream tap for timm ViT/DeiT: the direct children of
# ``model.blocks`` (e.g. "model.blocks.0", ..., "model.blocks.11").
_PER_BLOCK_RE = re.compile(r"^model\.blocks\.\d+$")


def select_tap_modules(classifier: nn.Module, strategy: str) -> "OrderedDict[str, nn.Module]":
    """Return an ordered ``{tap_name: module}`` mapping for ``strategy``.

    ``tap_name`` is the fully-qualified name from ``classifier.named_modules()``;
    it is used identically as a cache key (precompute), a hook target (both
    scripts), and a results field. Only ``per_block`` is implemented today.
    """

    if strategy == "per_block":
        taps = OrderedDict(
            (name, module)
            for name, module in classifier.named_modules()
            if _PER_BLOCK_RE.match(name)
        )
        if not taps:
            raise ValueError(
                f"per_block found no modules matching {_PER_BLOCK_RE.pattern!r}. "
                "Is this a timm ViT/DeiT? (Swin names blocks model.layers.*.blocks.* "
                "— see NOTE below.)"
            )
        return taps

    if strategy == "per_linear":
        # TODO/NOTE: the closest 1:1 parallel to the weight-space QV. Tap the
        # OUTPUT of every backbone nn.Linear inside the blocks:
        #   model.blocks.{i}.attn.qkv   -> [B, N, 3*dim]
        #   model.blocks.{i}.attn.proj  -> [B, N, dim]
        #   model.blocks.{i}.mlp.fc1    -> [B, N, hidden]
        #   model.blocks.{i}.mlp.fc2    -> [B, N, dim]
        # i.e. OrderedDict((name, m) for name, m in classifier.named_modules()
        #        if isinstance(m, nn.Linear) and name.startswith("model.blocks.")).
        # Each tap then carries its own steering vector sized to that layer's
        # output dim; the rest of the pipeline already keys everything by name.
        raise NotImplementedError("steering_strategy='per_linear' is not implemented yet")

    if strategy == "per_attn_mlp":
        # TODO/NOTE: tap the OUTPUT of each block's attention and MLP submodules
        # (2 taps per block): names matching r"^model\.blocks\.\d+\.(attn|mlp)$".
        raise NotImplementedError("steering_strategy='per_attn_mlp' is not implemented yet")

    raise ValueError(
        f"Unknown steering_strategy {strategy!r}; expected one of {STEERING_STRATEGIES}"
    )


def _as_activation(output):
    """Return the principal activation tensor from a module's forward output.

    Blocks/Linears return a plain tensor; some submodules return a tuple whose
    first element is the activation. Returns (tensor, is_tuple) so a hook can
    rebuild the original container when it replaces the activation.
    """
    if isinstance(output, tuple):
        return output[0], True
    return output, False


class ActivationMeanCapture:
    """Accumulate the per-token mean activation at each tap while a model runs.

    Per tap, keeps a running sum (over samples) of the ``[tokens, dim]``
    activation plus a sample count; :meth:`result` returns ``{tap_name: mean}``
    of shape ``[tokens, dim]``. Sums accumulate in fp32 on CPU. Assumes a fixed
    token count across batches (true for ViT). Use as a context manager so the
    forward hooks are always removed.
    """

    def __init__(self, taps: "OrderedDict[str, nn.Module]"):
        self._taps = taps
        self._sums: "dict[str, torch.Tensor]" = {}
        self._counts: "dict[str, int]" = {}
        self._handles = []

    def _make_hook(self, name):
        def hook(module, inputs, output):
            act, _ = _as_activation(output)
            if not torch.is_tensor(act) or act.dim() != 3:
                raise ValueError(
                    f"tap {name!r} produced {type(act).__name__}/"
                    f"{getattr(act, 'shape', None)}; expected a [batch, tokens, dim] tensor"
                )
            batch_sum = act.detach().to(torch.float32).sum(dim=0).cpu()  # [tokens, dim]
            if name not in self._sums:
                self._sums[name] = batch_sum
                self._counts[name] = act.shape[0]
            else:
                if self._sums[name].shape != batch_sum.shape:
                    raise ValueError(
                        f"tap {name!r} token/dim shape changed across batches: "
                        f"{tuple(self._sums[name].shape)} vs {tuple(batch_sum.shape)}"
                    )
                self._sums[name] += batch_sum
                self._counts[name] += act.shape[0]

        return hook

    def __enter__(self):
        for name, module in self._taps.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def result(self) -> "OrderedDict[str, torch.Tensor]":
        out = OrderedDict()
        for name in self._taps:
            if name not in self._sums:
                raise RuntimeError(f"tap {name!r} never fired; did the model run?")
            out[name] = self._sums[name] / self._counts[name]
        return out

    @property
    def num_samples(self) -> int:
        return max(self._counts.values()) if self._counts else 0


def reduce_steering(vec: torch.Tensor, token_reduce: str) -> torch.Tensor:
    """Collapse a per-token ``[tokens, dim]`` steering vector per ``token_reduce``.

    ``per_token`` returns it unchanged; ``mean`` averages over the token axis
    to ``[dim]``.
    """
    if token_reduce == "per_token":
        return vec
    if token_reduce == "mean":
        return vec.mean(dim=0)
    raise ValueError(f"Unknown token_reduce {token_reduce!r}; expected one of {TOKEN_REDUCES}")


class ActivationInjector:
    """Add ``alpha * steer[tap]`` to each tapped module's output during forward.

    ``steer`` maps ``tap_name -> [tokens, dim]`` (as cached by precompute).
    ``token_reduce`` decides how it is applied:
      * ``mean``      -> collapse to ``[dim]``, broadcast over batch and tokens;
      * ``per_token`` -> keep ``[tokens, dim]``, broadcast over batch as
        ``[1, tokens, dim]`` (requires the live token count to match).
    Use as a context manager so the forward hooks are always removed.
    """

    def __init__(self, taps, steer, alpha, token_reduce, device=None):
        self._taps = taps
        self._alpha = float(alpha)
        self._token_reduce = token_reduce
        self._steer = OrderedDict()
        for name in taps:
            if name not in steer:
                raise KeyError(f"steering vector missing tap {name!r}")
            v = reduce_steering(steer[name], token_reduce)
            self._steer[name] = v.to(device) if device is not None else v
        self._handles = []

    def _make_hook(self, name):
        steer = self._steer[name]
        alpha = self._alpha
        per_token = self._token_reduce == "per_token"

        def hook(module, inputs, output):
            act, is_tuple = _as_activation(output)
            add = steer.to(dtype=act.dtype, device=act.device)
            if per_token:
                if add.shape[0] != act.shape[1]:
                    raise ValueError(
                        f"tap {name!r}: per_token steering has {add.shape[0]} tokens "
                        f"but activation has {act.shape[1]}"
                    )
                add = add.unsqueeze(0)  # [1, tokens, dim]
            new_act = act + alpha * add
            if is_tuple:
                return (new_act,) + tuple(output[1:])
            return new_act

        return hook

    def __enter__(self):
        for name, module in self._taps.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
