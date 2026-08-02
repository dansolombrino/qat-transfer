"""Apply official RepQ-ViT scale reparameterization to current timm models.

Derived from the official RepQ-ViT classification ``quant_model.py`` and
``test_quant.py`` (Apache-2.0).  Modifications: current timm 1.x attention
compatibility, safe recursive replacement, explicit skip semantics, structural
LayerNorm-pair discovery, validation, and a project-style in-place public API.
The RepQ-ViT algorithm itself is unchanged: initial W/A calibration, post-LN
scale folding, log-sqrt(2) post-Softmax quantization, and recalibration.
"""

from __future__ import annotations

from copy import deepcopy
from types import MethodType

import torch
from torch import nn

from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul


class _MatMul(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


def _vit_attention_forward(
    self,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Current timm Attention forward with official RepQ-ViT matmuls."""

    from timm.models.vision_transformer import maybe_add_mask, resolve_self_attn_mask

    batch, tokens, _ = x.shape
    qkv = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    # The official implementation quantizes unscaled q and applies scale after
    # QK^T. Keep that order: moving the scale before QuantMatMul changes its grid.
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn_bias = resolve_self_attn_mask(tokens, attn, attn_mask, is_causal)
    attn = maybe_add_mask(attn, attn_bias)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(batch, tokens, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    return self.proj_drop(x)


def _swin_attention_forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Current timm WindowAttention forward with official RepQ-ViT matmuls."""

    batch_windows, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch_windows, tokens, 3, self.num_heads, -1
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        num_windows = mask.shape[0]
        attn = attn.view(
            -1, num_windows, self.num_heads, tokens, tokens
        ) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, tokens, tokens)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(
        batch_windows, tokens, channels
    )
    x = self.proj(x)
    return self.proj_drop(x)


def _inject_attention_matmuls_(model: nn.Module) -> list[str]:
    from timm.models.swin_transformer import WindowAttention
    from timm.models.vision_transformer import Attention

    injected = []
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.fused_attn = False
            module.matmul1 = _MatMul()
            module.matmul2 = _MatMul()
            module.forward = MethodType(_vit_attention_forward, module)
            injected.append(name)
        elif isinstance(module, WindowAttention):
            module.fused_attn = False
            module.matmul1 = _MatMul()
            module.matmul2 = _MatMul()
            module.forward = MethodType(_swin_attention_forward, module)
            injected.append(name)
    return injected


def _copy_parameter_(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None:
            target.bias.copy_(source.bias)
            target.bias.requires_grad_(source.bias.requires_grad)
    target.train(source.training)


def _wrap_modules_(
    module: nn.Module,
    input_quant_params: dict,
    weight_quant_params: dict,
    skip_modules: frozenset[str],
    prefix: str = "",
) -> list[str]:
    channel_wise_names = frozenset({"qkv", "fc1", "reduction"})
    input_channel = deepcopy(input_quant_params)
    input_channel["channel_wise"] = True
    input_log = deepcopy(input_quant_params)
    input_log["log_quant"] = True
    quantized = []

    for child_name, child in list(module.named_children()):
        if child_name in skip_modules:
            continue
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Conv2d):
            replacement = QuantConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                input_quant_params=input_quant_params,
                weight_quant_params=weight_quant_params,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            _copy_parameter_(replacement, child)
            setattr(module, child_name, replacement)
            quantized.append(full_name)
        elif isinstance(child, nn.Linear):
            replacement = QuantLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                input_quant_params=(
                    input_channel
                    if child_name in channel_wise_names
                    else input_quant_params
                ),
                weight_quant_params=weight_quant_params,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            _copy_parameter_(replacement, child)
            setattr(module, child_name, replacement)
            quantized.append(full_name)
        elif isinstance(child, _MatMul):
            setattr(
                module,
                child_name,
                QuantMatMul(input_log if child_name == "matmul2" else input_quant_params),
            )
            quantized.append(full_name)
        else:
            quantized.extend(
                _wrap_modules_(
                    child,
                    input_quant_params,
                    weight_quant_params,
                    skip_modules,
                    prefix=full_name,
                )
            )
    return quantized


def _set_quant_state_(
    model: nn.Module,
    input_quant: bool,
    weight_quant: bool,
) -> None:
    for module in model.modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            module.set_quant_state(input_quant, weight_quant)


def _reparameterize_pair_(norm: nn.Module, linear: QuantLinear) -> None:
    quantizer = linear.input_quantizer
    if not quantizer.inited or not quantizer.channel_wise:
        raise RuntimeError("RepQ-ViT reparameterization requires calibrated channel scales")
    if norm.weight is None:
        raise RuntimeError("RepQ-ViT requires affine LayerNorm weights")

    act_delta = quantizer.delta.reshape(-1)
    act_zero = quantizer.zero_point.reshape(-1)
    if act_delta.numel() != linear.in_features:
        raise RuntimeError(
            "RepQ-ViT LayerNorm/Linear width mismatch: "
            f"{act_delta.numel()} vs {linear.in_features}"
        )
    if not torch.all(torch.isfinite(act_delta)) or torch.any(act_delta <= 0):
        raise RuntimeError("RepQ-ViT produced invalid post-LayerNorm scales")

    act_min = -act_zero * act_delta
    target_delta = act_delta.mean()
    target_zero = act_zero.mean()
    target_min = -target_zero * target_delta
    ratio = act_delta / target_delta
    shift = act_min / ratio - target_min

    norm_bias = norm.bias
    if norm_bias is None:
        norm.bias = nn.Parameter(torch.zeros_like(norm.weight))
        norm_bias = norm.bias

    norm.weight.div_(ratio)
    norm_bias.div_(ratio).sub_(shift)
    linear.weight.mul_(ratio)
    correction = linear.weight @ shift
    if linear.bias is None:
        linear.bias = nn.Parameter(correction.clone())
    else:
        linear.bias.add_(correction)

    quantizer.channel_wise = False
    quantizer.delta = target_delta
    quantizer.zero_point = target_zero
    linear.weight_quantizer.reset()


def _scale_reparameterize_(inner_model: nn.Module) -> list[str]:
    names = {id(module): name for name, module in inner_model.named_modules()}
    pairs: list[tuple[nn.Module, QuantLinear]] = []

    for module in inner_model.modules():
        norm1 = getattr(module, "norm1", None)
        attn = getattr(module, "attn", None)
        qkv = getattr(attn, "qkv", None) if attn is not None else None
        if norm1 is not None and isinstance(qkv, QuantLinear):
            pairs.append((norm1, qkv))

        norm2 = getattr(module, "norm2", None)
        mlp = getattr(module, "mlp", None)
        fc1 = getattr(mlp, "fc1", None) if mlp is not None else None
        if norm2 is not None and isinstance(fc1, QuantLinear):
            pairs.append((norm2, fc1))

        norm = getattr(module, "norm", None)
        reduction = getattr(module, "reduction", None)
        if norm is not None and isinstance(reduction, QuantLinear):
            pairs.append((norm, reduction))

    seen = set()
    reparameterized = []
    with torch.no_grad():
        for norm, linear in pairs:
            key = (id(norm), id(linear))
            if key in seen:
                continue
            seen.add(key)
            _reparameterize_pair_(norm, linear)
            reparameterized.append(
                f"{names.get(id(norm), '<norm>')}->{names.get(id(linear), '<linear>')}"
            )
    return reparameterized


def apply_repqvit_(
    model: nn.Module,
    calib_data: torch.Tensor,
    w_bits: int,
    a_bits: int,
    skip_modules: frozenset[str],
) -> list[str]:
    """Apply RepQ-ViT W/A post-training quantization in place.

    ``calib_data`` must be one receiver-training image batch.  The caller owns
    deterministic materialization and can reuse the same tensor across models.
    ``skip_modules`` contains child attribute names (for example ``{"head"}``).
    """

    if not isinstance(skip_modules, frozenset):
        raise TypeError("skip_modules must be an explicit frozenset of child names")
    if not 2 <= int(w_bits) <= 8 or not 2 <= int(a_bits) <= 8:
        raise ValueError(f"RepQ-ViT supports 2-8 bits, got W{w_bits}/A{a_bits}")
    if not isinstance(calib_data, torch.Tensor) or calib_data.ndim != 4:
        raise ValueError("calib_data must be an image tensor shaped (N, C, H, W)")
    if calib_data.shape[0] == 0:
        raise ValueError("calib_data must contain at least one sample")

    injected = _inject_attention_matmuls_(model)
    if not injected:
        raise ValueError(
            "RepQ-ViT supports timm VisionTransformer/DeiT and Swin attention; "
            "no compatible attention modules were found"
        )

    quantized = _wrap_modules_(
        model,
        input_quant_params={"n_bits": int(a_bits), "channel_wise": False},
        weight_quant_params={"n_bits": int(w_bits), "channel_wise": True},
        skip_modules=skip_modules,
    )
    if not quantized:
        raise RuntimeError("RepQ-ViT found no eligible modules after applying skips")

    device = next(model.parameters()).device
    calib_data = calib_data.to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    _set_quant_state_(model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        model(calib_data)

    inner_model = model.model if hasattr(model, "model") else model
    reparameterized = _scale_reparameterize_(inner_model)
    if not reparameterized:
        raise RuntimeError(
            "RepQ-ViT calibrated no supported LayerNorm-to-projection pairs; "
            "refusing to report a partial quantization baseline"
        )

    with torch.no_grad():
        model(calib_data)
    return quantized
