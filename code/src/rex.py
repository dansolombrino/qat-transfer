import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Optional, Tuple

from src.quantization import RexLinear, fake_quantize_tensor


# =============================================================================
# Core REx helpers (residual expansion for post-training quantization)
# =============================================================================


def _sparsify_by_output_channel(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity <= 0.0:
        return weight
    if sparsity >= 1.0:
        return torch.zeros_like(weight)

    if weight.ndim < 2:
        return weight

    out_channels = weight.shape[0]
    keep_fraction = 1.0 - sparsity
    if keep_fraction >= 1.0:
        return weight
    if keep_fraction <= 0.0:
        return torch.zeros_like(weight)

    channel_norms = torch.linalg.vector_norm(
        weight.flatten(start_dim=1),
        ord=2,
        dim=1,
    )
    threshold = torch.quantile(channel_norms, sparsity)
    mask = channel_norms > threshold
    if mask.all():
        return weight
    if not mask.any():
        return torch.zeros_like(weight)

    view_shape = [out_channels] + [1] * (weight.ndim - 1)
    return weight * mask.view(view_shape)


def _rex_expand_terms(
    weight: torch.Tensor,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
) -> List[torch.Tensor]:
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    residual = weight
    terms: List[torch.Tensor] = []

    for k in range(order):
        if k == 0:
            quantized_residual = fake_quantize_tensor(residual, bits, granularity)
            residual = residual - quantized_residual
        else:
            masked_residual = _sparsify_by_output_channel(residual, sparsity)
            quantized_residual = fake_quantize_tensor(
                masked_residual,
                bits,
                granularity,
            )
            residual = masked_residual - quantized_residual
        terms.append(quantized_residual)

    return terms


def _rex_expand_weight(
    weight: torch.Tensor,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
) -> torch.Tensor:
    terms = _rex_expand_terms(
        weight=weight,
        bits=bits,
        granularity=granularity,
        order=order,
        sparsity=sparsity,
    )
    return torch.stack(terms, dim=0).sum(dim=0)


# =============================================================================
# RexMultiheadAttention (inference wrapper for nn.MultiheadAttention)
# =============================================================================


class RexMultiheadAttention(nn.Module):

    """
    Inference-time wrapper around nn.MultiheadAttention that stores the
    in_proj_weight as a base quantized weight plus K-1 correction terms
    and reconstructs the full weight on forward.  out_proj is handled
    separately (replaced with RexLinear by the caller).
    """

    def __init__(
        self,
        original_mha: nn.MultiheadAttention,
        base_in_proj_weight: torch.Tensor,
        correction_in_proj_terms: List[torch.Tensor],
    ):
        super().__init__()
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.dropout = original_mha.dropout
        self.batch_first = original_mha.batch_first
        self.add_zero_attn = original_mha.add_zero_attn
        self._qkv_same_embed_dim = original_mha._qkv_same_embed_dim

        self.register_buffer("base_in_proj_weight", base_in_proj_weight)
        if len(correction_in_proj_terms) == 0:
            corrections = torch.empty(
                (0, *base_in_proj_weight.shape),
                dtype=base_in_proj_weight.dtype,
                device=base_in_proj_weight.device,
            )
        else:
            corrections = torch.stack(correction_in_proj_terms, dim=0)
        self.register_buffer("correction_in_proj_terms", corrections)

        if original_mha.in_proj_bias is not None:
            self.register_buffer("in_proj_bias", original_mha.in_proj_bias.data.clone())
        else:
            self.in_proj_bias = None

        if original_mha.bias_k is not None:
            self.register_buffer("bias_k", original_mha.bias_k.data.clone())
        else:
            self.bias_k = None

        if original_mha.bias_v is not None:
            self.register_buffer("bias_v", original_mha.bias_v.data.clone())
        else:
            self.bias_v = None

        # out_proj is kept as-is; the caller replaces it with RexLinear
        # via recursion into children.
        self.out_proj = original_mha.out_proj

    @property
    def in_proj_weight(self) -> torch.Tensor:
        weight = self.base_in_proj_weight
        if self.correction_in_proj_terms.numel() > 0:
            weight = weight + self.correction_in_proj_terms.sum(dim=0)
        return weight

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        w_in = self.in_proj_weight

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            w_in,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_weights


# =============================================================================
# apply_rex_ptq_ (nn.Linear only — generic models)
# =============================================================================


def apply_rex_ptq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> List[str]:

    quantized: List[str] = []

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.copy_(
                    _rex_expand_weight(
                        weight=module.weight,
                        bits=bits,
                        granularity=granularity,
                        order=order,
                        sparsity=sparsity,
                    )
                )
            quantized.append(full)
        else:
            quantized.extend(
                apply_rex_ptq_(
                    model=module,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                    skip_modules=skip_modules,
                    _prefix=full + ".",
                )
            )

    return quantized


# =============================================================================
# apply_rex_ptq_terms_ (nn.Linear only — generic models, wraps in RexLinear)
# =============================================================================


def apply_rex_ptq_terms_(
    model: nn.Module,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> List[str]:

    quantized: List[str] = []

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                terms = _rex_expand_terms(
                    weight=module.weight,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                )
                base_weight = terms[0]
                correction_terms = terms[1:]
                rex_linear = RexLinear(
                    base_weight=base_weight,
                    correction_terms=correction_terms,
                    bias=module.bias,
                )
            setattr(model, name, rex_linear)
            quantized.append(full)
        else:
            quantized.extend(
                apply_rex_ptq_terms_(
                    model=module,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                    skip_modules=skip_modules,
                    _prefix=full + ".",
                )
            )

    return quantized


# =============================================================================
# Open-CLIP variants (handle nn.MultiheadAttention's fused in_proj_weight)
# =============================================================================


def apply_rex_ptq_open_clip_(
    model: nn.Module,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> List[str]:

    """
    Apply REx PTQ in-place: replace weights with the summed residual
    expansion.  Handles both nn.Linear and nn.MultiheadAttention
    (fused in_proj_weight).  Skips modules in *skip_modules*.

    Returns fully-qualified names of every quantized layer/parameter.
    """

    quantized: List[str] = []

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.MultiheadAttention):
            with torch.no_grad():
                module.in_proj_weight.copy_(
                    _rex_expand_weight(
                        weight=module.in_proj_weight,
                        bits=bits,
                        granularity=granularity,
                        order=order,
                        sparsity=sparsity,
                    )
                )
            quantized.append(f"{full}.in_proj_weight")
            quantized.extend(
                apply_rex_ptq_open_clip_(
                    model=module,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                    skip_modules=skip_modules,
                    _prefix=full + ".",
                )
            )
        elif isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.copy_(
                    _rex_expand_weight(
                        weight=module.weight,
                        bits=bits,
                        granularity=granularity,
                        order=order,
                        sparsity=sparsity,
                    )
                )
            quantized.append(full)
        else:
            quantized.extend(
                apply_rex_ptq_open_clip_(
                    model=module,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                    skip_modules=skip_modules,
                    _prefix=full + ".",
                )
            )

    return quantized


def apply_rex_ptq_terms_open_clip_(
    model: nn.Module,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> List[str]:

    """
    Apply REx PTQ with separate correction terms.  Wraps nn.Linear in
    RexLinear and nn.MultiheadAttention in RexMultiheadAttention.
    Skips modules in *skip_modules*.

    Returns fully-qualified names of every quantized layer/parameter.
    """

    quantized: List[str] = []

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.MultiheadAttention):
            with torch.no_grad():
                in_proj_terms = _rex_expand_terms(
                    weight=module.in_proj_weight,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                )
                base_in_proj = in_proj_terms[0]
                correction_in_proj = in_proj_terms[1:]

            # Recurse into MHA children first to wrap out_proj with RexLinear.
            apply_rex_ptq_terms_open_clip_(
                model=module,
                bits=bits,
                granularity=granularity,
                order=order,
                sparsity=sparsity,
                skip_modules=skip_modules,
                _prefix=full + ".",
            )

            rex_mha = RexMultiheadAttention(
                original_mha=module,
                base_in_proj_weight=base_in_proj,
                correction_in_proj_terms=correction_in_proj,
            )
            setattr(model, name, rex_mha)
            quantized.append(f"{full}.in_proj_weight")

            # Collect out_proj (now RexLinear) name.
            for child_name, child in rex_mha.named_children():
                if isinstance(child, RexLinear):
                    quantized.append(f"{full}.{child_name}")

        elif isinstance(module, nn.Linear):
            with torch.no_grad():
                terms = _rex_expand_terms(
                    weight=module.weight,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                )
                base_weight = terms[0]
                correction_terms = terms[1:]
                rex_linear = RexLinear(
                    base_weight=base_weight,
                    correction_terms=correction_terms,
                    bias=module.bias,
                )
            setattr(model, name, rex_linear)
            quantized.append(full)
        else:
            quantized.extend(
                apply_rex_ptq_terms_open_clip_(
                    model=module,
                    bits=bits,
                    granularity=granularity,
                    order=order,
                    sparsity=sparsity,
                    skip_modules=skip_modules,
                    _prefix=full + ".",
                )
            )

    return quantized
