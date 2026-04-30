from copy import deepcopy
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm

from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul


class MatMul(nn.Module):
    """Explicit matrix-multiply module so RepQ-ViT can wrap it with QuantMatMul."""
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B


# ---------------------------------------------------------------------------
# Patched attention forward methods (use self.matmul1/matmul2 instead of @)
# ---------------------------------------------------------------------------

def _vit_attention_forward(
    self, x: torch.Tensor, attn_mask=None, is_causal: bool = False
) -> torch.Tensor:
    B, N, _ = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    # Match original RepQ-ViT: matmul on raw q, k — scale is applied AFTER.
    # (Newer timm scales q first; we follow the original so QuantMatMul sees
    # the same value ranges the authors calibrated against.)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    # attn_mask / is_causal ignored: standard classification doesn't use them
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _swin_window_attention_forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(
        self.window_size[0] * self.window_size[1],
        self.window_size[0] * self.window_size[1],
        -1,
    )
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _open_clip_mha_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
):
    """Patched forward for nn.MultiheadAttention in open_clip ResidualAttentionBlock.

    Replaces F.multi_head_attention_forward with explicit matmul1/matmul2
    so that RepQ-ViT can quantize each matrix multiplication independently.
    Supports optional input/weight quantization via attached quantizers.
    """
    B, L, D = query.shape
    head_dim = D // self.num_heads
    scale = head_dim ** -0.5

    # QKV projection with optional quantization
    x = query
    if getattr(self, 'use_input_quant', False):
        x = self.input_quantizer(x)
    w = self.in_proj_weight
    if getattr(self, 'use_weight_quant', False):
        w = self.weight_quantizer(w)
    qkv = F.linear(x, w, self.in_proj_bias)

    # Split into q, k, v and reshape for multi-head
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, L, self.num_heads, head_dim).transpose(1, 2)  # (B, H, L, hd)
    k = k.reshape(B, L, self.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, L, self.num_heads, head_dim).transpose(1, 2)

    # Pre-softmax attention (matmul1)
    attn = self.matmul1(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)

    # Post-softmax attention (matmul2)
    x = self.matmul2(attn, v)

    # Reshape back and output projection
    x = x.transpose(1, 2).reshape(B, L, D)
    x = self.out_proj(x)

    return x, None


# ---------------------------------------------------------------------------
# Step 1 — inject MatMul modules into Attention layers
# ---------------------------------------------------------------------------

def inject_matmul_modules_(model: nn.Module, family: str = "timm") -> None:
    """Replace @ in attention layers with explicit MatMul modules.

    Args:
        model:  The top-level nn.Module.
        family: ``"timm"`` for timm ViT/Swin, ``"open_clip"`` for open_clip
                ResidualAttentionBlock with nn.MultiheadAttention.
    """
    if family == "timm":
        try:
            from timm.models.vision_transformer import Attention
        except ImportError:
            Attention = None
        try:
            from timm.models.swin_transformer import WindowAttention
        except ImportError:
            WindowAttention = None

        for module in model.modules():
            if Attention is not None and type(module) is Attention:
                module.fused_attn = False
                module.matmul1 = MatMul()
                module.matmul2 = MatMul()
                module.forward = MethodType(_vit_attention_forward, module)
            elif WindowAttention is not None and type(module) is WindowAttention:
                module.fused_attn = False
                module.matmul1 = MatMul()
                module.matmul2 = MatMul()
                module.forward = MethodType(_swin_window_attention_forward, module)

    elif family == "open_clip":
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.matmul1 = MatMul()
                module.matmul2 = MatMul()
                module.forward = MethodType(_open_clip_mha_forward, module)

    else:
        raise ValueError(f"Unknown family={family!r}. Supported: 'timm', 'open_clip'")


# ---------------------------------------------------------------------------
# Step 2 — recursively wrap Conv2d / Linear / MatMul with quantized versions
# ---------------------------------------------------------------------------

_TIMM_CHANNEL_WISE_NAMES = frozenset(('qkv', 'fc1', 'reduction'))


def _wrap_modules_(
    model: nn.Module,
    aq_params: dict,
    wq_params: dict,
    skip_modules: frozenset,
    channel_wise_names: frozenset = _TIMM_CHANNEL_WISE_NAMES,
    _prefix: str = "",
) -> List[str]:
    """
    Recursively replace nn.Conv2d, nn.Linear, and MatMul with their quantized
    counterparts.  For nn.MultiheadAttention modules (open_clip), attach
    quantizers directly instead of replacing the module.

    Skips any child whose *attribute name* is in *skip_modules*.

    Linears whose name is in *channel_wise_names* receive channel-wise
    activation quantization; all others receive tensor-wise.
    Post-softmax matmul (matmul2) receives log-sqrt(2) activation quantization.
    """
    from .quantizer import UniformQuantizer

    aq_channel = deepcopy(aq_params)
    aq_channel['channel_wise'] = True

    aq_log = deepcopy(aq_params)
    aq_log['log_quant'] = True

    quantized: List[str] = []

    for name, m in list(model.named_children()):
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}" if _prefix else name

        if isinstance(m, nn.Conv2d):
            new_m = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                m.stride, m.padding, m.dilation, m.groups,
                m.bias is not None, aq_params, wq_params,
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(model, name, new_m)
            quantized.append(full)

        elif isinstance(m, nn.MultiheadAttention):
            # open_clip: attach quantizers for fused in_proj_weight directly
            m.input_quantizer = UniformQuantizer(**aq_channel)
            m.weight_quantizer = UniformQuantizer(**wq_params)
            m.use_input_quant = False
            m.use_weight_quant = False
            quantized.append(full)
            # Recurse into MHA children (out_proj, matmul1, matmul2)
            quantized.extend(
                _wrap_modules_(m, aq_params, wq_params, skip_modules,
                               channel_wise_names, _prefix=full + ".")
            )

        elif isinstance(m, nn.Linear):
            if name in channel_wise_names:
                new_m = QuantLinear(m.in_features, m.out_features, aq_channel, wq_params)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, aq_params, wq_params)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(model, name, new_m)
            quantized.append(full)

        elif isinstance(m, MatMul):
            # matmul2 is post-softmax: use log-sqrt(2) for attention weights
            new_m = QuantMatMul(aq_log if name == 'matmul2' else aq_params)
            setattr(model, name, new_m)
            quantized.append(full)

        else:
            quantized.extend(
                _wrap_modules_(m, aq_params, wq_params, skip_modules,
                               channel_wise_names, _prefix=full + ".")
            )

    return quantized


# ---------------------------------------------------------------------------
# Quant-state toggle
# ---------------------------------------------------------------------------

def _set_quant_state_(model: nn.Module, input_quant: bool, weight_quant: bool) -> None:
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
        elif isinstance(m, nn.MultiheadAttention) and hasattr(m, 'use_input_quant'):
            m.use_input_quant = input_quant
            m.use_weight_quant = weight_quant


# ---------------------------------------------------------------------------
# Step 4 — scale reparameterization (RepQ-ViT core contribution)
# ---------------------------------------------------------------------------

def _scale_reparameterize_(inner_model: nn.Module, family: str = "timm") -> None:
    """Dispatch scale reparameterization based on model family."""
    if family == "timm":
        _scale_reparameterize_timm_(inner_model)
    elif family == "open_clip":
        _scale_reparameterize_open_clip_(inner_model)
    else:
        raise ValueError(f"Unknown family={family!r}")


def _scale_reparameterize_timm_(timm_model: nn.Module) -> None:
    """
    RepQ-ViT scale reparameterization for timm ViT/DeiT/Swin.
    Iterates ``blocks`` or ``layers`` and handles all three patterns:
        block.norm1         -> block.attn.qkv     (channel-wise activation quant)
        block.norm2         -> block.mlp.fc1      (channel-wise activation quant)
        patch_merging.norm  -> patch_merging.reduction
    """
    if hasattr(timm_model, 'blocks'):
        slice_ = timm_model.blocks
    elif hasattr(timm_model, 'layers'):
        slice_ = timm_model.layers
    else:
        return

    module_dict: dict = {}
    for name, module in slice_.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        father_name = name[:idx] if idx != -1 else ''
        if father_name not in module_dict:
            continue
        father_module = module_dict[father_name]

        if 'norm1' in name:
            next_module = getattr(getattr(father_module, 'attn', None), 'qkv', None)
        elif 'norm2' in name:
            next_module = getattr(getattr(father_module, 'mlp', None), 'fc1', None)
        elif name.endswith('norm'):
            # Swin PatchMerging.norm; also matches newer timm Attention.q_norm/
            # k_norm/norm — those are guarded by the next-module check below.
            next_module = getattr(father_module, 'reduction', None)
        else:
            continue

        if not isinstance(next_module, QuantLinear):
            continue
        if not next_module.input_quantizer.inited:
            continue
        if not next_module.input_quantizer.channel_wise:
            continue

        _reparam_norm_and_linear_(module, next_module)


def _scale_reparameterize_open_clip_(open_clip_model: nn.Module) -> None:
    """
    RepQ-ViT scale reparameterization for open_clip VisionTransformer.
    Iterates ``visual.transformer.resblocks`` and handles two patterns per block:
        block.ln_1  -> block.attn   (MHA with channel-wise input quant on in_proj_weight)
        block.ln_2  -> block.mlp.c_fc  (channel-wise activation quant)
    """
    visual = getattr(open_clip_model, 'visual', None)
    if visual is None:
        return
    transformer = getattr(visual, 'transformer', None)
    if transformer is None:
        return
    resblocks = getattr(transformer, 'resblocks', None)
    if resblocks is None:
        return

    for block in resblocks:
        # ln_1 -> attn (MHA with attached quantizers)
        ln_1 = getattr(block, 'ln_1', None)
        attn = getattr(block, 'attn', None)
        if (
            ln_1 is not None
            and isinstance(attn, nn.MultiheadAttention)
            and hasattr(attn, 'input_quantizer')
            and attn.input_quantizer.inited
            and attn.input_quantizer.channel_wise
        ):
            _reparam_norm_and_mha_(ln_1, attn)

        # ln_2 -> mlp.c_fc (QuantLinear)
        ln_2 = getattr(block, 'ln_2', None)
        mlp = getattr(block, 'mlp', None)
        c_fc = getattr(mlp, 'c_fc', None) if mlp is not None else None
        if (
            ln_2 is not None
            and isinstance(c_fc, QuantLinear)
            and c_fc.input_quantizer.inited
            and c_fc.input_quantizer.channel_wise
        ):
            _reparam_norm_and_linear_(ln_2, c_fc)


def _reparam_norm_and_linear_(norm: nn.Module, linear: QuantLinear) -> None:
    """
    Absorb per-channel activation scales r (and offsets b) from the calibrated
    channel-wise quantizer of `linear` into `norm`'s weight/bias and `linear`'s
    weight/bias, then switch the input quantizer of `linear` to tensor-wise.
    """
    act_delta = linear.input_quantizer.delta.reshape(-1)        # (C,)
    act_zero_point = linear.input_quantizer.zero_point.reshape(-1)  # (C,)
    act_min = -act_zero_point * act_delta                        # per-channel min

    target_delta = act_delta.mean()
    target_zero_point = act_zero_point.mean()
    target_min = -target_zero_point * target_delta

    r = act_delta / target_delta         # per-channel scale ratio   (C,)
    b = act_min / r - target_min         # per-channel bias offset   (C,)

    # Absorb into LayerNorm: divide weight and bias by r, subtract b from bias
    norm.weight.data = norm.weight.data / r
    norm.bias.data = norm.bias.data / r - b

    # Absorb into linear: scale rows by r, absorb b into bias
    linear.weight.data = linear.weight.data * r
    bias_correction = torch.mm(linear.weight.data, b.reshape(-1, 1)).reshape(-1)
    if linear.bias is not None:
        linear.bias.data = linear.bias.data + bias_correction
    else:
        linear.bias = Parameter(torch.zeros(linear.out_features, device=linear.weight.device))
        linear.bias.data = bias_correction

    # Switch input quantizer to tensor-wise with the mean scale
    linear.input_quantizer.channel_wise = False
    linear.input_quantizer.delta = target_delta
    linear.input_quantizer.zero_point = target_zero_point
    # Reset weight quantizer so it re-calibrates in the next forward pass
    linear.weight_quantizer.inited = False


def _reparam_norm_and_mha_(norm: nn.Module, mha: nn.MultiheadAttention) -> None:
    """
    Absorb per-channel activation scales from the calibrated channel-wise
    quantizer attached to *mha* into *norm*'s weight/bias and *mha*'s
    ``in_proj_weight`` / ``in_proj_bias``, then switch to tensor-wise.

    Same maths as :func:`_reparam_norm_and_linear_` but operates on the MHA's
    fused ``in_proj_weight`` parameter instead of an nn.Linear's weight.
    """
    act_delta = mha.input_quantizer.delta.reshape(-1)               # (C,)
    act_zero_point = mha.input_quantizer.zero_point.reshape(-1)     # (C,)
    act_min = -act_zero_point * act_delta                            # per-channel min

    target_delta = act_delta.mean()
    target_zero_point = act_zero_point.mean()
    target_min = -target_zero_point * target_delta

    r = act_delta / target_delta         # per-channel scale ratio   (C,)
    b = act_min / r - target_min         # per-channel bias offset   (C,)

    # Absorb into LayerNorm: divide weight and bias by r, subtract b from bias
    norm.weight.data = norm.weight.data / r
    norm.bias.data = norm.bias.data / r - b

    # Absorb into in_proj_weight: multiply along input dimension (dim=1)
    # in_proj_weight shape: (3*D, D), r shape: (D,) → broadcast on dim=1
    mha.in_proj_weight.data = mha.in_proj_weight.data * r.unsqueeze(0)
    bias_correction = (mha.in_proj_weight.data @ b.reshape(-1, 1)).reshape(-1)
    if mha.in_proj_bias is not None:
        mha.in_proj_bias.data = mha.in_proj_bias.data + bias_correction
    else:
        mha.in_proj_bias = Parameter(bias_correction)

    # Switch input quantizer to tensor-wise with the mean scale
    mha.input_quantizer.channel_wise = False
    mha.input_quantizer.delta = target_delta
    mha.input_quantizer.zero_point = target_zero_point
    # Reset weight quantizer so it re-calibrates in the next forward pass
    mha.weight_quantizer.inited = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_inner_model(model: nn.Module, family: str) -> nn.Module:
    """Navigate from the top-level classifier to the inner backbone model."""
    if family == "timm":
        # ImageClassifier.model (timm model)
        return model.model if hasattr(model, 'model') else model
    elif family == "open_clip":
        # ImageClassifier.image_encoder.model (open_clip CLIP model)
        encoder = getattr(model, 'image_encoder', None)
        if encoder is not None:
            return getattr(encoder, 'model', model)
        return model
    else:
        raise ValueError(f"Unknown family={family!r}")


def _get_quant_modules(model: nn.Module) -> dict:
    """Collect all quantized modules (including MHA with attached quantizers)."""
    quant_types = (QuantConv2d, QuantLinear, QuantMatMul)
    result = {}
    for name, m in model.named_modules():
        if isinstance(m, quant_types):
            result[name] = m
        elif isinstance(m, nn.MultiheadAttention) and hasattr(m, 'input_quantizer'):
            result[name] = m
    return result


def apply_repqvit_(
    model: nn.Module,
    calib_data: torch.Tensor,
    w_bits: int,
    a_bits: int,
    skip_modules: frozenset,
    family: str = "timm",
    channel_wise_names: frozenset = None,
    tqdm_kw: dict = None,
) -> List[str]:
    """
    Apply RepQ-ViT post-training quantization.

    Steps:
      1. Inject MatMul modules into Attention layers.
      2. Wrap Conv2d / Linear / MatMul with quantized counterparts.
      3. Initial calibration forward pass — initialises quantizer scales.
      4. Scale reparameterization — converts channel-wise to tensor-wise quant
         without accuracy loss.
      5. Re-calibration — re-initialises weight quantizers after reparameterization.

    Args:
        model:              The top-level nn.Module (e.g. ImageClassifier).
        calib_data:         A calibration batch of images, shape (N, C, H, W).
        w_bits:             Bit-width for weights.
        a_bits:             Bit-width for activations.
        skip_modules:       Frozenset of child attribute names to leave unquantized.
        family:             ``"timm"`` or ``"open_clip"``.
        channel_wise_names: Frozenset of nn.Linear child names that receive
                            channel-wise activation quantization.  Defaults to
                            ``{'qkv', 'fc1', 'reduction'}`` for timm.
        tqdm_kw:            Extra kwargs forwarded to tqdm (e.g. disable, colour).

    Returns:
        List of fully-qualified names of every module that was quantized.
    """
    if tqdm_kw is None:
        tqdm_kw = {}
    if channel_wise_names is None:
        channel_wise_names = _TIMM_CHANNEL_WISE_NAMES

    device = next(model.parameters()).device

    steps = [
        "Inject MatMul modules",
        "Wrap modules with quantized counterparts",
        "Initial calibration forward pass",
        "Scale reparameterization",
        "Re-calibration forward pass",
    ]
    step_bar = tqdm(steps, desc="RepQ-ViT", leave=False, **tqdm_kw)

    # Step 1
    step_bar.set_postfix_str(steps[0])
    inject_matmul_modules_(model, family=family)
    step_bar.update(1)

    # Step 2
    step_bar.set_postfix_str(steps[1])
    wq_params = {'n_bits': w_bits, 'channel_wise': True}
    aq_params = {'n_bits': a_bits, 'channel_wise': False}
    quantized_names = _wrap_modules_(model, aq_params, wq_params, skip_modules,
                                     channel_wise_names)
    step_bar.update(1)

    # Build name→module mapping for hook-based progress bars
    quant_modules = _get_quant_modules(model)

    # Step 3 — initial calibration
    step_bar.set_postfix_str(steps[2])
    _set_quant_state_(model, input_quant=True, weight_quant=True)
    model.eval()
    calib_data = calib_data.to(device)
    calib_bar = tqdm(total=len(quant_modules), desc="Calibrating (init)", leave=False, **tqdm_kw)
    hooks = []
    for name, m in quant_modules.items():
        def _make_hook(mod_name):
            def _hook(module, input, output):
                calib_bar.set_postfix_str(mod_name)
                calib_bar.update(1)
            return _hook
        hooks.append(m.register_forward_hook(_make_hook(name)))
    with torch.no_grad():
        model(calib_data)
    for h in hooks:
        h.remove()
    calib_bar.close()
    step_bar.update(1)

    # Step 4 — scale reparameterization
    step_bar.set_postfix_str(steps[3])
    inner_model = _get_inner_model(model, family)
    with torch.no_grad():
        _scale_reparameterize_(inner_model, family=family)
    step_bar.update(1)

    # Step 5 — re-calibration (resets weight quantizers after reparameterization)
    step_bar.set_postfix_str(steps[4])
    _set_quant_state_(model, input_quant=True, weight_quant=True)
    recalib_bar = tqdm(total=len(quant_modules), desc="Calibrating (re-init)", leave=False, **tqdm_kw)
    hooks = []
    for name, m in quant_modules.items():
        def _make_hook(mod_name):
            def _hook(module, input, output):
                recalib_bar.set_postfix_str(mod_name)
                recalib_bar.update(1)
            return _hook
        hooks.append(m.register_forward_hook(_make_hook(name)))
    with torch.no_grad():
        model(calib_data)
    for h in hooks:
        h.remove()
    recalib_bar.close()
    step_bar.update(1)

    step_bar.close()

    return quantized_names
