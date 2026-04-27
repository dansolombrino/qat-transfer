from copy import deepcopy
from types import MethodType
from typing import List

import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# Step 1 — inject MatMul modules into Attention layers
# ---------------------------------------------------------------------------

def inject_matmul_modules_(model: nn.Module) -> None:
    """Replace @ in timm Attention/WindowAttention with explicit MatMul modules."""
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
            module.fused_attn = False  # force explicit matmul so QuantMatMul can wrap it
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = MethodType(_vit_attention_forward, module)
        elif WindowAttention is not None and type(module) is WindowAttention:
            module.fused_attn = False
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = MethodType(_swin_window_attention_forward, module)


# ---------------------------------------------------------------------------
# Step 2 — recursively wrap Conv2d / Linear / MatMul with quantized versions
# ---------------------------------------------------------------------------

def _wrap_modules_(
    model: nn.Module,
    aq_params: dict,
    wq_params: dict,
    skip_modules: frozenset,
    _prefix: str = "",
) -> List[str]:
    """
    Recursively replace nn.Conv2d, nn.Linear, and MatMul with their quantized
    counterparts. Skips any child whose *attribute name* is in skip_modules.

    QKV, FC1, and reduction projections receive channel-wise activation
    quantization; all other linears receive tensor-wise activation quantization.
    Post-softmax matmul (matmul2) receives log-sqrt(2) activation quantization.
    """
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

        elif isinstance(m, nn.Linear):
            # QKV / first MLP linear / Swin downsampling: channel-wise activation quant
            if name in ('qkv', 'fc1', 'reduction'):
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
                _wrap_modules_(m, aq_params, wq_params, skip_modules, _prefix=full + ".")
            )

    return quantized


# ---------------------------------------------------------------------------
# Quant-state toggle
# ---------------------------------------------------------------------------

def _set_quant_state_(model: nn.Module, input_quant: bool, weight_quant: bool) -> None:
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)


# ---------------------------------------------------------------------------
# Step 4 — scale reparameterization (RepQ-ViT core contribution)
# ---------------------------------------------------------------------------

def _scale_reparameterize_(timm_model: nn.Module) -> None:
    """
    RepQ-ViT scale reparameterization (matches original test_quant.py).
    Iterates ViT/DeiT `blocks` or Swin `layers` and handles all three patterns
    in one pass:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_repqvit_(
    model: nn.Module,
    calib_data: torch.Tensor,
    w_bits: int,
    a_bits: int,
    skip_modules: frozenset,
    tqdm_kw: dict = None,
) -> List[str]:
    """
    Apply RepQ-ViT post-training quantization to a timm ViT/DeiT/Swin model
    wrapped in an ImageClassifier.

    Steps:
      1. Inject MatMul modules into Attention layers.
      2. Wrap Conv2d / Linear / MatMul with quantized counterparts.
      3. Initial calibration forward pass — initialises quantizer scales.
      4. Scale reparameterization — converts channel-wise to tensor-wise quant
         without accuracy loss.
      5. Re-calibration — re-initialises weight quantizers after reparameterization.

    Args:
        model:        The nn.Module (e.g. ImageClassifier).
        calib_data:   A calibration batch of images, shape (N, C, H, W).
        w_bits:       Bit-width for weights.
        a_bits:       Bit-width for activations.
        skip_modules: Frozenset of child attribute names to leave unquantized.
        tqdm_kw:      Extra kwargs forwarded to tqdm (e.g. disable, colour).

    Returns:
        List of fully-qualified names of every module that was quantized.
    """
    if tqdm_kw is None:
        tqdm_kw = {}

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
    inject_matmul_modules_(model)
    step_bar.update(1)

    # Step 2
    step_bar.set_postfix_str(steps[1])
    wq_params = {'n_bits': w_bits, 'channel_wise': True}
    aq_params = {'n_bits': a_bits, 'channel_wise': False}
    quantized_names = _wrap_modules_(model, aq_params, wq_params, skip_modules)
    step_bar.update(1)

    # Build name→module mapping for hook-based progress bars
    quant_types = (QuantConv2d, QuantLinear, QuantMatMul)
    quant_modules = {
        name: m for name, m in model.named_modules() if isinstance(m, quant_types)
    }

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
    # Reparameterization operates on the inner timm model (image_classifier.model)
    timm_model = model.model if hasattr(model, 'model') else model
    with torch.no_grad():
        _scale_reparameterize_(timm_model)
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
