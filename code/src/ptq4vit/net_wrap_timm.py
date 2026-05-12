"""PTQ4ViT timm model wrapping and public API."""

from collections import OrderedDict
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import PTQ4ViTConfig
from .net_wrap import MatMul
from .quant_calib import HessianQuantCalibrator, QuantCalibrator


# ---------------------------------------------------------------------------
# Patched attention forwards for timm
# ---------------------------------------------------------------------------

def _vit_attention_forward_ptq4vit(
    self, x: torch.Tensor, attn_mask=None, is_causal: bool = False
) -> torch.Tensor:
    B, N, _ = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _swin_window_attention_forward_ptq4vit(self, x: torch.Tensor, mask=None) -> torch.Tensor:
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
# Step 1: inject MatMul modules into timm Attention layers
# ---------------------------------------------------------------------------

def _inject_matmul_modules_timm_(model: nn.Module) -> None:
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
            module.forward = MethodType(_vit_attention_forward_ptq4vit, module)
        elif WindowAttention is not None and type(module) is WindowAttention:
            module.fused_attn = False
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = MethodType(_swin_window_attention_forward_ptq4vit, module)


# ---------------------------------------------------------------------------
# Step 2: wrap modules with PTQ4ViT quantized counterparts
# ---------------------------------------------------------------------------

# Timm attribute name → PTQ4ViT module type
_TIMM_MODULE_TYPES = {
    "qkv": "qlinear_qkv",
    "proj": "qlinear_proj",
    "fc1": "qlinear_MLP_1",
    "fc2": "qlinear_MLP_2",
    "reduction": "qlinear_proj",
    "matmul1": "qmatmul_qk",
    "matmul2": "qmatmul_scorev",
}


def _is_classifier_head(name: str, m: nn.Module) -> bool:
    if not isinstance(m, nn.Linear):
        return False
    return name.endswith("model.head") or name.endswith("model.head.fc")


def _wrap_modules_in_net_timm(
    model: nn.Module,
    ptq4vit_cfg: PTQ4ViTConfig,
    skip_modules: frozenset,
    device: torch.device,
) -> OrderedDict:
    """Walk the timm model and replace modules with PTQ4ViT quantized versions.

    Returns an OrderedDict of {full_name: quantized_module} in forward-pass order.
    """
    wrapped_modules = OrderedDict()
    module_dict = {}

    it = [(name, m) for name, m in model.named_modules()]
    for name, m in it:
        module_dict[name] = m

        # Get parent module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            continue

        attr_name = name[idx + 1 if idx != 0 else idx:]

        # Skip modules the user wants to exclude
        if attr_name in skip_modules:
            continue

        if isinstance(m, nn.Conv2d):
            new_m = ptq4vit_cfg.get_module(
                "qconv",
                m.in_channels, m.out_channels, m.kernel_size,
                m.stride, m.padding, m.dilation, m.groups,
                m.bias is not None, m.padding_mode,
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            wrapped_modules[name] = new_m
            setattr(father_module, attr_name, new_m)

        elif isinstance(m, nn.Linear):
            if _is_classifier_head(name, m):
                module_type = "qlinear_classifier"
            else:
                module_type = _TIMM_MODULE_TYPES.get(attr_name)
                if module_type is None:
                    continue
            new_m = ptq4vit_cfg.get_module(module_type, m.in_features, m.out_features)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            wrapped_modules[name] = new_m
            setattr(father_module, attr_name, new_m)

        elif isinstance(m, MatMul):
            module_type = _TIMM_MODULE_TYPES.get(attr_name)
            if module_type is None:
                continue
            new_m = ptq4vit_cfg.get_module(module_type)
            new_m._dev = device
            wrapped_modules[name] = new_m
            setattr(father_module, attr_name, new_m)

    print(f"Completed net wrap (timm). {len(wrapped_modules)} modules wrapped.")
    return wrapped_modules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_ptq4vit_timm_(
    model: nn.Module,
    calib_loader,
    cfg,
    skip_modules: frozenset,
    device: torch.device,
    tqdm_kw: dict = None,
) -> List[str]:
    """Apply PTQ4ViT post-training quantization to a timm model (ViT or Swin).

    Args:
        model:        Top-level nn.Module (e.g. ImageClassifier).
        calib_loader: DataLoader yielding (images, labels) tuples for calibration.
        cfg:          The ptq4vit sub-config (DictConfig or dict).
        skip_modules: Frozenset of child attribute names to leave unquantized.
        device:       Target device.
        tqdm_kw:      Extra kwargs for tqdm bars.

    Returns:
        List of fully-qualified names of every wrapped module.
    """
    if tqdm_kw is None:
        tqdm_kw = {}

    ptq4vit_cfg = PTQ4ViTConfig(cfg)

    steps = [
        "Inject MatMul modules",
        "Wrap modules",
        "Calibrate",
    ]
    step_bar = tqdm(steps, desc="PTQ4ViT (timm)", leave=False, **tqdm_kw)

    # Step 1: inject MatMul
    step_bar.set_postfix_str(steps[0])
    _inject_matmul_modules_timm_(model)
    step_bar.update(1)

    # Step 2: wrap modules
    step_bar.set_postfix_str(steps[1])
    wrapped_modules = _wrap_modules_in_net_timm(model, ptq4vit_cfg, skip_modules, device)
    step_bar.update(1)

    # Step 3: calibrate
    step_bar.set_postfix_str(steps[2])
    calibrator_type = str(cfg.calibrator)
    if calibrator_type == "hessian":
        calibrator = HessianQuantCalibrator(
            net=model,
            wrapped_modules=wrapped_modules,
            calib_loader=calib_loader,
            sequential=False,
            batch_size=int(cfg.hessian_batch_size),
            device=device,
        )
        calibrator.batching_quant_calib()
    elif calibrator_type == "basic":
        calibrator = QuantCalibrator(
            net=model,
            wrapped_modules=wrapped_modules,
            calib_loader=calib_loader,
            sequential=True,
            device=device,
        )
        calibrator.quant_calib()
    else:
        raise ValueError(f"Unknown calibrator: {calibrator_type!r}. Expected 'hessian' or 'basic'.")
    step_bar.update(1)

    step_bar.close()

    return list(wrapped_modules.keys())
