"""PTQ4ViT open_clip model wrapping and public API."""

from collections import OrderedDict
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import PTQ4ViTConfig
from .mha_wrapper import FusedQKVQuantLinear
from .quant_calib import HessianQuantCalibrator, QuantCalibrator
from .quant_conv import MinMaxQuantConv2d
from .quant_linear import MinMaxQuantLinear
from .quant_matmul import MinMaxQuantMatMul


# ---------------------------------------------------------------------------
# MatMul placeholder (same as RepQ-ViT and PTQ4ViT reference)
# ---------------------------------------------------------------------------

class MatMul(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B


# ---------------------------------------------------------------------------
# Patched MHA forward for PTQ4ViT
# ---------------------------------------------------------------------------

def _open_clip_mha_forward_ptq4vit(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
):
    B, L, D = query.shape
    head_dim = D // self.num_heads
    scale = head_dim ** -0.5

    # QKV via the FusedQKVQuantLinear (mode-aware)
    qkv = self.qkv_quant(query)

    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, L, self.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, L, self.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, L, self.num_heads, head_dim).transpose(1, 2)

    attn = self.matmul1(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)

    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, L, D)
    x = self.out_proj(x)

    return x, None


# ---------------------------------------------------------------------------
# Step 1: inject MatMul modules into MHA layers
# ---------------------------------------------------------------------------

def _inject_matmul_modules_(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = MethodType(_open_clip_mha_forward_ptq4vit, module)


# ---------------------------------------------------------------------------
# Step 2: wrap modules with PTQ4ViT quantized counterparts
# ---------------------------------------------------------------------------

# Open_clip attribute name → PTQ4ViT module type
_OPEN_CLIP_MODULE_TYPES = {
    "out_proj": "qlinear_proj",
    "matmul1": "qmatmul_qk",
    "matmul2": "qmatmul_scorev",
    "c_fc": "qlinear_MLP_1",
    "c_proj": "qlinear_MLP_2",
}


def _wrap_modules_in_net(
    model: nn.Module,
    ptq4vit_cfg: PTQ4ViTConfig,
    skip_modules: frozenset,
    device: torch.device,
) -> OrderedDict:
    """Walk the open_clip model and replace modules with PTQ4ViT quantized versions.

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

        elif isinstance(m, nn.MultiheadAttention):
            # Create FusedQKVQuantLinear for the fused in_proj_weight
            qkv_kwargs = dict(ptq4vit_cfg.ptqsl_linear_kwargs)
            qkv_kwargs["n_V"] = qkv_kwargs.get("n_V", 1) * 3  # triple for Q,K,V
            qkv_quant = FusedQKVQuantLinear(
                m,
                w_bit=ptq4vit_cfg.w_bit["qlinear_qkv"],
                a_bit=ptq4vit_cfg.a_bit["qlinear_qkv"],
                **qkv_kwargs,
            )
            m.qkv_quant = qkv_quant
            wrapped_modules[name + ".qkv_quant"] = qkv_quant
            # out_proj, matmul1, matmul2 will be wrapped when we encounter them

        elif isinstance(m, nn.Linear):
            module_type = _OPEN_CLIP_MODULE_TYPES.get(attr_name)
            if module_type is None:
                continue
            new_m = ptq4vit_cfg.get_module(module_type, m.in_features, m.out_features)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            wrapped_modules[name] = new_m
            setattr(father_module, attr_name, new_m)

        elif isinstance(m, MatMul):
            module_type = _OPEN_CLIP_MODULE_TYPES.get(attr_name)
            if module_type is None:
                continue
            new_m = ptq4vit_cfg.get_module(module_type)
            new_m._dev = device
            wrapped_modules[name] = new_m
            setattr(father_module, attr_name, new_m)

    print(f"Completed net wrap. {len(wrapped_modules)} modules wrapped.")
    return wrapped_modules


# ---------------------------------------------------------------------------
# Calibration data adapter
# ---------------------------------------------------------------------------

class _TupleLoaderWrapper:
    """Wraps a DataLoader that yields dicts to yield (images, labels) tuples."""

    def __init__(self, loader, dictionarize_fn):
        self._loader = loader
        self._dictionarize_fn = dictionarize_fn

    def __iter__(self):
        for batch in self._loader:
            batch = self._dictionarize_fn(batch)
            yield batch['images'], batch['labels']

    def __len__(self):
        return len(self._loader)

    @property
    def batch_size(self):
        return self._loader.batch_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_ptq4vit_(
    model: nn.Module,
    calib_loader,
    cfg,
    skip_modules: frozenset,
    device: torch.device,
    tqdm_kw: dict = None,
) -> List[str]:
    """Apply PTQ4ViT post-training quantization to an open_clip model.

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
    step_bar = tqdm(steps, desc="PTQ4ViT", leave=False, **tqdm_kw)

    # Step 1: inject MatMul
    step_bar.set_postfix_str(steps[0])
    _inject_matmul_modules_(model)
    step_bar.update(1)

    # Step 2: wrap modules
    step_bar.set_postfix_str(steps[1])
    wrapped_modules = _wrap_modules_in_net(model, ptq4vit_cfg, skip_modules, device)
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
