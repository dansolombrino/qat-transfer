import copy
from types import SimpleNamespace
from typing import Iterable, List, Mapping

import torch
from torch import nn

from .utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from .utils.calibrator import QuantCalibrator
from .utils.mlp_recon import MLPReconstructor
from .utils.block_recon import BlockReconstructor


SUPPORTED_MODELS = (
    # From the reference's model_zoo (test_quant.py:180-194).
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_base_patch4_window12_384",
    # Project-specific additions. Same architectural families as above
    # (timm.models.vision_transformer.VisionTransformer with `Attention`, or
    # timm.models.swin_transformer.SwinTransformer with `WindowAttention`),
    # so `wrap_modules_in_net` finds the same hookable modules.
    "vit_huge_patch14_224",
    "deit3_base_patch16_224",
    "deit3_large_patch16_224",
    "swin_large_patch4_window7_224",
)


class TupleLoaderWrapper:
    """Adapts a DataLoader producing dict-style or tuple-style batches into the
    `(images, labels)` tuple form expected by the ported APHQ-ViT reconstructors
    and calibrator."""

    def __init__(self, loader, maybe_dictionarize):
        self._loader = loader
        self._maybe_dictionarize = maybe_dictionarize

    def __iter__(self):
        for batch in self._loader:
            batch = self._maybe_dictionarize(batch)
            yield batch["images"], batch["labels"]

    def __len__(self):
        return len(self._loader)


def _check_supported(model_name: str) -> None:
    # timm model names carry an optional pretrained-tag suffix after the first
    # dot (e.g. `vit_base_patch16_224.orig_in21k`). The architecture is fully
    # determined by the part before the dot, so we match on that.
    base = model_name.split(".", 1)[0]
    if base not in SUPPORTED_MODELS:
        raise NotImplementedError(
            f"APHQ-ViT only supports ViT/DeiT/Swin variants from its model_zoo; "
            f"got {model_name!r} (architecture {base!r}). "
            f"Supported architectures: {SUPPORTED_MODELS}"
        )


def _build_cfg_namespace(aphq_cfg) -> SimpleNamespace:
    """Convert a flat OmegaConf-style sub-config into the attribute-access
    `Config`-like object that the ported APHQ-ViT modules expect."""
    return SimpleNamespace(
        w_bit=int(aphq_cfg.w_bit),
        a_bit=int(aphq_cfg.a_bit),
        qconv_a_bit=int(aphq_cfg.qconv_a_bit),
        qhead_a_bit=int(aphq_cfg.qhead_a_bit),
        calib_size=int(aphq_cfg.calib_size),
        calib_batch_size=int(aphq_cfg.calib_batch_size),
        calib_metric=str(aphq_cfg.calib_metric),
        optim_size=int(aphq_cfg.optim_size),
        optim_batch_size=int(aphq_cfg.optim_batch_size),
        optim_metric=str(aphq_cfg.optim_metric),
        optim_mode=str(aphq_cfg.optim_mode),
        drop_prob=float(aphq_cfg.drop_prob),
        use_mean_hessian=bool(aphq_cfg.use_mean_hessian),
        temp=float(aphq_cfg.temp),
        keep_gpu=bool(aphq_cfg.keep_gpu),
        recon_metric=str(aphq_cfg.recon_metric),
        pct=float(aphq_cfg.pct),
        eq_n=int(aphq_cfg.eq_n),
        search_round=int(aphq_cfg.search_round),
        matmul_head_channel_wise=bool(aphq_cfg.matmul_head_channel_wise),
        token_channel_wise=bool(aphq_cfg.token_channel_wise),
    )


def apply_aphq_vit_(
    image_classifier: nn.Module,
    model_name: str,
    calib_loader,
    optim_loader,
    aphq_cfg,
    skip_modules: Iterable[str],
    device: torch.device,
) -> List[str]:
    """Run the full APHQ-ViT pipeline on `image_classifier` in-place.

    Stages:
      1. (optional) MLP reconstruction: GELU -> ReLU + Hessian-perturb fine-tune.
      2. Wrap modules with APHQ-ViT quant layers + MSE-guided calibration.
      3. (optional) AdaRound block reconstruction with QDrop.

    Args:
        image_classifier:  Top-level wrapper (e.g. ImageClassifier). Its
            `.model` attribute must be the underlying timm ViT/DeiT/Swin model.
        model_name:        Original timm model name; checked against the
            APHQ-ViT supported set.
        calib_loader:      DataLoader yielding `(images, labels)` tuples for
            Stage 2 (QuantCalibrator). Should have ~`calib_size` samples.
        optim_loader:      DataLoader yielding `(images, labels)` tuples for
            Stages 1 and 3 (MLPReconstructor / BlockReconstructor). Should
            have ~`optim_size` samples.
        aphq_cfg:          Flat sub-config (e.g. `cfg.aphq_vit`).
        skip_modules:      Substrings of fully-qualified module names to leave
            unquantized (e.g. `{"head"}` to keep the classifier head in FP).
        device:            Target device.

    Returns:
        List of fully-qualified names of modules that were wrapped in quant
        layers (after the post-calibration reparam pass).
    """
    _check_supported(model_name)
    skip_modules = frozenset(skip_modules)
    ns = _build_cfg_namespace(aphq_cfg)

    # The reconstructors and wrappers operate on the timm submodule directly:
    # they look up timm-specific attributes (`.mlp`, `.norm1`, `.blocks`,
    # `Attention` instances). Our `ImageClassifier` wrapper is transparent
    # via `named_modules()`, but to stay byte-for-byte faithful to
    # `references/aphq_vit/code/test_quant.py:200-247` we hand them
    # `image_classifier.model` directly.
    inner = image_classifier.model
    full_model = copy.deepcopy(inner).to(device)
    full_model.eval()
    inner.to(device)
    inner.eval()

    # Stage 1 -- MLP reconstruction.
    if bool(aphq_cfg.reconstruct_mlp):
        for name, module in inner.named_modules():
            if name.split(".")[-1] == "mlp":
                module.act = nn.ReLU()
        mlp_reconstructor = MLPReconstructor(
            inner,
            full_model,
            optim_loader,
            metric=ns.recon_metric,
            temp=ns.temp,
        )
        mlp_reconstructor.reconstruct_model(pct=ns.pct)

    # Stage 2 -- wrap modules + MSE calibration.
    inner, _ = wrap_modules_in_net(
        inner,
        ns,
        reparam=True,
        recon=bool(aphq_cfg.reconstruct_mlp),
        skip_modules=skip_modules,
    )
    inner.to(device)
    inner.eval()

    quant_calibrator = QuantCalibrator(inner, calib_loader)
    quant_calibrator.batching_quant_calib()

    inner = wrap_reparamed_modules_in_net(inner)
    inner.to(device)
    inner.eval()

    # Stage 3 -- AdaRound block reconstruction.
    if bool(aphq_cfg.optimize):
        block_reconstructor = BlockReconstructor(
            inner,
            full_model,
            optim_loader,
            metric=ns.optim_metric,
            temp=ns.temp,
            use_mean_hessian=ns.use_mean_hessian,
        )
        block_reconstructor.reconstruct_model(
            quant_act=True,
            mode=ns.optim_mode,
            drop_prob=ns.drop_prob,
            keep_gpu=ns.keep_gpu,
        )

    # Re-link the (possibly mutated) inner model back into the wrapper. In
    # practice the wrappers mutate `inner` in place, but wrap_reparamed_*
    # returns a new reference; reassigning is defensive.
    image_classifier.model = inner

    quantized_names = [
        name for name, _module in inner.named_modules()
        if name and any(
            cls in type(_module).__name__
            for cls in (
                "AsymmetricallyBatchingQuantLinear",
                "AsymmetricallyChannelWiseBatchingQuantLinear",
                "AsymmetricallyBatchingQuantConv2d",
                "AsymmetricallyBatchingQuantMatMul",
            )
        )
    ]
    return quantized_names
