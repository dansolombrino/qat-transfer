import copy
import json
import os
import sys

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv

load_dotenv()

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from src.quantization import apply_ptq_, fake_quantize_tensor
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    maybe_dictionarize,
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_hf_clip.heads import get_classification_head as get_hf_classification_head
from src.vision.ilharco_hf_clip.modeling import (
    ImageClassifier as HFImageClassifier,
    ImageEncoder as HFImageEncoder,
)
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier as TimmImageClassifier
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_hf_model_name,
    sanitize_timm_model_name,
    set_seed,
)


def _default_skip_modules(model_family: str) -> frozenset[str]:
    if model_family == "ilharco_hf_clip":
        return frozenset(["classification_head"])
    if model_family == "ilharco_timm_supervised":
        return frozenset(["head"])
    raise ValueError(f"Unsupported model_family: {model_family}")


def _head_key_prefixes(model_family: str) -> Tuple[str, ...]:
    if model_family == "ilharco_hf_clip":
        return ("classification_head.",)
    if model_family == "ilharco_timm_supervised":
        return ("model.head.",)
    raise ValueError(f"Unsupported model_family: {model_family}")


def _is_head_key(key: str, head_prefixes: Tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in head_prefixes)


def _resolve_skip_modules(
    configured: Optional[Sequence[str]],
    model_family: str,
    force_head_exclusion: bool = False,
) -> frozenset[str]:
    """
    Resolve the set of modules to skip during rex transfer.

    Args:
        configured: User-configured sequence of module names to skip. If None,
            uses default skip modules for the model family.
        model_family: The model family (e.g., 'resnet', 'vit') used to determine
            default skip modules.
        force_head_exclusion: If True, always includes default skip modules in
            the result. If False, uses configured modules as-is when provided.

    Returns:
        A frozenset of module names that should be skipped during rex transfer.
        If configured is None, returns the default skip modules for the model family.
        If force_head_exclusion is True, returns the union of configured and default
        modules. Otherwise returns configured modules when provided.
    """
    default = set(_default_skip_modules(model_family))
    if configured is None:
        return frozenset(default)
    resolved = set(configured)
    if force_head_exclusion:
        resolved.update(default)
    return frozenset(resolved)


def _skip_tag(skip_modules: frozenset[str]) -> str:
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_tag(cfg: DictConfig) -> str:
    return (
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}"
    )


def _resolve_epochs(dataset_name: str, limit_num_epochs: Optional[int]) -> int:
    if limit_num_epochs is not None:
        return int(limit_num_epochs)
    return int(DATASET_NAME_TO_EPOCHS[dataset_name])


def _sanitize_model_name(model_family: str, model_name: str) -> str:
    if model_family == "ilharco_hf_clip":
        return sanitize_hf_model_name(model_name)
    if model_family == "ilharco_timm_supervised":
        return sanitize_timm_model_name(model_name)
    raise ValueError(f"Unsupported model_family: {model_family}")


def _checkpoint_path(
    cfg: DictConfig,
    dataset_name: str,
    seed: int,
    epochs: int,
) -> str:
    if cfg.model_family == "ilharco_hf_clip":
        return os.path.join(
            cfg.checkpoint_roots.hf_clip_fp,
            sanitize_hf_model_name(cfg.model_name),
            dataset_name,
            _optim_tag(cfg),
            f"seed={seed}",
            f"epoch_{epochs}.pt",
        )
    if cfg.model_family == "ilharco_timm_supervised":
        return os.path.join(
            cfg.checkpoint_roots.timm_supervised_fp,
            sanitize_timm_model_name(cfg.model_name),
            dataset_name,
            _optim_tag(cfg),
            f"seed={seed}",
            f"classifier_epoch_{epochs}.pt",
        )
    raise ValueError(f"Unsupported model_family: {cfg.model_family}")


def _load_classifier(
    cfg: DictConfig,
    dataset_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load a classification model based on the specified model family configuration.
    
    Supports loading classifiers for different model families:
    - "ilharco_hf_clip": Loads a Hugging Face CLIP-based image classifier with a 
      dataset-specific classification head. The head is looked up first in the 
      configured location, then in a derived path based on checkpoint storage root if needed.
    - "ilharco_timm_supervised": Loads a TIMM-based image classifier.
    
    Args:
        cfg (DictConfig): Configuration object containing model family, model name, 
            head base path, and checkpoint roots.
        dataset_name (str): Name of the dataset used to locate the appropriate 
            classification head.
        checkpoint_path (str): Path to the model checkpoint file to load.
        device (torch.device): Device on which to load the model.
    
    Returns:
        nn.Module: A loaded image classifier module (HFImageClassifier or 
            TimmImageClassifier).
    
    Raises:
        ValueError: If the specified model_family in cfg is not supported.
    """
    if cfg.model_family == "ilharco_hf_clip":
        configured_head_base_path = str(cfg.head_base_path)
        expected_head_path = os.path.join(
            configured_head_base_path,
            "vision",
            "ilharco_hf_clip",
            sanitize_hf_model_name(cfg.model_name),
            f"head_{dataset_name}.pt",
        )

        checkpoint_storage_root = Path(cfg.checkpoint_roots.hf_clip_fp).resolve().parents[3]
        derived_head_base_path = str(checkpoint_storage_root / "heads")
        derived_expected_head_path = os.path.join(
            derived_head_base_path,
            "vision",
            "ilharco_hf_clip",
            sanitize_hf_model_name(cfg.model_name),
            f"head_{dataset_name}.pt",
        )

        head_base_path = configured_head_base_path
        if not os.path.exists(expected_head_path) and os.path.exists(derived_expected_head_path):
            head_base_path = derived_head_base_path

        image_encoder = HFImageEncoder.load(
            model_name=cfg.model_name,
            filename=checkpoint_path,
            map_location="cpu",
        )
        classification_head = get_hf_classification_head(
            model_name=cfg.model_name,
            dataset_name=dataset_name,
            save_dir=head_base_path,
            device=device,
        )
        image_classifier = HFImageClassifier(
            image_encoder=image_encoder,
            classification_head=classification_head,
        )
        return image_classifier

    if cfg.model_family == "ilharco_timm_supervised":
        return TimmImageClassifier.load(
            model_name=cfg.model_name,
            num_classes=DATASET_NAME_TO_NUM_CLASSES[dataset_name],
            filename=checkpoint_path,
            map_location="cpu",
        )

    raise ValueError(f"Unsupported model_family: {cfg.model_family}")


def _evaluate(
    dataset,
    model: nn.Module,
    device: torch.device,
    limit_num_batches: Optional[int] = None,
    desc: str = "eval",
) -> float:
    loader = dataset.test_loader
    num_batches = len(loader)
    effective_num_batches = (
        min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches
    )

    model.to(device=device)
    model.eval()

    correct = 0.0
    total = 0
    batch_color = random_tqdm_color()

    with torch.no_grad():
        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc=desc,
            colour=batch_color,
            leave=False,
        )
        for i, batch in batch_bar:
            if i >= effective_num_batches:
                break

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device=device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = model(inputs)
            (top1,) = accuracy(logits, labels, topk=(1,))
            correct += top1
            total += labels.size(0)
            batch_bar.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")

    return float(correct / max(total, 1))


def _sparsify_by_output_channel(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity <= 0.0:
        return weight
    if sparsity >= 1.0:
        return torch.zeros_like(weight)

    if weight.ndim < 2:
        return weight

    out_channels = weight.shape[0]
    keep_fraction = 1.0 - sparsity
    keep_channels = int(round(keep_fraction * out_channels))
    keep_channels = max(0, min(out_channels, keep_channels))

    if keep_channels == out_channels:
        return weight
    if keep_channels == 0:
        return torch.zeros_like(weight)

    channel_norms = weight.abs().flatten(start_dim=1).sum(dim=1)
    topk_indices = torch.topk(
        channel_norms,
        k=keep_channels,
        largest=True,
        sorted=False,
    ).indices

    mask = torch.zeros(out_channels, dtype=torch.bool, device=weight.device)
    mask[topk_indices] = True
    view_shape = [out_channels] + [1] * (weight.ndim - 1)
    return weight * mask.view(view_shape)


def _rex_expand_weight(
    weight: torch.Tensor,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
) -> torch.Tensor:
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    residual = weight
    expanded = torch.zeros_like(weight)

    for k in range(order):
        quantized_residual = fake_quantize_tensor(residual, bits, granularity)
        if k > 0:
            quantized_residual = _sparsify_by_output_channel(
                quantized_residual,
                sparsity,
            )
        expanded = expanded + quantized_residual
        residual = residual - quantized_residual

    return expanded


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


def _build_rex_displacement(
    source_classifier: nn.Module,
    bits: int,
    granularity: str,
    order: int,
    sparsity: float,
    rex_skip_modules: frozenset[str],
    head_prefixes: Tuple[str, ...],
) -> tuple[Dict[str, torch.Tensor], Dict]:
    source_fp = source_classifier.to("cpu")
    source_rex = copy.deepcopy(source_fp).to("cpu")
    rex_quantized_layers = apply_rex_ptq_(
        model=source_rex,
        bits=bits,
        granularity=granularity,
        order=order,
        sparsity=sparsity,
        skip_modules=rex_skip_modules,
    )

    fp_sd = source_fp.state_dict()
    rex_sd = source_rex.state_dict()

    delta: Dict[str, torch.Tensor] = {}
    num_head_filtered = 0
    num_dtype_filtered = 0

    with torch.no_grad():
        for key, fp_value in fp_sd.items():
            if _is_head_key(key, head_prefixes):
                num_head_filtered += 1
                continue
            if fp_value.dtype in (torch.int64, torch.uint8):
                num_dtype_filtered += 1
                continue
            if key not in rex_sd:
                continue
            rex_value = rex_sd[key]
            if rex_value.shape != fp_value.shape:
                raise ValueError(
                    f"Shape mismatch while building REx displacement on key '{key}': "
                    f"fp={tuple(fp_value.shape)} rex={tuple(rex_value.shape)}"
                )
            delta[key] = rex_value - fp_value

    meta = {
        "num_keys_in_displacement": len(delta),
        "num_head_keys_excluded": num_head_filtered,
        "num_dtype_keys_excluded": num_dtype_filtered,
        "source_rex_quantized_layers_count": len(rex_quantized_layers),
        "source_rex_quantized_layers": rex_quantized_layers,
    }
    return delta, meta


def _build_patched_state_dict(
    target_state_dict: Dict[str, torch.Tensor],
    displacement: Dict[str, torch.Tensor],
    alpha: float,
    head_prefixes: Tuple[str, ...],
) -> tuple[Dict[str, torch.Tensor], int, int]:
    patched: Dict[str, torch.Tensor] = {}
    num_applied = 0

    with torch.no_grad():
        for key, target_value in target_state_dict.items():
            if _is_head_key(key, head_prefixes):
                patched[key] = target_value
                continue

            delta = displacement.get(key)
            if delta is None:
                patched[key] = target_value
                continue

            if delta.shape != target_value.shape:
                raise ValueError(
                    f"Shape mismatch while patching key '{key}': "
                    f"target={tuple(target_value.shape)} delta={tuple(delta.shape)}"
                )

            delta_aligned = delta.to(
                device=target_value.device,
                dtype=target_value.dtype,
            )
            patched[key] = target_value + float(alpha) * delta_aligned
            num_applied += 1

    num_missing_on_target = sum(1 for key in displacement if key not in target_state_dict)
    return patched, num_applied, num_missing_on_target


def _build_eval_dir(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    rex_skip_modules: frozenset[str],
    ptq_skip_modules: frozenset[str],
) -> str:
    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    model_tag = _sanitize_model_name(cfg.model_family, cfg.model_name)
    parts = [
        evaluation_base_path,
        "vision",
        "rex_transfer",
        cfg.model_family,
        model_tag,
        f"src={source_dataset_name}_seed={cfg.source.seed}",
        f"tgt={target_dataset_name}_seed={cfg.target.seed}",
        _optim_tag(cfg),
        (
            f"rex=bits={cfg.rex.bits}_order={cfg.rex.order}_sparsity={cfg.rex.sparsity}"
            f"_gran={cfg.rex.granularity}_skip={_skip_tag(rex_skip_modules)}"
        ),
        (
            f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}"
            f"_skip={_skip_tag(ptq_skip_modules)}"
        ),
    ]
    if (
        cfg.limit_num_batches is not None
        or cfg.source.limit_num_epochs is not None
        or cfg.target.limit_num_epochs is not None
    ):
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        src_lne = cfg.source.limit_num_epochs if cfg.source.limit_num_epochs is not None else "all"
        tgt_lne = cfg.target.limit_num_epochs if cfg.target.limit_num_epochs is not None else "all"
        parts.append(f"lnb={lnb}_src_lne={src_lne}_tgt_lne={tgt_lne}")
    return os.path.join(*parts)


@hydra.main(
    """
    Main entry point for REx transfer evaluation.

    This function orchestrates a comprehensive transfer evaluation pipeline
    that tests weight displacement transfer (REx) across multiple source-target dataset
    pairs with various alpha blending coefficients. It evaluates both full-precision (FP)
    and post-training quantized (PTQ) model variants.

    Args:
        cfg (DictConfig): Hydra configuration containing:
            - target: Target dataset configuration (dataset_names, seed, limit_num_epochs)
            - source: Source dataset configuration (dataset_names, seed, limit_num_epochs)
            - alphas: List of blending coefficients for displacement transfer
            - rex: REx configuration (bits, order, sparsity, granularity, skip_modules)
            - ptq: PTQ configuration (bits, granularity, skip_modules)
            - model_family: Model architecture family identifier
            - model_name: Specific model name
            - num_workers: Number of data loading workers
            - batch_size: Training/evaluation batch size
            - lr: Learning rate
            - wd: Weight decay
            - ls: Label smoothing
            - wl: Weight loss
            - max_grad_norm: Maximum gradient norm
            - limit_num_batches: Limit evaluation batches for testing
            - skip_missing_pairs: Whether to skip missing checkpoint pairs

    Returns:
        None. Results are saved as JSON files to the evaluation directory.

    Raises:
        ValueError: If dataset_names or alphas lists are empty
        FileNotFoundError: If required checkpoint files are missing and skip_missing_pairs is False
        RuntimeError: If no source->target pairs produce valid results

    Side Effects:
        - Sets random seed for reproducibility
        - Creates evaluation directories and writes JSON result files
        - Manages GPU memory by moving tensors to CPU and clearing cache
        - Prints progress and result summaries to stdout
    """
    config_path="../../../../config/experiments/vision/rex",
    config_name="rex_transfer",
    version_base=None,
)
def main(cfg: DictConfig):
    set_seed(int(cfg.target.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.num_workers = int(os.environ.get("TORCH_NUM_WORKERS", cfg.num_workers))

    source_datasets = [str(x) for x in cfg.source.dataset_names]
    if len(source_datasets) == 0:
        raise ValueError("cfg.source.dataset_names must contain at least one value")
    target_datasets = [str(x) for x in cfg.target.dataset_names]
    if len(target_datasets) == 0:
        raise ValueError("cfg.target.dataset_names must contain at least one value")
    alphas = [float(x) for x in cfg.alphas]
    if len(alphas) == 0:
        raise ValueError("cfg.alphas must contain at least one value")

    # User choice: always keep transfer backbone-only.
    rex_skip_modules = _resolve_skip_modules(
        configured=cfg.rex.skip_modules,
        model_family=cfg.model_family,
        force_head_exclusion=True,
    )
    ptq_skip_modules = _resolve_skip_modules(
        configured=cfg.ptq.skip_modules,
        model_family=cfg.model_family,
        force_head_exclusion=False,
    )
    head_prefixes = _head_key_prefixes(cfg.model_family)

    produced_pairs = 0
    skipped_pairs = 0
    total_pairs = len(source_datasets) * len(target_datasets)
    print(
        f"Starting REx transfer evaluation: total_pairs={total_pairs}, "
        f"rex_skip_modules={sorted(rex_skip_modules)}, "
        f"ptq_skip_modules={sorted(ptq_skip_modules)}"
    )

    for target_dataset_name in target_datasets:
        target_epochs = _resolve_epochs(
            dataset_name=target_dataset_name,
            limit_num_epochs=cfg.target.limit_num_epochs,
        )
        target_checkpoint_path = _checkpoint_path(
            cfg=cfg,
            dataset_name=target_dataset_name,
            seed=int(cfg.target.seed),
            epochs=target_epochs,
        )
        if not os.path.exists(target_checkpoint_path):
            msg = f"Missing target checkpoint: {target_checkpoint_path}"
            if cfg.skip_missing_pairs:
                print(f"[SKIP] {msg}")
                skipped_pairs += len(source_datasets)
                continue
            raise FileNotFoundError(msg)

        print(f"\n[Target] {target_dataset_name} trained for {target_epochs} epochs")
        print(f"Loading target checkpoint: {target_checkpoint_path}")

        target_classifier = _load_classifier(
            cfg=cfg,
            dataset_name=target_dataset_name,
            checkpoint_path=target_checkpoint_path,
            device=device,
        ).to("cpu")

        target_dataset = get_dataset(
            dataset_name=target_dataset_name,
            preprocess_train=target_classifier.train_preprocess,
            preprocess_inference=target_classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(cfg.num_workers),
            seed=cfg.target.seed,
        )

        target_fp_accuracy = _evaluate(
            dataset=target_dataset,
            model=target_classifier,
            device=device,
            limit_num_batches=cfg.limit_num_batches,
            desc=f"Target FP ({target_dataset_name})",
        )

        target_fp_ptq_model = copy.deepcopy(target_classifier).to("cpu")
        target_fp_ptq_layers = apply_ptq_(
            model=target_fp_ptq_model,
            bits=int(cfg.ptq.bits),
            granularity=cfg.ptq.granularity,
            skip_modules=ptq_skip_modules,
        )
        target_fp_ptq_accuracy = _evaluate(
            dataset=target_dataset,
            model=target_fp_ptq_model,
            device=device,
            limit_num_batches=cfg.limit_num_batches,
            desc=f"Target FP+PTQ ({target_dataset_name})",
        )
        del target_fp_ptq_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Keep target state_dict on CPU so patching works regardless of eval device.
        target_classifier.to("cpu")
        target_state_dict = target_classifier.state_dict()
        num_classes = len(target_dataset.class_names)
        random_chance = 1.0 / num_classes

        for source_dataset_name in source_datasets:
            source_epochs = _resolve_epochs(
                dataset_name=source_dataset_name,
                limit_num_epochs=cfg.source.limit_num_epochs,
            )
            source_checkpoint_path = _checkpoint_path(
                cfg=cfg,
                dataset_name=source_dataset_name,
                seed=int(cfg.source.seed),
                epochs=source_epochs,
            )
            if not os.path.exists(source_checkpoint_path):
                msg = f"Missing source checkpoint: {source_checkpoint_path}"
                if cfg.skip_missing_pairs:
                    print(f"[SKIP] {msg}")
                    skipped_pairs += 1
                    continue
                raise FileNotFoundError(msg)

            print(
                f"\n[Pair] source={source_dataset_name} -> target={target_dataset_name}\n"
                f"Loading source checkpoint: {source_checkpoint_path} trained for {source_epochs} epochs"
            )

            source_classifier = _load_classifier(
                cfg=cfg,
                dataset_name=source_dataset_name,
                checkpoint_path=source_checkpoint_path,
                device=device,
            ).to("cpu")

            displacement, displacement_meta = _build_rex_displacement(
                source_classifier=source_classifier,
                bits=int(cfg.rex.bits),
                granularity=cfg.rex.granularity,
                order=int(cfg.rex.order),
                sparsity=float(cfg.rex.sparsity),
                rex_skip_modules=rex_skip_modules,
                head_prefixes=head_prefixes,
            )

            alpha_results: List[Dict] = []
            for alpha in alphas:
                patched_classifier = copy.deepcopy(target_classifier).to("cpu")
                patched_state_dict, num_applied, num_missing = _build_patched_state_dict(
                    target_state_dict=target_state_dict,
                    displacement=displacement,
                    alpha=alpha,
                    head_prefixes=head_prefixes,
                )
                patched_classifier.load_state_dict(patched_state_dict)

                patched_fp_accuracy = _evaluate(
                    dataset=target_dataset,
                    model=patched_classifier,
                    device=device,
                    limit_num_batches=cfg.limit_num_batches,
                    desc=(
                        f"REx transfer FP src={source_dataset_name} "
                        f"tgt={target_dataset_name} a={alpha:g}"
                    ),
                )

                patched_ptq_layers = apply_ptq_(
                    model=patched_classifier,
                    bits=int(cfg.ptq.bits),
                    granularity=cfg.ptq.granularity,
                    skip_modules=ptq_skip_modules,
                )
                patched_fp_ptq_accuracy = _evaluate(
                    dataset=target_dataset,
                    model=patched_classifier,
                    device=device,
                    limit_num_batches=cfg.limit_num_batches,
                    desc=(
                        f"REx transfer FP+PTQ src={source_dataset_name} "
                        f"tgt={target_dataset_name} a={alpha:g}"
                    ),
                )

                alpha_results.append(
                    {
                        "alpha": float(alpha),
                        "test_accuracy_patched_fp": patched_fp_accuracy,
                        "test_accuracy_patched_fp_ptq": patched_fp_ptq_accuracy,
                        "delta_vs_target_fp": patched_fp_accuracy - target_fp_accuracy,
                        "delta_vs_target_fp_ptq": patched_fp_ptq_accuracy - target_fp_ptq_accuracy,
                        "num_displacement_keys_applied_to_target": num_applied,
                        "num_displacement_keys_missing_on_target": num_missing,
                        "ptq_quantized_layers_count": len(patched_ptq_layers),
                        "ptq_quantized_layers": patched_ptq_layers,
                    }
                )

                del patched_classifier
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            eval_dir = _build_eval_dir(
                cfg=cfg,
                source_dataset_name=source_dataset_name,
                target_dataset_name=target_dataset_name,
                rex_skip_modules=rex_skip_modules,
                ptq_skip_modules=ptq_skip_modules,
            )
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, "eval_results.json")

            payload = {
                "experiment": "rex_transfer_weight_only",
                "model_family": cfg.model_family,
                "model_name": cfg.model_name,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "wd": cfg.wd,
                "ls": cfg.ls,
                "wl": cfg.wl,
                "max_grad_norm": cfg.max_grad_norm,
                "limit_num_batches": cfg.limit_num_batches,
                "source": {
                    "dataset_name": source_dataset_name,
                    "seed": int(cfg.source.seed),
                    "limit_num_epochs": cfg.source.limit_num_epochs,
                    "epochs": source_epochs,
                    "fp_checkpoint_path": source_checkpoint_path,
                },
                "target": {
                    "dataset_name": target_dataset_name,
                    "seed": int(cfg.target.seed),
                    "limit_num_epochs": cfg.target.limit_num_epochs,
                    "epochs": target_epochs,
                    "fp_checkpoint_path": target_checkpoint_path,
                },
                "rex": {
                    "bits": int(cfg.rex.bits),
                    "order": int(cfg.rex.order),
                    "sparsity": float(cfg.rex.sparsity),
                    "granularity": cfg.rex.granularity,
                    "skip_modules": sorted(rex_skip_modules),
                },
                "ptq": {
                    "bits": int(cfg.ptq.bits),
                    "granularity": cfg.ptq.granularity,
                    "skip_modules": sorted(ptq_skip_modules),
                },
                "alphas": [float(x) for x in alphas],
                "target_baselines": {
                    "test_accuracy_target_fp": target_fp_accuracy,
                    "test_accuracy_target_fp_ptq": target_fp_ptq_accuracy,
                    "target_fp_ptq_quantized_layers_count": len(target_fp_ptq_layers),
                    "target_fp_ptq_quantized_layers": target_fp_ptq_layers,
                    "num_classes": num_classes,
                    "random_chance": random_chance,
                },
                "displacement": displacement_meta,
                "results": alpha_results,
            }

            with open(eval_path, "w") as f:
                json.dump(payload, f, indent=2)

            print(f"Results saved to: {eval_path}")
            produced_pairs += 1

            del source_classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del target_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if produced_pairs == 0:
        raise RuntimeError(
            "No source->target pair produced results. "
            "Check dataset lists, checkpoint roots, seeds, and optimizer path fragment."
        )

    print(
        f"\nCompleted REx transfer evaluation: produced_pairs={produced_pairs}, "
        f"skipped_pairs={skipped_pairs}, total_requested_pairs={total_pairs}"
    )


if __name__ == "__main__":
    main()
