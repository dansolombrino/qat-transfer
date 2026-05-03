import copy
import json
import os
import sys

from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv
load_dotenv()

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from src.quantization import RexLinear, apply_ptq_, fake_quantize_tensor
from src.vision.data.common import DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES, maybe_dictionarize
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


def _optim_tag(cfg: DictConfig) -> str:
    return (
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}"
    )


def _resolve_epochs(cfg: DictConfig) -> int:
    if cfg.limit_num_epochs is not None:
        return int(cfg.limit_num_epochs)
    return int(DATASET_NAME_TO_EPOCHS[cfg.dataset_name])


def _checkpoint_path(cfg: DictConfig, epochs: int) -> str:
    if cfg.model_family == "ilharco_hf_clip":
        return os.path.join(
            cfg.checkpoint_roots.hf_clip_fp,
            sanitize_hf_model_name(cfg.model_name),
            cfg.dataset_name,
            _optim_tag(cfg),
            f"seed={cfg.seed}",
            f"epoch_{epochs}.pt",
        )
    if cfg.model_family == "ilharco_timm_supervised":
        return os.path.join(
            cfg.checkpoint_roots.timm_supervised_fp,
            sanitize_timm_model_name(cfg.model_name),
            cfg.dataset_name,
            _optim_tag(cfg),
            f"seed={cfg.seed}",
            f"classifier_epoch_{epochs}.pt",
        )
    raise ValueError(f"Unsupported model_family: {cfg.model_family}")


def _sanitize_model_name(model_family: str, model_name: str) -> str:
    if model_family == "ilharco_hf_clip":
        return sanitize_hf_model_name(model_name)
    if model_family == "ilharco_timm_supervised":
        return sanitize_timm_model_name(model_name)
    raise ValueError(f"Unsupported model_family: {model_family}")


def _load_classifier(cfg: DictConfig, checkpoint_path: str, device: torch.device) -> nn.Module:
    if cfg.model_family == "ilharco_hf_clip":
        configured_head_base_path = str(cfg.head_base_path)
        expected_head_path = os.path.join(
            configured_head_base_path,
            "vision",
            "ilharco_hf_clip",
            sanitize_hf_model_name(cfg.model_name),
            f"head_{cfg.dataset_name}.pt",
        )
        checkpoint_storage_root = Path(cfg.checkpoint_roots.hf_clip_fp).resolve().parents[3]
        derived_head_base_path = str(checkpoint_storage_root / "heads")
        derived_expected_head_path = os.path.join(
            derived_head_base_path,
            "vision",
            "ilharco_hf_clip",
            sanitize_hf_model_name(cfg.model_name),
            f"head_{cfg.dataset_name}.pt",
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
            dataset_name=cfg.dataset_name,
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
            num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
            filename=checkpoint_path,
            map_location="cpu",
        )

    raise ValueError(f"Unsupported model_family: {cfg.model_family}")


def _build_dataset(cfg: DictConfig, classifier: nn.Module):
    return get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(cfg.num_workers),
        seed=cfg.seed,
    )


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
        min(limit_num_batches, num_batches)
        if limit_num_batches is not None
        else num_batches
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
            top1, = accuracy(logits, labels, topk=(1,))
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


def _effective_budget_multiplier(order: int, sparsity: float) -> float:
    return 1.0 + float(order - 1) * (1.0 - float(sparsity))


def _build_eval_dir(cfg: DictConfig, skip_modules: frozenset[str]) -> str:
    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    skip_tag = "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"
    model_tag = _sanitize_model_name(cfg.model_family, cfg.model_name)
    parts = [
        evaluation_base_path,
        "vision",
        "rex",
        cfg.model_family,
        model_tag,
        cfg.dataset_name,
        _optim_tag(cfg),
        f"seed={cfg.seed}",
        f"order={cfg.rex.order}",
        f"granularity={cfg.rex.granularity}",
        f"skip={skip_tag}",
    ]
    if cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        parts.append(f"lnb={lnb}_lne={lne}")
    return os.path.join(*parts)


@hydra.main(
    config_path="../../../../config/experiments/vision/rex",
    config_name="evaluate_rex",
    version_base=None,
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.num_workers = int(os.environ.get("TORCH_NUM_WORKERS", cfg.num_workers))

    skip_modules = (
        frozenset(cfg.skip_modules)
        if cfg.skip_modules is not None
        else _default_skip_modules(cfg.model_family)
    )

    epochs = _resolve_epochs(cfg)
    checkpoint_path = _checkpoint_path(cfg, epochs)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    base_classifier = _load_classifier(cfg, checkpoint_path, device=device)
    base_classifier.to(device)

    dataset = _build_dataset(cfg, base_classifier)
    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes

    ptq_reference: Dict[float, float] = {}
    results: List[Dict] = []

    for bits in cfg.rex.bits:
        bits = int(bits)
        baseline_model = copy.deepcopy(base_classifier).to("cpu")
        ptq_layers = apply_ptq_(
            model=baseline_model,
            bits=bits,
            granularity=cfg.rex.granularity,
            skip_modules=skip_modules,
        )
        ptq_acc = _evaluate(
            dataset=dataset,
            model=baseline_model,
            device=device,
            limit_num_batches=cfg.limit_num_batches,
            desc=f"PTQ b={bits}",
        )
        ptq_reference[float(bits)] = ptq_acc
        results.append(
            {
                "method": "ptq",
                "bits": bits,
                "effective_weight_bit_budget": float(bits),
                "budget_multiplier_vs_single_ptq": 1.0,
                "order": 1,
                "sparsity": 1.0,
                "accuracy": ptq_acc,
                "quantized_layers_count": len(ptq_layers),
                "quantized_layers": ptq_layers,
            }
        )
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for sparsity in cfg.rex.sparsity:
            sparsity = float(sparsity)
            rex_model = copy.deepcopy(base_classifier).to("cpu")
            rex_layers = apply_rex_ptq_terms_(
                model=rex_model,
                bits=bits,
                granularity=cfg.rex.granularity,
                order=int(cfg.rex.order),
                sparsity=sparsity,
                skip_modules=skip_modules,
            )
            rex_acc = _evaluate(
                dataset=dataset,
                model=rex_model,
                device=device,
                limit_num_batches=cfg.limit_num_batches,
                desc=f"REx b={bits} s={sparsity:.2f}",
            )
            budget_multiplier = _effective_budget_multiplier(
                order=int(cfg.rex.order),
                sparsity=sparsity,
            )
            effective_budget = float(bits) * budget_multiplier
            equal_budget_baseline = ptq_reference.get(effective_budget)
            results.append(
                {
                    "method": "rex",
                    "bits": bits,
                    "effective_weight_bit_budget": effective_budget,
                    "budget_multiplier_vs_single_ptq": budget_multiplier,
                    "order": int(cfg.rex.order),
                    "sparsity": sparsity,
                    "accuracy": rex_acc,
                    "delta_vs_ptq_same_bits": rex_acc - ptq_acc,
                    "ptq_equal_budget_accuracy": equal_budget_baseline,
                    "delta_vs_ptq_equal_budget": (
                        None
                        if equal_budget_baseline is None
                        else rex_acc - equal_budget_baseline
                    ),
                    "quantized_layers_count": len(rex_layers),
                    "quantized_layers": rex_layers,
                }
            )
            del rex_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    eval_dir = _build_eval_dir(cfg, skip_modules=skip_modules)
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(eval_dir, "eval_results.json")

    payload = {
        "experiment": "rex_weight_only",
        "model_family": cfg.model_family,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "limit_num_epochs": cfg.limit_num_epochs,
        "limit_num_batches": cfg.limit_num_batches,
        "epochs": epochs,
        "device": str(device),
        "checkpoint_path": checkpoint_path,
        "num_classes": num_classes,
        "random_chance": random_chance,
        "granularity": cfg.rex.granularity,
        "skip_modules": sorted(skip_modules),
        "bits": [int(x) for x in cfg.rex.bits],
        "order": int(cfg.rex.order),
        "sparsity": [float(x) for x in cfg.rex.sparsity],
        "results": results,
    }

    with open(eval_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to: {eval_path}")


if __name__ == "__main__":
    main()
