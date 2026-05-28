"""Steered PTQ evaluation: apply weight-only PTQ to a finetuned timm ViT, load
the per-block steering vectors fitted by fit_steering_vector.py, then sweep
(method, block_index, alpha) and record val + test accuracy for each setting.

The plain-PTQ baseline (alpha=0) is evaluated once and replicated across the
sweep so we never run the same forward twice. Best (method*, block*, alpha*)
is selected on the val split; the corresponding test accuracy is reported.
"""

import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import (
    DATASET_NAME_TO_NUM_CLASSES,
    DATASET_NAME_TO_EPOCHS,
    maybe_dictionarize,
)
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.quantization import apply_ptq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


def _evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    limit_num_batches: int = None,
    desc: str = "eval",
) -> float:
    model.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches
    correct, total = 0, 0
    with torch.no_grad():
        bar = tqdm(
            enumerate(loader),
            total=effective,
            desc=desc,
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        for i, batch in bar:
            if i >= effective:
                break
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(device=device)
            labels = batch['labels'].to(device=device, dtype=torch.long)
            logits = model(inputs)
            top1, = accuracy(logits, labels, topk=(1,))
            correct += top1
            total += labels.size(0)
    return correct / total if total > 0 else float('nan')


def _attach_steering_hook(block_module, vector: torch.Tensor, alpha: float):
    """Register a forward hook that adds alpha * vector to the block's output,
    broadcast across batch and token positions. Returns the handle."""
    v_bcast = vector.view(1, 1, -1)

    def _hook(_mod, _inp, out):
        return out + alpha * v_bcast

    return block_module.register_forward_hook(_hook)


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/002_quant_steering",
    config_name="evaluate_steered_ptq",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    epochs = DATASET_NAME_TO_EPOCHS[
        cfg.dataset_name
    ] if cfg.limit_num_epochs is None else cfg.limit_num_epochs

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    ############################################################################
    # BEGIN checkpoint loading
    ############################################################################

    checkpoint_base_path = os.environ['CHECKPOINT_BASE_PATH']

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    checkpoint_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "fp_dryrun" if is_dryrun else "fp",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        checkpoint_dir_parts.append(f"lnb={lnb}_lne={lne}")
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")
    head_path = os.path.join(checkpoint_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading encoder from: {classifier_path}")
    image_classifier = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        filename=classifier_path,
    )
    image_classifier.to(device)

    assert hasattr(image_classifier.model, 'blocks'), (
        "Expected timm ViT with .model.blocks; got "
        f"{type(image_classifier.model).__name__}"
    )
    blocks = list(image_classifier.model.blocks)
    num_blocks = len(blocks)

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_classifier.train_preprocess,
        preprocess_inference=image_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP baseline (val + test, before PTQ)
    ############################################################################

    print("\nFP baseline (no quantization):")
    fp_val_acc = _evaluate_loader(
        image_classifier, dataset.val_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="FP (val)",
    )
    fp_test_acc = _evaluate_loader(
        image_classifier, dataset.test_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="FP (test)",
    )
    print(f"  val={fp_val_acc * 100:.2f}%  test={fp_test_acc * 100:.2f}%")

    ############################################################################
    # END FP baseline
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN PTQ (in-place fake-quantize)
    ############################################################################

    skip_modules = frozenset(cfg.ptq.skip_modules)

    quantized_names = apply_ptq_(
        model=image_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )
    print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, "
          f"skip_modules={list(cfg.ptq.skip_modules)}")
    print(f"  Quantized layers: {len(quantized_names)}")

    ############################################################################
    # END PTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN load steering vectors
    ############################################################################

    skip_modules_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    if cfg.steering.vectors_path is not None:
        vectors_path = cfg.steering.vectors_path
    else:
        vectors_dir_parts = [
            checkpoint_base_path,
            "vision",
            "ilharco_timm_supervised",
            "steering_vectors",
            sanitize_timm_model_name(cfg.model_name),
            cfg.dataset_name,
            f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
            f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_modules_tag}",
            f"seed={cfg.seed}",
        ]
        if is_dryrun:
            lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
            lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
            vectors_dir_parts.append(f"lnb={lnb}_lne={lne}")
        vectors_path = os.path.join(*vectors_dir_parts, "steering_vectors.pt")

    print(f"\nLoading steering vectors from: {vectors_path}")
    payload = torch.load(vectors_path, map_location='cpu')

    assert payload['num_blocks'] == num_blocks, (
        f"steering vectors fitted for num_blocks={payload['num_blocks']} but "
        f"model has {num_blocks}"
    )

    vectors_by_method = {
        'mean_diff': payload['mean_diff'].to(device),
        'contrastive_svd': payload['contrastive_svd'].to(device),
    }

    ############################################################################
    # END load steering vectors
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN sweep
    ############################################################################

    methods = list(cfg.steering.methods)
    for m in methods:
        if m not in vectors_by_method:
            raise ValueError(
                f"Unknown steering method {m!r}; expected one of "
                f"{sorted(vectors_by_method.keys())}"
            )

    if cfg.steering.block_sweep == "all":
        block_indices = list(range(num_blocks))
    else:
        block_indices = [int(b) for b in cfg.steering.block_sweep]
    alpha_grid = [float(a) for a in cfg.steering.alpha_grid]

    print(f"\nSweep size: methods={len(methods)} × blocks={len(block_indices)} "
          f"× alphas={len(alpha_grid)} = {len(methods) * len(block_indices) * len(alpha_grid)}")

    # Plain-PTQ baseline (alpha=0) is identical across (method, block); compute once.
    print("\nPlain PTQ baseline (alpha=0):")
    plain_val_acc = _evaluate_loader(
        image_classifier, dataset.val_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="plain-PTQ (val)",
    )
    plain_test_acc = _evaluate_loader(
        image_classifier, dataset.test_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="plain-PTQ (test)",
    )
    print(f"  val={plain_val_acc * 100:.2f}%  test={plain_test_acc * 100:.2f}%")

    sweep_results = []

    for method in methods:
        v_per_block = vectors_by_method[method]
        for bi in block_indices:
            v = v_per_block[bi]
            for alpha in alpha_grid:
                if alpha == 0.0:
                    val_acc, test_acc = plain_val_acc, plain_test_acc
                else:
                    handle = _attach_steering_hook(blocks[bi], v, alpha)
                    try:
                        val_acc = _evaluate_loader(
                            image_classifier, dataset.val_loader, device,
                            limit_num_batches=cfg.limit_num_batches,
                            desc=f"{method}/b={bi}/a={alpha:+.2f} (val)",
                        )
                        test_acc = _evaluate_loader(
                            image_classifier, dataset.test_loader, device,
                            limit_num_batches=cfg.limit_num_batches,
                            desc=f"{method}/b={bi}/a={alpha:+.2f} (test)",
                        )
                    finally:
                        handle.remove()
                sweep_results.append({
                    'method': method,
                    'block': bi,
                    'alpha': alpha,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                })
                print(f"  [{method}] block={bi:>2}  α={alpha:+.3f}  "
                      f"val={val_acc * 100:.2f}%  test={test_acc * 100:.2f}%")

    ############################################################################
    # END sweep
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN best selection
    ############################################################################

    best = max(sweep_results, key=lambda r: r['val_acc'])
    test_gain = best['test_acc'] - plain_test_acc
    ptq_drop_from_fp = fp_test_acc - plain_test_acc

    print()
    print(f"  FP             :  test={fp_test_acc * 100:.2f}%")
    print(f"  Plain PTQ      :  test={plain_test_acc * 100:.2f}%  "
          f"(Δ vs FP = {-ptq_drop_from_fp * 100:+.2f}pp)")
    print(f"  Best by val    :  method={best['method']}  block={best['block']}  "
          f"α={best['alpha']:+.3f}  val={best['val_acc'] * 100:.2f}%  "
          f"test={best['test_acc'] * 100:.2f}%  "
          f"(Δ vs plain PTQ = {test_gain * 100:+.2f}pp)")

    ############################################################################
    # END best selection
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save results
    ############################################################################

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "002_quant_steering",
        "vision",
        "fp_ptq_steered",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_modules_tag}",
        f"seed={cfg.seed}",
    )

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes

    results = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "limit_num_epochs": cfg.limit_num_epochs,
        "limit_num_batches": cfg.limit_num_batches,
        "epochs": epochs,
        "device": str(device),
        "num_classes": num_classes,
        "random_chance": random_chance,
        "encoder_path": classifier_path,
        "head_path": head_path,
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "quantized_layers": quantized_names,
        "steering_vectors_path": vectors_path,
        "steering_methods": methods,
        "steering_block_sweep": block_indices,
        "steering_alpha_grid": alpha_grid,
        "fp_val_accuracy": fp_val_acc,
        "fp_test_accuracy": fp_test_acc,
        "plain_ptq_val_accuracy": plain_val_acc,
        "plain_ptq_test_accuracy": plain_test_acc,
        "ptq_drop_from_fp": ptq_drop_from_fp,
        "sweep_results": sweep_results,
        "best_by_val": best,
        "test_accuracy": best['test_acc'],
        "test_gain_over_plain_ptq": test_gain,
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {eval_results_path}")

    ############################################################################
    # END save results
    ############################################################################


if __name__ == "__main__":
    main()
