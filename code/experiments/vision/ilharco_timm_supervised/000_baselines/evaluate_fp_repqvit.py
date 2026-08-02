"""Evaluate RepQ-ViT on receiver-task FP checkpoints (rebuttal WP4).

Reviewers asked for a ViT-native strong PTQ baseline rather than only vanilla
weight-only RTN. RepQ-ViT specifically addresses the extreme post-LayerNorm
and post-Softmax activation distributions of vision transformers, so this
script applies the official W/A calibration and scale reparameterization to
each receiver's FP checkpoint before measuring held-out accuracy.

Calibration is one deterministically sampled batch from that receiver's
training split, matching the official classification protocol. The identical
batch construction is used by 006_qat_transfer_repqvit, making this baseline
an end-to-end cross-check of each receiver's alpha=0 transfer cell.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm

from src.repqvit import apply_repqvit_
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    make_seeded_loader,
    maybe_dictionarize,
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)


log = logging.getLogger(__name__)
IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


def _materialize_calibration_batch(
    dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> torch.Tensor:
    # The loader generator controls sampling and worker RNGs. Reset the main
    # process as well so num_workers=0 produces the same augmented images in
    # this baseline and the transfer phase's alpha=0 cross-check.
    set_seed(seed)
    loader = make_seeded_loader(
        dataset=dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Receiver training split is empty; cannot calibrate RepQ-ViT") from exc
    return maybe_dictionarize(batch)["images"].cpu()


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    limit_num_batches: int | None = None,
) -> float:
    if split == "test":
        loader = dataset.test_loader
    elif split == "val":
        loader = dataset.val_loader
    else:
        raise ValueError(f"Unsupported split {split!r}; expected 'val' or 'test'")

    num_batches = len(loader)
    effective_num_batches = (
        min(limit_num_batches, num_batches)
        if limit_num_batches is not None
        else num_batches
    )
    model.to(device=device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc=f"Evaluating ({split})",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
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
            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / total:.2f}%",
            )
    if total == 0:
        raise RuntimeError(f"No samples were evaluated on split={split!r}")
    return float(correct / total)


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/000_baselines",
    config_name="evaluate_fp_repqvit",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    if cfg.split not in ("val", "test"):
        raise ValueError(f"Unsupported split {cfg.split!r}; expected 'val' or 'test'")

    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])
    epochs = (
        DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
        if cfg.limit_num_epochs is None
        else cfg.limit_num_epochs
    )
    is_dryrun = cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None

    ########################################################################
    # BEGIN checkpoint loading
    ########################################################################

    checkpoint_dir_parts = [
        os.environ["CHECKPOINT_BASE_PATH"],
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

    image_classifier = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        filename=classifier_path,
    )
    image_classifier.to(device=device, dtype=torch.float32)

    ########################################################################
    # END checkpoint loading
    ########################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ########################################################################
    # BEGIN receiver data and RepQ-ViT calibration
    ########################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_classifier.train_preprocess,
        preprocess_inference=image_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )
    calib_data = _materialize_calibration_batch(
        dataset=dataset,
        batch_size=cfg.repqvit.calib_batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )
    quantized_names = apply_repqvit_(
        model=image_classifier,
        calib_data=calib_data,
        w_bits=cfg.repqvit.w_bits,
        a_bits=cfg.repqvit.a_bits,
        skip_modules=frozenset(cfg.repqvit.skip_modules),
    )
    if IS_SLURM:
        log.info(
            "RepQ-ViT W%s/A%s calibrated on %d receiver-train samples; quantized %d modules",
            cfg.repqvit.w_bits,
            cfg.repqvit.a_bits,
            calib_data.shape[0],
            len(quantized_names),
        )
    else:
        print(
            f"\nRepQ-ViT W{cfg.repqvit.w_bits}/A{cfg.repqvit.a_bits}: "
            f"calibration_samples={calib_data.shape[0]}, "
            f"quantized_modules={len(quantized_names)}\n"
        )

    ########################################################################
    # END receiver data and RepQ-ViT calibration
    ########################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    accuracy_value = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        split=cfg.split,
        limit_num_batches=cfg.limit_num_batches,
    )
    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes

    skip_tag = (
        "-".join(sorted(cfg.repqvit.skip_modules))
        if len(cfg.repqvit.skip_modules) > 0
        else "none"
    )
    repqvit_frag = (
        f"repqvit=wbits={cfg.repqvit.w_bits}_abits={cfg.repqvit.a_bits}"
        f"_skip={skip_tag}_cbs={cfg.repqvit.calib_batch_size}"
    )
    eval_dir_parts = [
        os.environ["EVALUATION_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "fp_repqvit_dryrun" if is_dryrun else "fp_repqvit",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        repqvit_frag,
        f"seed={cfg.seed}",
    ]
    if cfg.split != "test":
        eval_dir_parts.append(f"split={cfg.split}")
    eval_dir = os.path.join(*eval_dir_parts)

    results = {
        "experiment": "fp_repqvit",
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
        "eval_split": cfg.split,
        f"{cfg.split}_accuracy": accuracy_value,
        "random_chance": random_chance,
        "num_classes": num_classes,
        "encoder_path": classifier_path,
        "head_path": head_path,
        "repqvit": {
            "w_bits": cfg.repqvit.w_bits,
            "a_bits": cfg.repqvit.a_bits,
            "skip_modules": list(cfg.repqvit.skip_modules),
            "calib_batch_size": cfg.repqvit.calib_batch_size,
            "actual_calibration_samples": int(calib_data.shape[0]),
            "calibration_split": "train",
        },
        "repqvit_quantized_modules": quantized_names,
        "alpha_zero_crosscheck": (
            "Must match the self-pair alpha=0 cell in 006_qat_transfer_repqvit "
            "for the same receiver/configuration."
        ),
    }
    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    if IS_SLURM:
        log.info("%s_accuracy=%s; results=%s", cfg.split, accuracy_value, eval_results_path)
    else:
        print(f"\n{cfg.split}_accuracy (RepQ-ViT): {accuracy_value}")
        print(f"Results saved to: {eval_results_path}")


if __name__ == "__main__":
    main()
