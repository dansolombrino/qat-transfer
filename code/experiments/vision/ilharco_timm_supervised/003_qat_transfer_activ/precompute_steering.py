# ==============================================================================
# Activation steering vector — precompute step (003_qat_transfer_activ)
# ==============================================================================
# This is the activation-space analog of the weight-space quantization vector
# (QV) used in 001_qat_transfer. Instead of differencing checkpoints in weight
# space (QV = QAT_src - FP_src), we run BOTH the FP and QAT donor checkpoints
# over the donor (source) VAL split in FULL PRECISION, capture per-layer
# activations, and difference them into a steering vector:
#
#     S[tap] = mean_over_val( act_qat[tap] ) - mean_over_val( act_fp[tap] )
#
# "Full precision" mirrors the weight-space QV, which never quantizes either
# side: QAT checkpoints are saved with their QAT wrappers stripped (plain
# nn.Linear), so loading and running one is an ordinary fp forward pass whose
# weights happen to carry the QAT-trained signature.
#
# The cache stores the PER-TOKEN displacement [tokens, dim] per tap (a
# reduce-agnostic superset: mean-over-tokens of it recovers the [dim] vector),
# so the runner can switch token_reduce={per_token,mean} without recomputing.
# The cached vectors are consumed by qv_transfer.py in this same directory.
#
# This script never quantizes and never touches the target/receiver datasets.
# ==============================================================================

import gc
import json
import logging
import os
import sys

from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import (
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.vision.steering import select_tap_modules, ActivationMeanCapture

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={seed}",
    )


def _qat_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int) -> str:
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "qat",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
    )


def _steering_dir(cfg: DictConfig, source_dataset_name: str) -> str:
    """Cache location for the steering vectors of one source/donor.

    No ptq fragment (donor is full precision), no target, and no token_reduce
    (the cached per-token vector is reduce-agnostic).
    """
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "steering",
        sanitize_timm_model_name(cfg.model_name),
        source_dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"strategy={cfg.steering_strategy}",
        f"seed={cfg.source.seed}",
    )


def _capture_means(classifier, taps, loader, device, limit_num_batches):
    """Run `classifier` over `loader` (val split) and return per-token activation means."""

    num_batches = len(loader)
    effective_num_batches = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    classifier.to(device)
    classifier.eval()

    batch_color = random_tqdm_color()

    with ActivationMeanCapture(taps) as cap:
        with torch.no_grad():
            batch_bar = tqdm(
                enumerate(loader),
                total=effective_num_batches,
                desc="Capturing activations",
                colour=batch_color,
                leave=False,
                **TQDM_KW,
            )
            for i, batch in batch_bar:
                if i >= effective_num_batches:
                    break
                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to(device=device)
                classifier(inputs)
                batch_bar.set_postfix(batch=f"{i}/{effective_num_batches}")

    return cap.result(), cap.num_samples


def _run_one_source(cfg: DictConfig, source_dataset_name: str, device: torch.device):
    """Compute and cache the steering vectors for a single source/donor dataset."""

    src_epochs = DATASET_NAME_TO_EPOCHS[
        source_dataset_name
    ] if cfg.source.limit_num_epochs is None else cfg.source.limit_num_epochs

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed)

    fp_source_path = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    qat_source_path = os.path.join(qat_src_dir, f"classifier_epoch_{src_epochs}.pt")

    if IS_SLURM:
        log.info("--- source=%s ---", source_dataset_name)
        log.info("FP source  classifier: %s", fp_source_path)
        log.info("QAT source classifier: %s", qat_source_path)
    else:
        print(f"\n--- source={source_dataset_name} ---")
        print(f"FP source  classifier: {fp_source_path}")
        print(f"QAT source classifier: {qat_source_path}")

    for path in (fp_source_path, qat_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    ############################################################################
    # END checkpoint paths
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation (source / donor)
    ############################################################################

    num_classes = DATASET_NAME_TO_NUM_CLASSES[source_dataset_name]

    _tmp_classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    dataset = get_dataset(
        dataset_name=source_dataset_name,
        preprocess_train=_tmp_classifier.train_preprocess,
        preprocess_inference=_tmp_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.source.seed,
    )
    del _tmp_classifier

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN capture FP donor activations (full precision)
    ############################################################################

    classifier_fp = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier_fp.load_state_dict(torch.load(fp_source_path, map_location="cpu"))
    fp_taps = select_tap_modules(classifier_fp, cfg.steering_strategy)
    mean_fp, n_fp = _capture_means(classifier_fp, fp_taps, dataset.val_loader, device, cfg.limit_num_batches)
    del classifier_fp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END capture FP donor activations
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN capture QAT donor activations (full precision)
    ############################################################################

    classifier_qat = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier_qat.load_state_dict(torch.load(qat_source_path, map_location="cpu"))
    qat_taps = select_tap_modules(classifier_qat, cfg.steering_strategy)
    mean_qat, n_qat = _capture_means(classifier_qat, qat_taps, dataset.val_loader, device, cfg.limit_num_batches)
    del classifier_qat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert n_fp == n_qat, f"sample-count mismatch between FP ({n_fp}) and QAT ({n_qat}) donor passes"

    ############################################################################
    # END capture QAT donor activations
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN steering vector (qat_act - fp_act) and cache
    ############################################################################

    steering = OrderedDict()
    for name in mean_fp:
        steering[name] = mean_qat[name] - mean_fp[name]

    tap_names = list(steering.keys())
    tokens, dim = (tuple(steering[tap_names[0]].shape) if tap_names else (None, None))

    steer_dir = _steering_dir(cfg, source_dataset_name)
    os.makedirs(steer_dir, exist_ok=True)
    steering_path = os.path.join(steer_dir, "steering_vectors.pt")
    torch.save(steering, steering_path)

    meta = {
        "experiment": "precompute_steering",
        "model_name": cfg.model_name,
        "steering_strategy": cfg.steering_strategy,
        "qat_donor_mode": "full_precision",
        "cache_layout": "per_token [tokens, dim] (reduce-agnostic; token_reduce applied at inject time)",
        "batch_size": cfg.batch_size,
        "limit_num_batches": cfg.limit_num_batches,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "device": str(device),
        "source": {
            "dataset_name": source_dataset_name,
            "seed": cfg.source.seed,
            "limit_num_epochs": cfg.source.limit_num_epochs,
            "epochs": src_epochs,
            "num_classes": num_classes,
            "fp_classifier_path": fp_source_path,
            "qat_classifier_path": qat_source_path,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "num_taps": len(tap_names),
        "tap_names": tap_names,
        "tokens": tokens,
        "dim": dim,
        "num_val_samples": n_fp,
        "steering_vectors_path": steering_path,
    }
    with open(os.path.join(steer_dir, "steering_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if IS_SLURM:
        log.info(
            "steering cached: %d taps, [tokens=%s, dim=%s], %d val samples -> %s",
            len(tap_names), tokens, dim, n_fp, steering_path,
        )
    else:
        print(
            f"\nsteering cached: {len(tap_names)} taps, [tokens={tokens}, dim={dim}], "
            f"{n_fp} val samples\n  -> {steering_path}\n"
        )

    ############################################################################
    # END steering vector and cache
    ############################################################################

    del dataset, mean_fp, mean_qat, steering
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/003_qat_transfer_activ",
    config_name="precompute_steering",
    version_base=None,
)
def main(cfg: DictConfig):

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)

    set_seed(cfg.source.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    for si, source_dataset_name in enumerate(source_dataset_names):

        if IS_SLURM:
            log.info("=== Source %d/%d: %s ===", si + 1, len(source_dataset_names), source_dataset_name)
        else:
            print(f"\n{'='*60}")
            print(f"  Source {si + 1}/{len(source_dataset_names)}: {source_dataset_name}")
            print(f"{'='*60}")

        _run_one_source(cfg=cfg, source_dataset_name=source_dataset_name, device=device)

    if IS_SLURM:
        log.info("All %d source(s) completed. Forcing exit.", len(source_dataset_names))
        os._exit(0)


if __name__ == "__main__":
    main()
