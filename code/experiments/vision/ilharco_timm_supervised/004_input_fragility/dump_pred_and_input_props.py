"""Dump per-sample FP / PTQ predictions + input properties for fragility analysis.

For each val (and test) sample, this script records:

  Identity & labels
    sample_idx, label

  FP-pass scalars (model BEFORE quantization)
    fp_pred, fp_logit_label, fp_logit_top1, fp_logit_top2,
    fp_margin (top1 − top2), fp_softmax_top1, fp_entropy,
    fp_cls_dist_to_class_centroid  (in the model's pre-head representation)

  Q-pass scalars (same model AFTER in-place weight-only PTQ)
    q_pred, q_logit_top1, q_logit_top2, q_margin

  Input-image properties (computed on the preprocessed image tensor)
    img_brightness   = mean over (C, H, W)
    img_contrast     = std over (C, H, W)
    img_edge_density = mean |Sobel-x| + |Sobel-y| of the grayscale image
    img_high_freq_ratio = high-frequency-energy / total-energy of the 2D-FFT

Derived flags (computed downstream in the analyzer)
    fp_correct, q_correct, good = FP-c ∩ Q-c, bad = FP-c ∩ Q-w

Saves one Parquet per split. Tiny — ~15 scalars × N samples, megabytes.
"""

import json
import logging
import math
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
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.quantization import apply_ptq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


def _make_sobel_kernels(device):
    sx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
    ).view(1, 1, 3, 3)
    sy = sx.transpose(2, 3).contiguous()
    return sx, sy


def _image_properties(images: torch.Tensor, sobel_x, sobel_y) -> dict:
    """Per-sample stats on the preprocessed image tensor (B, C, H, W).
    All computations stay on whatever device images is on; results returned
    on CPU as 1-D tensors of length B."""

    B, C, H, W = images.shape
    brightness = images.mean(dim=(1, 2, 3)).cpu()
    contrast = images.std(dim=(1, 2, 3)).cpu()

    # Edge density via Sobel on the grayscale (channel-averaged) image.
    gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    edge_density = (gx.abs() + gy.abs()).mean(dim=(1, 2, 3)).cpu()

    # FFT high-frequency-energy ratio. Distance-from-origin > 0.25 of Nyquist
    # is the outer band; this is a coarse "how much high-frequency content" stat.
    gray2d = gray.squeeze(1)  # (B, H, W)
    fft = torch.fft.rfft2(gray2d)
    energy = (fft.real ** 2 + fft.imag ** 2)  # (B, H, W//2 + 1)
    Hf, Wf = energy.shape[-2:]
    freqs_y = torch.fft.fftfreq(H, device=images.device).abs().view(-1, 1)
    freqs_x = torch.fft.rfftfreq(W, device=images.device).abs().view(1, -1)
    radial = (freqs_y ** 2 + freqs_x ** 2).sqrt()  # (Hf, Wf)
    high_mask = (radial > 0.25).float()
    high_energy = (energy * high_mask).sum(dim=(1, 2))
    total_energy = energy.sum(dim=(1, 2)).clamp_min(1e-12)
    high_freq_ratio = (high_energy / total_energy).cpu()

    return {
        "img_brightness": brightness,
        "img_contrast": contrast,
        "img_edge_density": edge_density,
        "img_high_freq_ratio": high_freq_ratio,
    }


def _forward_fp(model, loader, device, sobel_x, sobel_y, limit_num_batches=None, desc="FP"):
    """Returns a dict of numpy arrays of length N (number of samples), keyed by
    column name. Includes labels, FP scalars, image properties, and CLS
    embeddings (kept in memory until centroids are computed, then dropped)."""

    model.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    chunks: dict[str, list[torch.Tensor]] = {
        "label": [], "fp_pred": [],
        "fp_logit_label": [], "fp_logit_top1": [], "fp_logit_top2": [],
        "fp_margin": [], "fp_softmax_top1": [], "fp_entropy": [],
        "img_brightness": [], "img_contrast": [],
        "img_edge_density": [], "img_high_freq_ratio": [],
        # underscore = held only in memory for cross-model derived features;
        # dropped before parquet save (see main()).
        "_cls_embedding": [],
        "_fp_logits": [],
    }

    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=effective, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for i, batch in bar:
            if i >= effective:
                break
            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device=device, non_blocking=True)
            labels = batch['labels'].to(dtype=torch.long)

            # Image properties (no grad path needed; stay on device for speed).
            img_props = _image_properties(images, sobel_x, sobel_y)
            for k, v in img_props.items():
                chunks[k].append(v)

            # Pre-head pooled representation for centroid distance.
            features = model.model.forward_features(images)
            pooled = model.model.forward_head(features, pre_logits=True)  # (B, D)
            logits = model.model.head(pooled)                              # (B, C)

            preds = logits.argmax(dim=1)
            B, C = logits.shape

            # top-1 / top-2
            top2_vals, top2_idx = logits.topk(2, dim=1)
            fp_top1 = top2_vals[:, 0]
            fp_top2 = top2_vals[:, 1]

            logp = F.log_softmax(logits, dim=1)
            p = logp.exp()
            entropy = -(p * logp).sum(dim=1)
            sm_top1 = p.gather(1, preds.unsqueeze(1)).squeeze(1)

            # Logit at the true label index.
            label_logit = logits.gather(1, labels.to(device).unsqueeze(1)).squeeze(1)

            chunks["label"].append(labels)
            chunks["fp_pred"].append(preds.cpu())
            chunks["fp_logit_label"].append(label_logit.cpu())
            chunks["fp_logit_top1"].append(fp_top1.cpu())
            chunks["fp_logit_top2"].append(fp_top2.cpu())
            chunks["fp_margin"].append((fp_top1 - fp_top2).cpu())
            chunks["fp_softmax_top1"].append(sm_top1.cpu())
            chunks["fp_entropy"].append(entropy.cpu())
            chunks["_cls_embedding"].append(pooled.cpu())
            chunks["_fp_logits"].append(logits.cpu())

    out = {k: torch.cat(v, dim=0).numpy() for k, v in chunks.items()}
    return out


def _forward_q(model, loader, device, limit_num_batches=None, desc="Q"):
    """After PTQ has been applied in place. Returns Q-side scalars; same row
    order as the FP pass (deterministic loader, no shuffle)."""

    model.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    chunks: dict[str, list[torch.Tensor]] = {
        "q_pred": [], "q_logit_top1": [], "q_logit_top2": [], "q_margin": [],
        "q_softmax_top1": [], "q_entropy": [],
        "_q_logits": [],  # held in memory for cross-model features; dropped before save
    }

    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=effective, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for i, batch in bar:
            if i >= effective:
                break
            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device=device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            top2_vals, _ = logits.topk(2, dim=1)
            q_top1 = top2_vals[:, 0]
            q_top2 = top2_vals[:, 1]

            logp = F.log_softmax(logits, dim=1)
            p = logp.exp()
            q_entropy = -(p * logp).sum(dim=1)
            q_sm_top1 = p.gather(1, preds.unsqueeze(1)).squeeze(1)

            chunks["q_pred"].append(preds.cpu())
            chunks["q_logit_top1"].append(q_top1.cpu())
            chunks["q_logit_top2"].append(q_top2.cpu())
            chunks["q_margin"].append((q_top1 - q_top2).cpu())
            chunks["q_softmax_top1"].append(q_sm_top1.cpu())
            chunks["q_entropy"].append(q_entropy.cpu())
            chunks["_q_logits"].append(logits.cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in chunks.items()}


def _cross_model_features(fp_logits: np.ndarray, q_logits: np.ndarray,
                          fp_pred: np.ndarray, q_pred: np.ndarray) -> dict:
    """Per-sample cross-FP/Q features that distinguish 'bad' from 'lucky-Q'.

    For a 'bad' input (FP-correct, Q-wrong), Q typically picks FP's runner-up
    class — so fp_logit_at_q_pred is close to fp_logit_top2 (i.e. FP gives Q's
    answer high credence). For a 'lucky-Q' input (FP-wrong, Q-correct), the
    true class can be deep in FP's ranking — fp_logit_at_q_pred is small and
    fp_softmax_at_q_pred near zero. These features therefore predict the
    cross-model agreement pattern that distinguishes the two cases.
    """
    fp_logits = fp_logits.astype(np.float64)
    q_logits = q_logits.astype(np.float64)
    n = fp_logits.shape[0]

    # Direct gather of logits at the OTHER model's predicted class
    idx = np.arange(n)
    fp_logit_at_q_pred = fp_logits[idx, q_pred]
    q_logit_at_fp_pred = q_logits[idx, fp_pred]

    # Softmax-normalised versions (probability the OTHER model's answer was
    # under THIS model's distribution).
    fp_logits_shifted = fp_logits - fp_logits.max(axis=1, keepdims=True)
    fp_exp = np.exp(fp_logits_shifted)
    fp_softmax = fp_exp / fp_exp.sum(axis=1, keepdims=True)
    fp_softmax_at_q_pred = fp_softmax[idx, q_pred]

    q_logits_shifted = q_logits - q_logits.max(axis=1, keepdims=True)
    q_exp = np.exp(q_logits_shifted)
    q_softmax = q_exp / q_exp.sum(axis=1, keepdims=True)
    q_softmax_at_fp_pred = q_softmax[idx, fp_pred]

    # KL(fp || q) on softmax (symmetrised average of the two directions).
    eps = 1e-12
    kl_fp_q = (fp_softmax * (np.log(fp_softmax + eps) - np.log(q_softmax + eps))).sum(axis=1)
    kl_q_fp = (q_softmax * (np.log(q_softmax + eps) - np.log(fp_softmax + eps))).sum(axis=1)
    fp_q_kl_symmetric = 0.5 * (kl_fp_q + kl_q_fp)

    fp_q_disagree = (fp_pred != q_pred).astype(np.float32)

    return {
        "fp_logit_at_q_pred":   fp_logit_at_q_pred.astype(np.float32),
        "q_logit_at_fp_pred":   q_logit_at_fp_pred.astype(np.float32),
        "fp_softmax_at_q_pred": fp_softmax_at_q_pred.astype(np.float32),
        "q_softmax_at_fp_pred": q_softmax_at_fp_pred.astype(np.float32),
        "fp_q_kl_symmetric":    fp_q_kl_symmetric.astype(np.float32),
        "fp_q_disagree":        fp_q_disagree,
    }


def _compute_centroid_distances(cls_embeddings: np.ndarray, labels: np.ndarray, num_classes: int):
    """Per-sample Euclidean distance from the sample's pre-head representation
    to the mean pre-head representation of its true class. Returns (N,) array.
    Classes with zero samples are handled by reporting nan for their members
    (impossible by construction, but guarded against)."""

    centroids = np.zeros((num_classes, cls_embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    np.add.at(centroids, labels, cls_embeddings.astype(np.float64))
    np.add.at(counts, labels, 1)
    nonzero = counts > 0
    centroids[nonzero] /= counts[nonzero, None]

    distances = np.full(cls_embeddings.shape[0], np.nan, dtype=np.float64)
    for c in np.where(nonzero)[0]:
        idx = np.where(labels == c)[0]
        diffs = cls_embeddings[idx] - centroids[c]
        distances[idx] = np.linalg.norm(diffs, axis=1)
    return distances.astype(np.float32), centroids.astype(np.float32), counts


def _to_dataframe(fp: dict, q: dict, cross: dict, dist_to_centroid: np.ndarray) -> pd.DataFrame:
    n = len(fp["label"])
    df = pd.DataFrame({
        "sample_idx": np.arange(n, dtype=np.int64),
        "label": fp["label"].astype(np.int32),
        # FP-side scalars
        "fp_pred": fp["fp_pred"].astype(np.int32),
        "fp_logit_label": fp["fp_logit_label"].astype(np.float32),
        "fp_logit_top1": fp["fp_logit_top1"].astype(np.float32),
        "fp_logit_top2": fp["fp_logit_top2"].astype(np.float32),
        "fp_margin": fp["fp_margin"].astype(np.float32),
        "fp_softmax_top1": fp["fp_softmax_top1"].astype(np.float32),
        "fp_entropy": fp["fp_entropy"].astype(np.float32),
        "fp_cls_dist_to_class_centroid": dist_to_centroid.astype(np.float32),
        # Q-side scalars
        "q_pred": q["q_pred"].astype(np.int32),
        "q_logit_top1": q["q_logit_top1"].astype(np.float32),
        "q_logit_top2": q["q_logit_top2"].astype(np.float32),
        "q_margin": q["q_margin"].astype(np.float32),
        "q_softmax_top1": q["q_softmax_top1"].astype(np.float32),
        "q_entropy": q["q_entropy"].astype(np.float32),
        # Cross-model (FP↔Q) features
        "fp_logit_at_q_pred": cross["fp_logit_at_q_pred"],
        "q_logit_at_fp_pred": cross["q_logit_at_fp_pred"],
        "fp_softmax_at_q_pred": cross["fp_softmax_at_q_pred"],
        "q_softmax_at_fp_pred": cross["q_softmax_at_fp_pred"],
        "fp_q_kl_symmetric": cross["fp_q_kl_symmetric"],
        "fp_q_disagree": cross["fp_q_disagree"],
        # Image-pixel statistics
        "img_brightness": fp["img_brightness"].astype(np.float32),
        "img_contrast": fp["img_contrast"].astype(np.float32),
        "img_edge_density": fp["img_edge_density"].astype(np.float32),
        "img_high_freq_ratio": fp["img_high_freq_ratio"].astype(np.float32),
    })
    df["fp_correct"] = (df["fp_pred"] == df["label"])
    df["q_correct"] = (df["q_pred"] == df["label"])
    df["good"] = df["fp_correct"] & df["q_correct"]
    df["bad"] = df["fp_correct"] & ~df["q_correct"]
    return df


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="dump_pred_and_input_props",
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
    is_dryrun = cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None

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

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_classifier.train_preprocess,
        preprocess_inference=image_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.seed,
    )
    num_classes = DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]

    ############################################################################
    # END dataset
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP pass over val + test (collect predictions, embeddings, img props)
    ############################################################################

    sobel_x, sobel_y = _make_sobel_kernels(device)

    print("\nFP pass — val:")
    fp_val = _forward_fp(
        image_classifier, dataset.val_loader, device, sobel_x, sobel_y,
        limit_num_batches=cfg.limit_num_batches, desc="FP val",
    )
    print(f"  collected {len(fp_val['label'])} samples")

    print("\nFP pass — test:")
    fp_test = _forward_fp(
        image_classifier, dataset.test_loader, device, sobel_x, sobel_y,
        limit_num_batches=cfg.limit_num_batches, desc="FP test",
    )
    print(f"  collected {len(fp_test['label'])} samples")

    ############################################################################
    # END FP pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN class centroid distances
    ############################################################################

    # Centroids are computed from val so the per-sample distance on test is
    # against a held-out estimate.  Val samples are distance-to-their-own-set
    # centroid (slight in-sample bias; with N >> num_classes it's negligible
    # but we note it).

    print("\nComputing class centroids on val, distances on val + test ...")
    val_cls = fp_val.pop("_cls_embedding")
    test_cls = fp_test.pop("_cls_embedding")

    val_dist, centroids, val_counts = _compute_centroid_distances(
        val_cls, fp_val["label"], num_classes,
    )

    # Distance on test uses val-derived centroids
    test_dist = np.full(test_cls.shape[0], np.nan, dtype=np.float64)
    nonzero = val_counts > 0
    for c in np.where(nonzero)[0]:
        idx = np.where(fp_test["label"] == c)[0]
        if idx.size == 0:
            continue
        diffs = test_cls[idx] - centroids[c]
        test_dist[idx] = np.linalg.norm(diffs, axis=1)
    test_dist = test_dist.astype(np.float32)

    ############################################################################
    # END class centroid distances
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN apply PTQ
    ############################################################################

    skip_modules = frozenset(cfg.ptq.skip_modules)
    quantized_names = apply_ptq_(
        model=image_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )
    print(f"\nPTQ: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, "
          f"skip_modules={list(cfg.ptq.skip_modules)}; quantized {len(quantized_names)} layers")

    ############################################################################
    # END apply PTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Q pass over val + test (predictions only)
    ############################################################################

    print("\nQ pass — val:")
    q_val = _forward_q(
        image_classifier, dataset.val_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="Q val",
    )

    print("\nQ pass — test:")
    q_test = _forward_q(
        image_classifier, dataset.test_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="Q test",
    )

    assert len(q_val["q_pred"]) == len(fp_val["label"]), "val FP/Q row count mismatch"
    assert len(q_test["q_pred"]) == len(fp_test["label"]), "test FP/Q row count mismatch"

    ############################################################################
    # END Q pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN assemble + save
    ############################################################################

    # Compute cross-model (FP <-> Q) scalar features from the full logits
    # we held in memory, then drop the heavy arrays.
    val_fp_logits = fp_val.pop("_fp_logits")
    test_fp_logits = fp_test.pop("_fp_logits")
    val_q_logits = q_val.pop("_q_logits")
    test_q_logits = q_test.pop("_q_logits")

    val_cross = _cross_model_features(
        val_fp_logits, val_q_logits, fp_val["fp_pred"], q_val["q_pred"],
    )
    test_cross = _cross_model_features(
        test_fp_logits, test_q_logits, fp_test["fp_pred"], q_test["q_pred"],
    )
    del val_fp_logits, test_fp_logits, val_q_logits, test_q_logits

    df_val = _to_dataframe(fp_val, q_val, val_cross, val_dist)
    df_test = _to_dataframe(fp_test, q_test, test_cross, test_dist)

    n_val = len(df_val)
    n_test = len(df_test)
    n_val_good = int(df_val["good"].sum())
    n_val_bad = int(df_val["bad"].sum())
    n_test_good = int(df_test["good"].sum())
    n_test_bad = int(df_test["bad"].sum())

    print(f"\nVal:   N={n_val}  FP-correct={int(df_val['fp_correct'].sum())}  "
          f"Q-correct={int(df_val['q_correct'].sum())}  good={n_val_good}  bad={n_val_bad}")
    print(f"Test:  N={n_test}  FP-correct={int(df_test['fp_correct'].sum())}  "
          f"Q-correct={int(df_test['q_correct'].sum())}  good={n_test_good}  bad={n_test_bad}")

    skip_modules_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"
    out_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "input_fragility_dumps",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_modules_tag}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        out_dir_parts.append(f"lnb={lnb}_lne={lne}")
    out_dir = os.path.join(*out_dir_parts)
    os.makedirs(out_dir, exist_ok=True)

    val_path = os.path.join(out_dir, "predictions_val.parquet")
    test_path = os.path.join(out_dir, "predictions_test.parquet")
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    meta = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr, "wd": cfg.wd, "ls": cfg.ls, "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "epochs": epochs,
        "device": str(device),
        "num_classes": num_classes,
        "embed_dim": int(centroids.shape[1]),
        "encoder_path": classifier_path,
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "quantized_layers": quantized_names,
        "n_val": n_val,
        "n_val_good": n_val_good,
        "n_val_bad": n_val_bad,
        "n_test": n_test,
        "n_test_good": n_test_good,
        "n_test_bad": n_test_bad,
        "val_path": val_path,
        "test_path": test_path,
    }
    meta_path = os.path.join(out_dir, "dump_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nVal predictions saved:  {val_path}")
    print(f"Test predictions saved: {test_path}")
    print(f"Metadata saved:         {meta_path}")

    ############################################################################
    # END assemble + save
    ############################################################################


if __name__ == "__main__":
    main()
