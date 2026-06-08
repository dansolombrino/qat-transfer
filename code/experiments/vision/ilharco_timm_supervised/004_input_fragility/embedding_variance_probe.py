"""Side-quest: do `bad` samples (FP-correct, PTQ-wrong) have higher patch-
embedding variance than `good` samples (both correct)?

Hypothesis: high-variance embeddings carry spiky information that weight-only
PTQ smooths out, so `bad` should sit on the upper tail of patch-embedding
variance.

This script does NOT run the transformer body. It runs only `model.patch_embed`
(a single conv) over the test split, computes two per-sample variance scalars,
and joins with the existing predictions_test.parquet from 004_input_fragility
(which already has `good`/`bad` labels).

Per-sample scalars:
    var_flat  = Var(patch_embed(x).flatten())          # global variance
    var_token = mean_d Var_t(patch_embed(x)[:,d])      # mean across-token var

Per-task summary written to JSON:
    AUROC of each scalar as a bad-vs-good discriminator on FP-correct test rows
    Mann-Whitney U p-value (one-sided, alt: var_bad > var_good)
    Group sizes and median values
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
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


def _patch_embed_variances(patch_embed, loader, device, limit_num_batches=None,
                           desc="patch_embed"):
    """Forward only through `patch_embed`. Returns (var_flat, var_token) as
    numpy arrays of length N, in the loader's deterministic row order."""

    patch_embed.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    flat_chunks, token_chunks = [], []
    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=effective, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for i, batch in bar:
            if i >= effective:
                break
            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device=device, non_blocking=True)

            emb = patch_embed(images)  # timm ViT: (B, N_patches, D)
            assert emb.dim() == 3, f"expected (B,N,D) from patch_embed, got {emb.shape}"

            # Per-sample variance of the flattened embedding tensor.
            var_flat = emb.flatten(1).var(dim=1, unbiased=False)
            # Per-dim variance across tokens, then mean over dims.
            var_token = emb.var(dim=1, unbiased=False).mean(dim=1)

            flat_chunks.append(var_flat.cpu())
            token_chunks.append(var_token.cpu())

    var_flat = torch.cat(flat_chunks).numpy().astype(np.float64)
    var_token = torch.cat(token_chunks).numpy().astype(np.float64)
    return var_flat, var_token


def _auroc_higher_means_bad(score: np.ndarray, is_bad: np.ndarray) -> float:
    """AUROC for `is_bad ~ score`. score: (N,), is_bad: (N,) bool."""
    pos = score[is_bad]
    neg = score[~is_bad]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    # rank-based AUROC via Mann-Whitney U
    u, _ = mannwhitneyu(pos, neg, alternative='greater')
    return float(u / (pos.size * neg.size))


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="embedding_variance_probe",
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
    # BEGIN load classifier
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
    assert hasattr(image_classifier.model, 'patch_embed'), (
        "Expected timm ViT with .model.patch_embed; got "
        f"{type(image_classifier.model).__name__}"
    )

    ############################################################################
    # END load classifier
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset (same call as dump_pred_and_input_props.py — preserves row order)
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
    # END dataset
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN patch_embed forward pass (test split only)
    ############################################################################

    patch_embed = image_classifier.model.patch_embed

    print("\npatch_embed pass — test:")
    var_flat, var_token = _patch_embed_variances(
        patch_embed, dataset.test_loader, device,
        limit_num_batches=cfg.limit_num_batches, desc="patch_embed test",
    )
    print(f"  computed {var_flat.size} sample variances")

    ############################################################################
    # END patch_embed forward pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN load fragility parquet and join
    ############################################################################

    skip_modules_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"
    dump_dir_parts = [
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
        dump_dir_parts.append(f"lnb={lnb}_lne={lne}")
    dump_dir = os.path.join(*dump_dir_parts)
    test_parquet = os.path.join(dump_dir, "predictions_test.parquet")
    assert os.path.exists(test_parquet), (
        f"Required fragility parquet not found: {test_parquet}\n"
        f"Run dump_pred_and_input_props.py first."
    )

    df = pd.read_parquet(test_parquet)
    assert len(df) == var_flat.size, (
        f"row count mismatch: parquet has {len(df)}, variances have {var_flat.size}. "
        f"Seed / loader determinism may have drifted."
    )
    df["embed_var_flat"] = var_flat.astype(np.float32)
    df["embed_var_token"] = var_token.astype(np.float32)

    ############################################################################
    # END load fragility parquet and join
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN per-task summary
    ############################################################################

    fp_correct = df["fp_correct"].to_numpy()
    bad = df["bad"].to_numpy()
    good = df["good"].to_numpy()

    keep = fp_correct  # bad/good only defined on FP-correct rows
    is_bad = bad[keep]
    vf = var_flat[keep]
    vt = var_token[keep]

    n_bad = int(is_bad.sum())
    n_good = int((~is_bad).sum())

    if n_bad == 0 or n_good == 0:
        print(f"\n[skip] {cfg.dataset_name}: n_bad={n_bad}, n_good={n_good}")
        auc_flat = auc_token = float('nan')
        p_flat = p_token = float('nan')
        med_bad_flat = med_good_flat = float('nan')
        med_bad_token = med_good_token = float('nan')
    else:
        auc_flat = _auroc_higher_means_bad(vf, is_bad)
        auc_token = _auroc_higher_means_bad(vt, is_bad)
        _, p_flat = mannwhitneyu(vf[is_bad], vf[~is_bad], alternative='greater')
        _, p_token = mannwhitneyu(vt[is_bad], vt[~is_bad], alternative='greater')
        p_flat = float(p_flat)
        p_token = float(p_token)
        med_bad_flat = float(np.median(vf[is_bad]))
        med_good_flat = float(np.median(vf[~is_bad]))
        med_bad_token = float(np.median(vt[is_bad]))
        med_good_token = float(np.median(vt[~is_bad]))

    summary = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "n_test": int(len(df)),
        "n_fp_correct": int(fp_correct.sum()),
        "n_bad": n_bad,
        "n_good": n_good,
        "auroc_var_flat": auc_flat,
        "auroc_var_token": auc_token,
        "mw_p_one_sided_var_flat": p_flat,
        "mw_p_one_sided_var_token": p_token,
        "median_var_flat_bad": med_bad_flat,
        "median_var_flat_good": med_good_flat,
        "median_var_token_bad": med_bad_token,
        "median_var_token_good": med_good_token,
    }

    print("\n=== embedding-variance probe ===")
    pprint(summary, expand_all=True)
    print("AUROC > 0.5 supports the hypothesis (var_bad > var_good).")

    ############################################################################
    # END per-task summary
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save
    ############################################################################

    out_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "embedding_variance_probes",
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

    out_parquet = os.path.join(out_dir, "embedding_variance_test.parquet")
    out_json = os.path.join(out_dir, "summary.json")
    df[["sample_idx", "label", "fp_correct", "q_correct", "good", "bad",
        "embed_var_flat", "embed_var_token"]].to_parquet(out_parquet, index=False)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved parquet:  {out_parquet}")
    print(f"Saved summary:  {out_json}")

    ############################################################################
    # END save
    ############################################################################


if __name__ == "__main__":
    main()
