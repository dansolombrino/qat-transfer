"""Where in the network does PTQ fragility become visible?

The paper measures fragility only at the output: the final-layer top-1/top-2
margin (`q_margin`) predicts which inputs PTQ breaks (AUROC ~0.94), while the
patch-embedding carries no signal at all (`embedding_variance_probe.py`, AUROC
~0.43-0.50). This script fills the gap between those two endpoints.

For every transformer block i we read out intermediate logits with a logit lens
that reuses the model's own trained head:

    z_i = model.forward_head(model.norm(h_i))

where h_i is the output of block i. Because `forward_features` is
(patch_embed -> blocks -> norm) and `forward` is `forward_head(forward_features(x))`,
the readout at the LAST block reproduces the model's real logits exactly. That
identity is asserted at runtime as a correctness check.

Per sample and per layer we record, for both the FP and the PTQ model:

    fp_margin_l{i} / q_margin_l{i}      top-1 minus top-2 logit at layer i
    fp_top1_l{i}   / q_top1_l{i}        argmax class at layer i
    q_rank_of_fp_top1_l{i}              where PTQ ranks FP's layer-i choice
    top5_overlap_l{i}                   |top-5(FP_i) cap top-5(PTQ_i)|

Downstream these answer: at which layer do FP and PTQ rankings diverge, when
does each model commit to its final answer, and -- the payoff -- how early does
`margin_l{i}` already separate the inputs PTQ will break. If an early layer
suffices, the routing decision can be made before the forward pass finishes.

Caveat: the head is trained for the final layer, so intermediate logits are not
calibrated and margin MAGNITUDES are not comparable across layers. Rank-based
metrics (AUROC, top-k overlap, rank-of) are scale-free and unaffected.

Saves one Parquet per split, same path convention as
`dump_pred_and_input_props.py`.
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
from src.quantization import apply_ptq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

TOPK_OVERLAP = 5


def _layerwise_logits(model, images, n_blocks):
    """Run one forward pass, capturing every block output, and apply the logit
    lens to each. Returns a list of (B, C) logit tensors, one per block, plus
    the model's true final logits for the correctness assertion."""
    vit = model.model
    per_layer = []

    # Apply the lens inside the hook and keep only the (B, C) logits: storing all
    # (B, N, D) block outputs first costs ~1.2 GB on a 24-block ViT-L at bs=64.
    def hook(_module, _inp, out):
        per_layer.append(vit.forward_head(vit.norm(out)))

    handles = [vit.blocks[i].register_forward_hook(hook) for i in range(n_blocks)]
    try:
        true_logits = model(images)
    finally:
        for h in handles:
            h.remove()

    assert len(per_layer) == n_blocks, f"expected {n_blocks} block outputs, got {len(per_layer)}"
    return per_layer, true_logits


def _scalars_from_logits(z):
    """(B, C) logits -> (top1 idx, margin, full descending order)."""
    order = z.argsort(dim=1, descending=True)
    top2 = z.topk(2, dim=1).values
    return order[:, 0], (top2[:, 0] - top2[:, 1]), order


def _forward_layerwise(model, loader, device, n_blocks, prefix,
                       limit_num_batches=None, desc="pass"):
    """Per-layer scalars for one model (FP or PTQ). Returns dict of numpy arrays
    plus the per-layer top-k sets needed for the cross-model comparison."""
    model.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    top1 = {i: [] for i in range(n_blocks)}
    margin = {i: [] for i in range(n_blocks)}
    order_k = {i: [] for i in range(n_blocks)}   # top-K only, to bound memory
    labels_all = []

    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=effective, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for bi, batch in bar:
            if bi >= effective:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device=device, non_blocking=True)
            labels_all.append(batch["labels"].to(dtype=torch.long))

            per_layer, true_logits = _layerwise_logits(model, images, n_blocks)

            # correctness: the lens at the last block IS the model's own head path
            if bi == 0:
                lens_last = per_layer[-1]
                assert torch.allclose(lens_last, true_logits, atol=1e-4), (
                    "logit lens at the final block does not reproduce the model's "
                    f"logits (max abs diff {(lens_last - true_logits).abs().max().item():.3e}); "
                    "the readout path is wrong"
                )

            for i, z in enumerate(per_layer):
                t1, m, order = _scalars_from_logits(z)
                top1[i].append(t1.cpu())
                margin[i].append(m.cpu())
                order_k[i].append(order[:, :TOPK_OVERLAP].cpu())

    out = {"label": torch.cat(labels_all).numpy()}
    topk = {}
    for i in range(n_blocks):
        out[f"{prefix}_top1_l{i}"] = torch.cat(top1[i]).numpy().astype(np.int32)
        out[f"{prefix}_margin_l{i}"] = torch.cat(margin[i]).numpy().astype(np.float32)
        topk[i] = torch.cat(order_k[i]).numpy().astype(np.int32)
    return out, topk


def _cross_model_layer_stats(fp_topk, q_topk, fp_top1, n_blocks, n_classes):
    """Per-layer comparisons that need both models: top-k overlap and the rank
    PTQ assigns to FP's own layer-i choice."""
    out = {}
    for i in range(n_blocks):
        f, q = fp_topk[i], q_topk[i]
        n = f.shape[0]

        overlap = np.empty(n, dtype=np.int8)
        rank_of = np.empty(n, dtype=np.int32)
        for r in range(n):
            fs, qs = f[r], q[r]
            overlap[r] = len(np.intersect1d(fs, qs, assume_unique=True))
            hit = np.where(qs == fp_top1[i][r])[0]
            # rank within the stored top-K; TOPK_OVERLAP means "outside the top-K"
            rank_of[r] = hit[0] if hit.size else TOPK_OVERLAP
        out[f"top{TOPK_OVERLAP}_overlap_l{i}"] = overlap
        out[f"q_rank_of_fp_top1_l{i}"] = rank_of
    return out


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="layerwise_logit_lens",
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

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
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
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")

    print(f"\nLoading encoder from: {classifier_path}")
    num_classes = DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]
    image_classifier = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=num_classes,
        filename=classifier_path,
    )
    image_classifier.to(device)
    assert hasattr(image_classifier.model, "blocks"), (
        f"Expected timm ViT with .model.blocks; got {type(image_classifier.model).__name__}"
    )
    n_blocks = len(image_classifier.model.blocks)
    print(f"  {n_blocks} transformer blocks, {num_classes} classes")

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
        num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP layer-wise pass
    ############################################################################

    print("\nFP layer-wise pass — test:")
    fp_out, fp_topk = _forward_layerwise(
        image_classifier, dataset.test_loader, device, n_blocks, "fp",
        limit_num_batches=cfg.limit_num_batches, desc="FP test",
    )
    print(f"  collected {len(fp_out['label'])} samples x {n_blocks} layers")

    ############################################################################
    # END FP layer-wise pass
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
    # BEGIN PTQ layer-wise pass
    ############################################################################

    print("\nPTQ layer-wise pass — test:")
    q_out, q_topk = _forward_layerwise(
        image_classifier, dataset.test_loader, device, n_blocks, "q",
        limit_num_batches=cfg.limit_num_batches, desc="Q test",
    )
    assert len(q_out["label"]) == len(fp_out["label"]), "FP/Q row count mismatch"
    assert (q_out["label"] == fp_out["label"]).all(), "FP/Q row order mismatch"

    ############################################################################
    # END PTQ layer-wise pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN assemble + save
    ############################################################################

    print("\nComputing cross-model per-layer statistics ...")
    fp_top1_by_layer = {i: fp_out[f"fp_top1_l{i}"] for i in range(n_blocks)}
    cross = _cross_model_layer_stats(fp_topk, q_topk, fp_top1_by_layer, n_blocks, num_classes)

    df = pd.DataFrame({"sample_idx": np.arange(len(fp_out["label"]), dtype=np.int64)})
    df["label"] = fp_out["label"].astype(np.int32)
    for k, v in fp_out.items():
        if k != "label":
            df[k] = v
    for k, v in q_out.items():
        if k != "label":
            df[k] = v
    for k, v in cross.items():
        df[k] = v

    # final-layer taxonomy, so the analysis can group without re-deriving it
    last = n_blocks - 1
    df["fp_correct"] = df[f"fp_top1_l{last}"] == df["label"]
    df["q_correct"] = df[f"q_top1_l{last}"] == df["label"]
    df["good"] = df["fp_correct"] & df["q_correct"]
    df["bad"] = df["fp_correct"] & ~df["q_correct"]
    df["lucky_q"] = ~df["fp_correct"] & df["q_correct"]

    n_bad, n_good, n_lucky = int(df["bad"].sum()), int(df["good"].sum()), int(df["lucky_q"].sum())
    print(f"  test: N={len(df)}  good={n_good}  bad={n_bad}  lucky-Q={n_lucky}")

    skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"
    out_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "layerwise_logit_lens",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_tag}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        out_dir_parts.append(f"lnb={lnb}_lne={lne}")
    out_dir = os.path.join(*out_dir_parts)
    os.makedirs(out_dir, exist_ok=True)

    test_path = os.path.join(out_dir, "layerwise_test.parquet")
    df.to_parquet(test_path, index=False)

    meta = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "n_blocks": n_blocks,
        "num_classes": num_classes,
        "topk_overlap": TOPK_OVERLAP,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr, "wd": cfg.wd, "ls": cfg.ls, "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "epochs": epochs,
        "device": str(device),
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "n_test": len(df),
        "n_test_good": n_good,
        "n_test_bad": n_bad,
        "n_test_lucky_q": n_lucky,
        "encoder_path": classifier_path,
        "test_path": test_path,
    }
    meta_path = os.path.join(out_dir, "layerwise_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTest layer-wise stats saved: {test_path}")
    print(f"Metadata saved:              {meta_path}")

    ############################################################################
    # END assemble + save
    ############################################################################


if __name__ == "__main__":
    main()
