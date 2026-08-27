"""How far does PTQ push the representation away from FP, layer by layer?

F9 showed that at W3-channel the FP and PTQ models disagree on the top-1 class from
the very first block and never re-converge (agreement ends at ~23%, against ~91% at
W4). That was inferred from argmax agreement. This script measures the divergence in
the representation itself, with linear CKA (Kornblith et al., 2019).

For every block i we take the pooled pre-head representation of both models on the
same inputs,

    r_i = model.forward_head(model.norm(h_i), pre_logits=True)     # (B, D)

and accumulate the cross-products needed for

    CKA(X, Y) = ||Yc^T Xc||_F^2 / ( ||Xc^T Xc||_F * ||Yc^T Yc||_F )

where Xc, Yc are the column-centred feature matrices. Accumulating XtX, YtY, YtX and
the column sums lets us compute CKA exactly over the whole test set without ever
holding the (N, D) matrices: memory is 3 * D^2 per layer, not N * D.

Both models are held on the GPU simultaneously (FP and an in-place-quantised copy) so
that the two representations come from the same batch. That costs 2x model weights --
~0.7 GB for ViT-B, ~2.4 GB for ViT-L in fp32 -- which fits alongside activations on an
11 GB card.

CKA = 1 means the two representations are identical up to rotation and isotropic
scaling; lower means PTQ has genuinely moved the representation. The prediction from
F9 is a high, slowly-decaying curve at W4 and a curve that collapses early at W3.

Writes one JSON per (model, task, ptq-config).
"""

import copy
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
import torch


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


class _CKAAccumulator:
    """Streams the cross-products needed for exact linear CKA over a full split."""

    def __init__(self, n_blocks, dim):
        self.n = 0
        self.sx = [torch.zeros(dim, dtype=torch.float64) for _ in range(n_blocks)]
        self.sy = [torch.zeros(dim, dtype=torch.float64) for _ in range(n_blocks)]
        self.xtx = [torch.zeros(dim, dim, dtype=torch.float64) for _ in range(n_blocks)]
        self.yty = [torch.zeros(dim, dim, dtype=torch.float64) for _ in range(n_blocks)]
        self.ytx = [torch.zeros(dim, dim, dtype=torch.float64) for _ in range(n_blocks)]

    def update(self, xs, ys):
        """xs, ys: lists of (B, D) tensors, one per block, already on CPU float64."""
        self.n += xs[0].shape[0]
        for i, (x, y) in enumerate(zip(xs, ys)):
            self.sx[i] += x.sum(0)
            self.sy[i] += y.sum(0)
            self.xtx[i] += x.T @ x
            self.yty[i] += y.T @ y
            self.ytx[i] += y.T @ x

    def cka(self):
        out = []
        for i in range(len(self.sx)):
            n = float(self.n)
            xc = self.xtx[i] - torch.outer(self.sx[i], self.sx[i]) / n
            yc = self.yty[i] - torch.outer(self.sy[i], self.sy[i]) / n
            yx = self.ytx[i] - torch.outer(self.sy[i], self.sx[i]) / n
            num = (yx ** 2).sum()
            den = torch.linalg.matrix_norm(xc) * torch.linalg.matrix_norm(yc)
            out.append(float(num / den) if den > 0 else float("nan"))
        return out


def _pooled_reps(model, images, n_blocks):
    """Pre-head pooled representation at every block, via the model's own readout."""
    vit = model.model
    reps = []

    def hook(_m, _i, out):
        reps.append(vit.forward_head(vit.norm(out), pre_logits=True))

    handles = [vit.blocks[i].register_forward_hook(hook) for i in range(n_blocks)]
    try:
        model(images)
    finally:
        for h in handles:
            h.remove()
    assert len(reps) == n_blocks, f"expected {n_blocks} reps, got {len(reps)}"
    return reps


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="layerwise_cka",
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
    # BEGIN checkpoint loading (FP model + a quantised copy, both resident)
    ############################################################################

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
    checkpoint_dir = os.path.join(
        checkpoint_base_path, "vision", "ilharco_timm_supervised", "fp",
        sanitize_timm_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    )
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")
    print(f"\nLoading encoder from: {classifier_path}")

    num_classes = DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]
    fp_model = ImageClassifier.load(
        model_name=cfg.model_name, num_classes=num_classes, filename=classifier_path,
    )
    assert hasattr(fp_model.model, "blocks"), "expected a timm ViT with .model.blocks"
    n_blocks = len(fp_model.model.blocks)
    dim = fp_model.model.head.in_features

    # quantised copy: deepcopy on CPU first to avoid a transient 2x VRAM peak
    q_model = copy.deepcopy(fp_model).to("cpu")
    skip_modules = frozenset(cfg.ptq.skip_modules)
    quantized_names = apply_ptq_(
        model=q_model, bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity, skip_modules=skip_modules,
    )
    fp_model.to(device).eval()
    q_model.to(device).eval()
    print(f"  {n_blocks} blocks, dim {dim}, {num_classes} classes; "
          f"quantised {len(quantized_names)} layers "
          f"(bits={cfg.ptq.bits}, gran={cfg.ptq.granularity})")

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=fp_model.train_preprocess,
        preprocess_inference=fp_model.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN paired pass (both models on the same batch)
    ############################################################################

    acc = _CKAAccumulator(n_blocks, dim)
    agree = np.zeros(n_blocks, dtype=np.int64)
    n_seen = 0

    num_batches = len(dataset.test_loader)
    effective = min(cfg.limit_num_batches, num_batches) if cfg.limit_num_batches is not None else num_batches

    print("\nPaired FP/PTQ pass — test:")
    with torch.no_grad():
        bar = tqdm(enumerate(dataset.test_loader), total=effective, desc="CKA test",
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for bi, batch in bar:
            if bi >= effective:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device=device, non_blocking=True)

            xs = _pooled_reps(fp_model, images, n_blocks)
            ys = _pooled_reps(q_model, images, n_blocks)

            # per-layer top-1 agreement, as a cross-check against the lens sweep
            for i in range(n_blocks):
                zx = fp_model.model.head(xs[i])
                zy = q_model.model.head(ys[i])
                agree[i] += int((zx.argmax(1) == zy.argmax(1)).sum().item())
            n_seen += images.shape[0]

            acc.update([x.double().cpu() for x in xs], [y.double().cpu() for y in ys])

    cka = acc.cka()

    ############################################################################
    # END paired pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save
    ############################################################################

    print(f"\n  CKA by block: " + " ".join(f"{c:.3f}" for c in cka))
    print(f"  top-1 agree : " + " ".join(f"{a/n_seen*100:.0f}%" for a in agree))

    skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"
    out_dir = os.path.join(
        checkpoint_base_path, "vision", "ilharco_timm_supervised", "layerwise_cka",
        sanitize_timm_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_tag}",
        f"seed={cfg.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "n_blocks": n_blocks,
        "dim": dim,
        "num_classes": num_classes,
        "n_test": n_seen,
        "cka_per_block": cka,
        "top1_agree_per_block": (agree / max(n_seen, 1)).tolist(),
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "seed": cfg.seed,
        "epochs": epochs,
        "encoder_path": classifier_path,
    }
    out_path = os.path.join(out_dir, "cka_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCKA results saved: {out_path}")

    ############################################################################
    # END save
    ############################################################################


if __name__ == "__main__":
    main()
