"""The full top-k gap profile, of which `q_margin` is the k=2 case.

`q_margin = z_(1) - z_(2)` asks only whether the runner-up could overtake the top-1
under a perturbation. The general object is the epsilon-contender set

    C_eps(x) = { j : z_(1)(x) - z_j(x) < 2 eps }

whose size says how many classes are in play, not merely whether any is. Prop. 1 is
the statement |C_eps| >= 2. This script dumps the whole gap profile so that
|C_eps| is recoverable offline at any eps:

    gap_k(x) = z_(1)(x) - z_(k)(x),   k = 1 .. K      (gap_1 = 0 by construction)

so |C_eps(x)| = #{ k : gap_k(x) < 2 eps }.

It exists to de-risk three questions before any theory is written:

  1. GRADED ROUTING. What is the distribution of |C_eps| among the inputs the
     recipe routes? If most routed inputs are two-way confusions, they may be
     resolvable far more cheaply than a full FP pass, and the FP budget drops.
  2. DOES RANKING BEAT THE MARGIN? Does the gap profile predict `bad` better than
     gap_2 alone?
  3. THE LUCKY-Q CEILING. Prop. 2 says no function of the PTQ logits separates
     `bad` from `lucky-Q` *under a symmetry assumption*. Higher-order ranking
     structure was never tested against it. We also store the rank of the gold
     label in both models, which is what the two groups differ by definition on.

Final layer only -- these questions are about the deployed decision, not about
depth. Two forward passes per batch, so this is cheap relative to the lens sweeps.
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
from src.gptq import apply_gptq_
from src.awq import apply_awq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

K_GAPS = 10


def _profile(logits, labels, k):
    """Top-k gap profile, top-k class ids, and the rank of the gold label."""
    k = min(k, logits.shape[1])
    vals, idx = logits.topk(k, dim=1)
    gaps = vals[:, :1] - vals                      # (B, k); gaps[:, 0] == 0
    order = logits.argsort(dim=1, descending=True)
    true_rank = (order == labels[:, None]).float().argmax(dim=1)
    return gaps, idx, true_rank, order[:, 0]


def _pass(model, loader, device, k, limit_num_batches=None, desc="pass"):
    model.eval()
    nb = len(loader)
    eff = min(limit_num_batches, nb) if limit_num_batches is not None else nb
    G, I, R, T, L, Z = [], [], [], [], [], []
    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=eff, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for bi, batch in bar:
            if bi >= eff:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device=device, non_blocking=True)
            labels = batch["labels"].to(dtype=torch.long, device=device)
            logits = model(images)
            g, i, r, t = _profile(logits, labels, k)
            G.append(g.cpu()); I.append(i.cpu()); R.append(r.cpu())
            T.append(t.cpu()); L.append(labels.cpu()); Z.append(logits.cpu())
    return (torch.cat(G).numpy(), torch.cat(I).numpy(),
            torch.cat(R).numpy(), torch.cat(T).numpy(), torch.cat(L).numpy(),
            torch.cat(Z).numpy())


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="gap_profile",
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

    cbp = os.environ["CHECKPOINT_BASE_PATH"]
    ckpt_dir = os.path.join(
        cbp, "vision", "ilharco_timm_supervised", "fp",
        sanitize_timm_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    )
    classifier_path = os.path.join(ckpt_dir, f"classifier_epoch_{epochs}.pt")
    print(f"\nLoading encoder from: {classifier_path}")
    num_classes = DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]
    model = ImageClassifier.load(
        model_name=cfg.model_name, num_classes=num_classes, filename=classifier_path,
    )
    model.to(device)
    k = min(K_GAPS, num_classes)
    print(f"  {num_classes} classes; storing gap profile to k={k}")

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=model.train_preprocess,
        preprocess_inference=model.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP pass, then PTQ pass
    ############################################################################

    print("\nFP pass — test:")
    fg, fi, fr, ft, labels, fz = _pass(model, dataset.test_loader, device, k,
                                       cfg.limit_num_batches, "FP test")

    skip_modules = frozenset(cfg.ptq.skip_modules)
    method = str(getattr(cfg.ptq, "method", "rtn")).lower()
    if method == "rtn":
        qn = apply_ptq_(model=model, bits=cfg.ptq.bits,
                        granularity=cfg.ptq.granularity, skip_modules=skip_modules)
    elif method in ("gptq", "awq"):
        # Error-minimizing alternative to RTN. Calibration images come from the TRAIN split so
        # nothing the retrieval/eval measurement touches is used to fit the quantizer.
        calib = []
        for bi, batch in enumerate(dataset.train_loader):
            if bi >= int(getattr(cfg.ptq, "calib_batches", 4)):
                break
            calib.append(batch[0] if isinstance(batch, (list, tuple)) else batch)
        if not calib:
            raise RuntimeError("gptq: no calibration batches available from the train split")
        _fn = apply_gptq_ if method == "gptq" else apply_awq_
        qn = _fn(
            model=model, bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
            skip_modules=skip_modules, calib_batches=calib,
            forward_fn=lambda m, x: m(x.to(device)),
            **({"percdamp": float(getattr(cfg.ptq, "percdamp", 0.01))} if method == "gptq"
               else {"n_grid": int(getattr(cfg.ptq, "awq_grid", 20))}),
        )
    else:
        raise ValueError(f"ptq.method expected 'rtn', 'gptq' or 'awq', got '{method}'")
    print(f"\nPTQ[{method}]: bits={cfg.ptq.bits}, gran={cfg.ptq.granularity}; quantised {len(qn)} layers")

    print("\nPTQ pass — test:")
    qg, qi, qr, qt, labels2, qz = _pass(model, dataset.test_loader, device, k,
                                        cfg.limit_num_batches, "Q test")
    assert (labels == labels2).all(), "FP/Q row order mismatch"

    ############################################################################
    # END passes
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN assemble + save
    ############################################################################

    df = pd.DataFrame({
        "sample_idx": np.arange(len(labels), dtype=np.int64),
        "label": labels.astype(np.int32),
        "fp_top1": ft.astype(np.int32),
        "q_top1": qt.astype(np.int32),
        "fp_true_rank": fr.astype(np.int32),
        "q_true_rank": qr.astype(np.int32),
        # eps: the induced logit perturbation. The semiorder theory is written in
        # terms of this and we had never measured it.
        "eps_linf": np.abs(qz - fz).max(axis=1).astype(np.float32),
        "eps_l2": np.linalg.norm(qz - fz, axis=1).astype(np.float32),
    })
    for j in range(k):
        df[f"fp_gap_{j+1}"] = fg[:, j].astype(np.float32)
        df[f"q_gap_{j+1}"] = qg[:, j].astype(np.float32)
        df[f"fp_cls_{j+1}"] = fi[:, j].astype(np.int32)
        df[f"q_cls_{j+1}"] = qi[:, j].astype(np.int32)

    df["fp_correct"] = df["fp_top1"] == df["label"]
    df["q_correct"] = df["q_top1"] == df["label"]
    df["good"] = df["fp_correct"] & df["q_correct"]
    df["bad"] = df["fp_correct"] & ~df["q_correct"]
    df["lucky_q"] = ~df["fp_correct"] & df["q_correct"]

    print(f"\n  test: N={len(df)}  good={int(df['good'].sum())}  "
          f"bad={int(df['bad'].sum())}  lucky-Q={int(df['lucky_q'].sum())}")

    skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) else "none"
    out_dir = os.path.join(
        cbp, "vision", "ilharco_timm_supervised", "gap_profile",
        sanitize_timm_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_tag}"
        + ("" if method == "rtn" else f"_m={method}"),
        f"seed={cfg.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)
    test_path = os.path.join(out_dir, "gap_profile_test.parquet")
    df.to_parquet(test_path, index=False)

    meta = {
        "model_name": cfg.model_name, "dataset_name": cfg.dataset_name,
        "num_classes": num_classes, "k_gaps": k, "n_test": len(df),
        "n_good": int(df["good"].sum()), "n_bad": int(df["bad"].sum()),
        "n_lucky_q": int(df["lucky_q"].sum()),
        "ptq_bits": cfg.ptq.bits, "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "seed": cfg.seed, "epochs": epochs,
        "encoder_path": classifier_path, "test_path": test_path,
    }
    with open(os.path.join(out_dir, "gap_profile_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nGap profile saved: {test_path}")

    ############################################################################
    # END assemble + save
    ############################################################################


if __name__ == "__main__":
    main()
