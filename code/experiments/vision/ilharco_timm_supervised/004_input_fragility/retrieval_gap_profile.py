"""Ranking fragility on genuine image retrieval, on the checkpoints the paper already studies.

The text port (`code/experiments/text/sentence_transformers/.../retrieval_gap_profile.py`)
answers the "classifier logits are not rankings" objection to section 2.7 on Qwen3-Embedding.
It carries one caveat: it uses the off-the-shelf embedding model, not the fine-tuned
checkpoints the rest of the paper is built on.

This script has no such caveat. It runs image-to-image retrieval with the *same* fine-tuned
ViT checkpoints used everywhere else in the paper: take the pre-head pooled representation

    r = model.forward_head(model.norm(h), pre_logits=True)     # (B, D)

L2-normalise it, and rank the test set by cosine similarity. The classification head is not
used at all, so the ranking is over thousands of images rather than over 10-400 classes --
which is the variable that actually drives the effect (see the label-space analysis: the
separation ratio falls monotonically with candidate count).

Same free structure as the text port:
  * the corpus-size curve costs nothing -- restricting the corpus is a scoring-time operation;
  * three ranking conditions come from two encodings,
        FP             s = Q_fp @ D_fp.T
        both-quantized s = Q_q  @ D_q.T
        query-only     s = Q_q  @ D_fp.T   (corpus embedded offline in FP, encoder quantised)
    with FP as the reference for "did the ranking change".

Queries are drawn from the test set and the corpus is the test set, so every query is a member
of its own corpus; the self-match is masked out before ranking.
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


def _embed(model, loader, device, limit_num_batches=None, desc="embed"):
    """L2-normalised pre-head pooled representation for every test image."""
    model.eval()
    vit = model.model
    nb = len(loader)
    eff = min(limit_num_batches, nb) if limit_num_batches is not None else nb
    out, labels = [], []
    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=eff, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for bi, batch in bar:
            if bi >= eff:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device=device, non_blocking=True)
            feats = vit.forward_head(vit.norm(vit.forward_features(images)), pre_logits=True)
            feats = torch.nn.functional.normalize(feats.float(), dim=1)
            out.append(feats.cpu())
            labels.append(batch["labels"].to(dtype=torch.long).cpu())
    return torch.cat(out).numpy().astype(np.float32), torch.cat(labels).numpy()


def _profile(sims, k):
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(part, order, axis=1)
    return vals[:, :1] - vals, idx


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="retrieval_gap_profile",
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
    model = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        filename=classifier_path,
    )
    model.to(device)

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
    # BEGIN FP embedding, then PTQ embedding
    ############################################################################

    print("\nFP embedding — test:")
    d_fp, labels = _embed(model, dataset.test_loader, device, cfg.limit_num_batches, "FP")

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

    print("\nPTQ embedding — test:")
    d_q, labels2 = _embed(model, dataset.test_loader, device, cfg.limit_num_batches, "Q")
    assert (labels == labels2).all(), "FP/Q row order mismatch"

    n_docs = len(labels)
    print(f"\n  corpus: {n_docs} images, {DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]} classes")

    ############################################################################
    # END embedding
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN rank at every corpus size, under both quantization conditions
    ############################################################################

    rng = np.random.default_rng(cfg.seed)
    n_q = min(cfg.max_queries, n_docs)
    q_sel = rng.choice(n_docs, size=n_q, replace=False)
    perm = rng.permutation(n_docs)
    sizes = sorted({s for s in cfg.corpus_sizes if s < n_docs} | {n_docs})
    # depth of the dumped gap profile; must exceed the semiorder width or |C_eps| is
    # censored at k and the width-conditioned analysis is confounded (F30).
    k = min(int(getattr(cfg, 'k_gaps', K_GAPS)), min(sizes) - 1)

    frames = []
    for size in sizes:
        sub = perm[:size]
        pos = -np.ones(n_docs, dtype=np.int64)      # global id -> row in the subset
        pos[sub] = np.arange(size)
        Dfp, Dq = d_fp[sub], d_q[sub]
        for cond in ("both_quantized", "query_only"):
            Qq = d_q[q_sel]
            Dcmp = Dq if cond == "both_quantized" else Dfp
            gr_l, ir_l, gc_l, ic_l, ep_l = [], [], [], [], []
            for a in range(0, n_q, cfg.sim_chunk):
                b = slice(a, a + cfg.sim_chunk)
                ids = q_sel[b]
                s_fp = d_fp[ids] @ Dfp.T
                s_cm = Qq[b] @ Dcmp.T
                # a query that is itself in the corpus must not retrieve itself
                self_row = pos[ids]
                hit = self_row >= 0
                rows = np.nonzero(hit)[0]
                s_fp[rows, self_row[hit]] = -np.inf
                s_cm[rows, self_row[hit]] = -np.inf
                gr, ir = _profile(s_fp, k)
                gc, ic = _profile(s_cm, k)
                # the masked self-match is -inf in both, and -inf - -inf is NaN; drop those
                # positions explicitly rather than letting np.where hide the warning, so a
                # genuine NaN would still surface
                finite = np.isfinite(s_fp) & np.isfinite(s_cm)
                diff = np.zeros_like(s_fp)
                np.subtract(s_cm, s_fp, out=diff, where=finite)
                ep = np.where(finite, np.abs(diff), -np.inf).max(axis=1)
                assert np.isfinite(ep).all(), "eps is non-finite for some query"
                gr_l.append(gr); ir_l.append(ir); gc_l.append(gc); ic_l.append(ic); ep_l.append(ep)
            gr, ir = np.concatenate(gr_l), np.concatenate(ir_l)
            gc, ic = np.concatenate(gc_l), np.concatenate(ic_l)
            eps = np.concatenate(ep_l)
            df = pd.DataFrame({
                "query_idx": q_sel.astype(np.int64),
                "query_label": labels[q_sel].astype(np.int32),
                "corpus_size": np.int64(size),
                "condition": cond,
                "eps_linf": eps.astype(np.float32),
            })
            for j in range(k):
                df[f"fp_gap_{j+1}"] = gr[:, j].astype(np.float32)
                df[f"q_gap_{j+1}"] = gc[:, j].astype(np.float32)
                df[f"fp_cls_{j+1}"] = sub[ir[:, j]].astype(np.int64)
                df[f"q_cls_{j+1}"] = sub[ic[:, j]].astype(np.int64)
            df["fp_top1"] = df["fp_cls_1"]
            df["q_top1"] = df["q_cls_1"]
            frames.append(df)
        print(f"  corpus_size={size:>7d}  done")

    out_df = pd.concat(frames, ignore_index=True)

    ############################################################################
    # END ranking
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save
    ############################################################################

    full = out_df[(out_df.corpus_size == n_docs) & (out_df.condition == "both_quantized")]
    print(f"\n  full corpus, both-quantized: top-1 retrieval flip rate "
          f"{(full.fp_top1.values != full.q_top1.values).mean()*100:.2f}%")

    skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) else "none"
    out_dir = os.path.join(
        cbp, "vision", "ilharco_timm_supervised", "retrieval_gap_profile",
        sanitize_timm_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_tag}"
        + ("" if method == "rtn" else f"_m={method}"),
        f"seed={cfg.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)
    test_path = os.path.join(out_dir, "retrieval_gap_profile.parquet")
    out_df.to_parquet(test_path, index=False)

    meta = {
        "model_name": cfg.model_name, "dataset_name": cfg.dataset_name,
        "n_docs": n_docs, "n_queries": int(n_q), "corpus_sizes": sizes, "k_gaps": k,
        "num_classes": DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        "ptq_bits": cfg.ptq.bits, "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "seed": cfg.seed, "epochs": epochs,
        "encoder_path": classifier_path, "test_path": test_path,
    }
    with open(os.path.join(out_dir, "retrieval_gap_profile_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nRetrieval gap profile saved: {test_path}")

    ############################################################################
    # END save
    ############################################################################


if __name__ == "__main__":
    main()
