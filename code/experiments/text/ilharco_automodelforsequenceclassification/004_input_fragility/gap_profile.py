"""The full top-k gap profile on the text stack (NLP port of the vision script).

Same object as `code/experiments/vision/.../004_input_fragility/gap_profile.py`:

    gap_k(x) = z_(1)(x) - z_(k)(x),   k = 1 .. K      (gap_1 = 0 by construction)

so the epsilon-contender set  C_eps(x) = { j : z_(1) - z_j < 2 eps }  is recoverable
offline at any eps, since |C_eps(x)| = #{ k : gap_k(x) < 2 eps }.

It also stores the realised perturbation

    eps_linf(x) = || z_PTQ(x) - z_FP(x) ||_inf

which is the parameter the semiorder theory is written in, and which the existing
text dumps (`dump_pred_and_input_props.py`) do not carry -- they keep only scalar
top-1/top-2 summaries and cross-model features, not the full logit vector.

Why a third backbone. The certificate and the ranking results so far rest on two
vision backbones that share an architecture family. Qwen3 is a decoder-only model
wrapped by `AutoModelForSequenceClassification` with a learned `score` head, pooled
at the last non-pad token -- a genuinely different readout path. If the eps/margin
regime picture and the contender-only certificate replicate here, they are properties
of quantised classifiers, not of ViTs.

Differences from the vision port, all forced by the stack:
  - model is `AutoModelForSequenceClassification` loaded from split backbone+head
    checkpoints, matching `dump_pred_and_input_props.py`;
  - batches are (texts, labels) and are tokenised in the loop;
  - `skip_modules` is `[score]` rather than `[head]`;
  - checkpoint paths carry `_ml={max_length}` and use `sanitize_hf_model_name`.

Two forward passes over the test split. Final layer only -- these questions are
about the deployed decision, not about depth.
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

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import numpy as np
import pandas as pd
import torch

from src.quantization import apply_ptq_
from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.text.data.registry import get_dataset
from src.vision.utils import random_tqdm_color, sanitize_hf_model_name, set_seed


OmegaConf.register_new_resolver("sanitize_hf", sanitize_hf_model_name, replace=True)


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

K_GAPS = 10

# Matches `_HEAD_MODULE` in dump_pred_and_input_props.py.
_HEAD_MODULE = {
    "google-bert/bert-base-uncased": "classifier",
    "google-bert/bert-large-uncased": "classifier",
    "google/embeddinggemma-300m": "score",
    "Qwen/Qwen3-Embedding-0.6B": "score",
}


def _profile(logits, labels, k):
    """Top-k gap profile, top-k class ids, and the rank of the gold label."""
    k = min(k, logits.shape[1])
    vals, idx = logits.topk(k, dim=1)
    gaps = vals[:, :1] - vals                      # (B, k); gaps[:, 0] == 0
    order = logits.argsort(dim=1, descending=True)
    true_rank = (order == labels[:, None]).float().argmax(dim=1)
    return gaps, idx, true_rank, order[:, 0]


def _pass(model, tokenizer, loader, device, max_length, k,
          limit_num_batches=None, desc="pass"):
    """Same row order across calls: the test loader is deterministic and unshuffled."""
    model.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    G, I, R, T, L, Z = [], [], [], [], [], []
    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=effective, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for i, batch in bar:
            if i >= effective:
                break

            texts, labels = batch
            encoding = tokenizer(
                texts, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device, non_blocking=True)
            attention_mask = encoding["attention_mask"].to(device, non_blocking=True)
            labels = labels.to(dtype=torch.long, device=device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            g, idx, r, t = _profile(logits, labels, k)
            G.append(g.cpu()); I.append(idx.cpu()); R.append(r.cpu())
            T.append(t.cpu()); L.append(labels.cpu()); Z.append(logits.float().cpu())

    return (torch.cat(G).numpy(), torch.cat(I).numpy(), torch.cat(R).numpy(),
            torch.cat(T).numpy(), torch.cat(L).numpy(), torch.cat(Z).numpy())


def _load_split_checkpoint_(model, backbone_path, head_path, device):
    backbone_state = torch.load(backbone_path, map_location=device, weights_only=False)
    head_state = torch.load(head_path, map_location=device, weights_only=False)
    model.load_state_dict(backbone_state, strict=False)
    model.load_state_dict(head_state, strict=False)


@hydra.main(
    config_path="../../../../../config/experiments/text/ilharco_automodelforsequenceclassification/004_input_fragility",
    config_name="gap_profile",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    base_epochs = DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
    epochs = min(base_epochs, cfg.limit_num_epochs) if cfg.limit_num_epochs is not None else base_epochs

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    ############################################################################
    # BEGIN dataset
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
        seed=cfg.seed,
    )
    num_classes = len(dataset.class_names)

    ############################################################################
    # END dataset
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN model + checkpoint
    ############################################################################

    if cfg.model_name not in _HEAD_MODULE:
        raise ValueError(
            f"Unsupported model_name={cfg.model_name!r}. "
            f"Add it to _HEAD_MODULE in this script after checking pooling assumptions."
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_classes,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=torch.float32)

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
    ckpt_dir = os.path.join(
        checkpoint_base_path, "text", "ilharco_automodelforsequenceclassification", "fp",
        sanitize_hf_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        f"seed={cfg.seed}",
    )
    backbone_path = os.path.join(ckpt_dir, f"backbone_epoch_{epochs}.pt")
    head_path = os.path.join(ckpt_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading backbone from: {backbone_path}")
    print(f"Loading head from: {head_path}")
    _load_split_checkpoint_(
        model=model, backbone_path=backbone_path, head_path=head_path, device=device,
    )

    k = min(K_GAPS, num_classes)
    print(f"  {num_classes} classes; storing gap profile to k={k}")

    ############################################################################
    # END model + checkpoint
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP pass, then PTQ pass
    ############################################################################

    print("\nFP pass — test:")
    fg, fi, fr, ft, labels, fz = _pass(
        model, tokenizer, dataset.test_loader, device, cfg.max_length, k,
        cfg.limit_num_batches, "FP test",
    )

    skip_modules = frozenset(cfg.ptq.skip_modules)
    quantized_names = apply_ptq_(
        model=model, bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity, skip_modules=skip_modules,
    )
    print(f"\nPTQ: bits={cfg.ptq.bits}, gran={cfg.ptq.granularity}; "
          f"quantised {len(quantized_names)} layers")

    print("\nPTQ pass — test:")
    qg, qi, qr, qt, labels2, qz = _pass(
        model, tokenizer, dataset.test_loader, device, cfg.max_length, k,
        cfg.limit_num_batches, "Q test",
    )
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
        # eps: the induced logit perturbation, the parameter the theory is written in.
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
        checkpoint_base_path, "text", "ilharco_automodelforsequenceclassification",
        "gap_profile", sanitize_hf_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_tag}",
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
        "seed": cfg.seed, "epochs": epochs, "max_length": cfg.max_length,
        "backbone_path": backbone_path, "head_path": head_path,
        "test_path": test_path,
    }
    with open(os.path.join(out_dir, "gap_profile_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nGap profile saved: {test_path}")

    ############################################################################
    # END assemble + save
    ############################################################################


if __name__ == "__main__":
    main()
