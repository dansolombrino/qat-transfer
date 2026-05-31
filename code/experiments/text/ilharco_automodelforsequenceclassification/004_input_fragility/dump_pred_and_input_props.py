"""Dump per-sample FP / PTQ predictions + input properties for fragility analysis (NLP port).

For each val and test sample, this script records the same per-sample columns
as the vision dumper at `code/experiments/vision/.../004_input_fragility/`, but:

  - The "image-pixel properties" `img_*` are replaced with text-only properties
    `txt_n_tokens`, `txt_n_unique_tokens`, `txt_type_token_ratio`, `txt_punct_ratio`.
  - The pre-head pooled representation is taken at the last non-pad token
    position, mirroring how HuggingFace `AutoModelForSequenceClassification`
    pools for decoder-only backbones (e.g. Qwen3, Llama).
  - Model wrapper is HuggingFace `AutoModelForSequenceClassification` loaded
    from split backbone+head checkpoints (matching `evaluate_fp_ptq.py`).

Schema is otherwise identical to vision, so the downstream analyzers see the
same column names for everything except the input-domain features.

Saves one Parquet per split.
"""

import json
import logging
import os
import string
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
import torch.nn.functional as F

from src.quantization import apply_ptq_
from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.text.data.registry import get_dataset
from src.vision.utils import random_tqdm_color, sanitize_hf_model_name, set_seed


OmegaConf.register_new_resolver("sanitize_hf", sanitize_hf_model_name, replace=True)


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
_PUNCT_SET = frozenset(string.punctuation)


def _text_properties(texts: list[str], input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """Per-sample text statistics:
      - `txt_n_tokens`: count of real (non-pad) tokens, from attention_mask.
      - `txt_n_unique_tokens`: number of distinct token ids in the real positions.
      - `txt_type_token_ratio`: `n_unique / n_tokens` (defined for n_tokens>=1).
      - `txt_punct_ratio`: fraction of raw-string characters in
        `string.punctuation` (ASCII punct); 0 for empty strings.

    Returned as CPU float32 tensors of length B."""

    B = input_ids.size(0)
    assert len(texts) == B, f"texts len {len(texts)} != batch {B}"

    n_tokens = attention_mask.sum(dim=1).to(dtype=torch.long).cpu()

    # n_unique computed per-row in Python — B<=hundreds, so the cost is negligible.
    n_unique = torch.empty(B, dtype=torch.long)
    am_cpu = attention_mask.to("cpu", dtype=torch.bool)
    ids_cpu = input_ids.to("cpu")
    for i in range(B):
        real_ids = ids_cpu[i][am_cpu[i]]
        n_unique[i] = int(torch.unique(real_ids).numel())

    n_tokens_safe = n_tokens.clamp_min(1)
    type_token_ratio = (n_unique.to(torch.float64) / n_tokens_safe.to(torch.float64))

    punct = torch.empty(B, dtype=torch.float64)
    for i, t in enumerate(texts):
        if len(t) == 0:
            punct[i] = 0.0
        else:
            c = sum(1 for ch in t if ch in _PUNCT_SET)
            punct[i] = c / len(t)

    return {
        "txt_n_tokens": n_tokens.to(torch.float32),
        "txt_n_unique_tokens": n_unique.to(torch.float32),
        "txt_type_token_ratio": type_token_ratio.to(torch.float32),
        "txt_punct_ratio": punct.to(torch.float32),
    }


def _pooled_indices(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Index of the last non-pad token per row, mirroring HF's pooling logic
    for decoder-only `AutoModelForSequenceClassification` (e.g. Qwen3, Llama).

    Implementation lifted verbatim from `LlamaForSequenceClassification.forward`:
        sequence_lengths = (input_ids == pad_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
    The modulo handles the "no padding at all" case (argmax returns 0 → -1 →
    wraps to T-1 = last index).
    """

    eq_pad = (input_ids == pad_token_id).int()
    seq_lens = eq_pad.argmax(dim=-1) - 1
    seq_lens = seq_lens % input_ids.size(-1)
    return seq_lens


def _forward_fp(model, tokenizer, loader, device, max_length, head_module,
                limit_num_batches=None, desc="FP"):
    """Returns a dict of numpy arrays of length N (number of samples), keyed by
    column name. Includes labels, FP scalars, text properties, and pre-head
    pooled embeddings (kept in memory until centroids are computed, then dropped).
    """

    model.eval()
    pad_id = model.config.pad_token_id
    head_layer = getattr(model, head_module)

    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    chunks: dict[str, list[torch.Tensor]] = {
        "label": [], "fp_pred": [],
        "fp_logit_label": [], "fp_logit_top1": [], "fp_logit_top2": [],
        "fp_margin": [], "fp_softmax_top1": [], "fp_entropy": [],
        "txt_n_tokens": [], "txt_n_unique_tokens": [],
        "txt_type_token_ratio": [], "txt_punct_ratio": [],
        "_cls_embedding": [],
        "_fp_logits": [],
    }

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
            labels = labels.to(dtype=torch.long)

            # Text properties (no model needed).
            props = _text_properties(list(texts), input_ids, attention_mask)
            for k, v in props.items():
                chunks[k].append(v)

            # FP forward with hidden states for pre-head pooling.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits  # (B, C) — HF already pools at last non-pad token
            hidden = outputs.hidden_states[-1]  # (B, T, D)

            B = input_ids.size(0)
            seq_lens = _pooled_indices(input_ids, pad_id)
            arange_b = torch.arange(B, device=hidden.device)
            pooled = hidden[arange_b, seq_lens]  # (B, D)

            # Self-consistency: pooled fed through head must equal HF-pooled logits.
            head_logits = head_layer(pooled)
            assert torch.allclose(head_logits.float(), logits.float(), atol=1e-3, rtol=1e-3), (
                "pooled hidden state does not reproduce HF's pooled logits — "
                "pooling assumption is wrong for this model"
            )

            preds = logits.argmax(dim=1)
            top2_vals, _ = logits.topk(2, dim=1)
            fp_top1 = top2_vals[:, 0]
            fp_top2 = top2_vals[:, 1]

            logp = F.log_softmax(logits, dim=1)
            p = logp.exp()
            entropy = -(p * logp).sum(dim=1)
            sm_top1 = p.gather(1, preds.unsqueeze(1)).squeeze(1)
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


def _forward_q(model, tokenizer, loader, device, max_length,
               limit_num_batches=None, desc="Q"):
    """After PTQ has been applied in place. Same row order as the FP pass
    (deterministic loader, no shuffle)."""

    model.eval()
    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    chunks: dict[str, list[torch.Tensor]] = {
        "q_pred": [], "q_logit_top1": [], "q_logit_top2": [], "q_margin": [],
        "q_softmax_top1": [], "q_entropy": [],
        "_q_logits": [],
    }

    with torch.no_grad():
        bar = tqdm(enumerate(loader), total=effective, desc=desc,
                   colour=random_tqdm_color(), leave=False, **TQDM_KW)
        for i, batch in bar:
            if i >= effective:
                break

            texts, _ = batch
            encoding = tokenizer(
                texts, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device, non_blocking=True)
            attention_mask = encoding["attention_mask"].to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
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
    """Identical to the vision implementation — cross-model scalar features."""
    fp_logits = fp_logits.astype(np.float64)
    q_logits = q_logits.astype(np.float64)
    n = fp_logits.shape[0]

    idx = np.arange(n)
    fp_logit_at_q_pred = fp_logits[idx, q_pred]
    q_logit_at_fp_pred = q_logits[idx, fp_pred]

    fp_logits_shifted = fp_logits - fp_logits.max(axis=1, keepdims=True)
    fp_exp = np.exp(fp_logits_shifted)
    fp_softmax = fp_exp / fp_exp.sum(axis=1, keepdims=True)
    fp_softmax_at_q_pred = fp_softmax[idx, q_pred]

    q_logits_shifted = q_logits - q_logits.max(axis=1, keepdims=True)
    q_exp = np.exp(q_logits_shifted)
    q_softmax = q_exp / q_exp.sum(axis=1, keepdims=True)
    q_softmax_at_fp_pred = q_softmax[idx, fp_pred]

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
    """Identical semantics to the vision dumper."""
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
        "fp_pred": fp["fp_pred"].astype(np.int32),
        "fp_logit_label": fp["fp_logit_label"].astype(np.float32),
        "fp_logit_top1": fp["fp_logit_top1"].astype(np.float32),
        "fp_logit_top2": fp["fp_logit_top2"].astype(np.float32),
        "fp_margin": fp["fp_margin"].astype(np.float32),
        "fp_softmax_top1": fp["fp_softmax_top1"].astype(np.float32),
        "fp_entropy": fp["fp_entropy"].astype(np.float32),
        "fp_cls_dist_to_class_centroid": dist_to_centroid.astype(np.float32),
        "q_pred": q["q_pred"].astype(np.int32),
        "q_logit_top1": q["q_logit_top1"].astype(np.float32),
        "q_logit_top2": q["q_logit_top2"].astype(np.float32),
        "q_margin": q["q_margin"].astype(np.float32),
        "q_softmax_top1": q["q_softmax_top1"].astype(np.float32),
        "q_entropy": q["q_entropy"].astype(np.float32),
        "fp_logit_at_q_pred": cross["fp_logit_at_q_pred"],
        "q_logit_at_fp_pred": cross["q_logit_at_fp_pred"],
        "fp_softmax_at_q_pred": cross["fp_softmax_at_q_pred"],
        "q_softmax_at_fp_pred": cross["q_softmax_at_fp_pred"],
        "fp_q_kl_symmetric": cross["fp_q_kl_symmetric"],
        "fp_q_disagree": cross["fp_q_disagree"],
        "txt_n_tokens": fp["txt_n_tokens"].astype(np.float32),
        "txt_n_unique_tokens": fp["txt_n_unique_tokens"].astype(np.float32),
        "txt_type_token_ratio": fp["txt_type_token_ratio"].astype(np.float32),
        "txt_punct_ratio": fp["txt_punct_ratio"].astype(np.float32),
    })
    df["fp_correct"] = (df["fp_pred"] == df["label"])
    df["q_correct"] = (df["q_pred"] == df["label"])
    df["good"] = df["fp_correct"] & df["q_correct"]
    df["bad"] = df["fp_correct"] & ~df["q_correct"]
    return df


# Mapping from HF model name to the attribute name of the classification head.
# Matches `MODEL_NAME_TO_HEAD_MODULE` in finetune_fp.py.
_HEAD_MODULE = {
    "google-bert/bert-base-uncased": "classifier",
    "google-bert/bert-large-uncased": "classifier",
    "google/embeddinggemma-300m": "score",
    "Qwen/Qwen3-Embedding-0.6B": "score",
}


def _load_split_checkpoint_(model: torch.nn.Module, backbone_path: str, head_path: str, device: torch.device):
    backbone_state = torch.load(backbone_path, map_location=device, weights_only=False)
    head_state = torch.load(head_path, map_location=device, weights_only=False)
    model.load_state_dict(backbone_state, strict=False)
    model.load_state_dict(head_state, strict=False)


@hydra.main(
    config_path="../../../../../config/experiments/text/ilharco_automodelforsequenceclassification/004_input_fragility",
    config_name="dump_pred_and_input_props",
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
    head_module_name = _HEAD_MODULE[cfg.model_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_classes,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=torch.float32)

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
    is_dryrun = cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    ckpt_dir_parts = [
        checkpoint_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "fp_dryrun" if is_dryrun else "fp",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        ckpt_dir_parts.append(f"lnb={lnb}_lne={lne}")
    ckpt_dir = os.path.join(*ckpt_dir_parts)

    backbone_path = os.path.join(ckpt_dir, f"backbone_epoch_{epochs}.pt")
    head_path = os.path.join(ckpt_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading backbone from: {backbone_path}")
    print(f"Loading head from: {head_path}")
    _load_split_checkpoint_(
        model=model, backbone_path=backbone_path, head_path=head_path, device=device,
    )
    assert hasattr(model, head_module_name), (
        f"Loaded model has no `.{head_module_name}` attribute; pooling check would fail."
    )

    ############################################################################
    # END model + checkpoint
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP pass over val + test
    ############################################################################

    print("\nFP pass — val:")
    fp_val = _forward_fp(
        model, tokenizer, dataset.val_loader, device, cfg.max_length,
        head_module_name,
        limit_num_batches=cfg.limit_num_batches, desc="FP val",
    )
    print(f"  collected {len(fp_val['label'])} samples")

    print("\nFP pass — test:")
    fp_test = _forward_fp(
        model, tokenizer, dataset.test_loader, device, cfg.max_length,
        head_module_name,
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

    print("\nComputing class centroids on val, distances on val + test ...")
    val_cls = fp_val.pop("_cls_embedding")
    test_cls = fp_test.pop("_cls_embedding")

    val_dist, centroids, val_counts = _compute_centroid_distances(
        val_cls, fp_val["label"], num_classes,
    )

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
        model=model,
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
    # BEGIN Q pass
    ############################################################################

    print("\nQ pass — val:")
    q_val = _forward_q(
        model, tokenizer, dataset.val_loader, device, cfg.max_length,
        limit_num_batches=cfg.limit_num_batches, desc="Q val",
    )

    print("\nQ pass — test:")
    q_test = _forward_q(
        model, tokenizer, dataset.test_loader, device, cfg.max_length,
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
        "text",
        "ilharco_automodelforsequenceclassification",
        "input_fragility_dumps",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
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
        "lr": cfg.lr, "wd": cfg.wd, "ls": cfg.ls,
        "max_grad_norm": cfg.max_grad_norm,
        "max_length": cfg.max_length,
        "seed": cfg.seed,
        "epochs": epochs,
        "device": str(device),
        "num_classes": num_classes,
        "embed_dim": int(centroids.shape[1]),
        "backbone_path": backbone_path,
        "head_path": head_path,
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
