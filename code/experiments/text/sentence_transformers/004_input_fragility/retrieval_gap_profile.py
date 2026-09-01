"""Ranking fragility on genuine retrieval, not classification heads.

Section 2.7 of the draft material says quantization preserves the argmax far better than
it preserves the ranking: the top-1 gap is ~10x larger relative to the perturbation than
every deeper gap, so the exact top-5 set survives for only 18-45% of inputs. Every number
there comes from a *classification head* over 6-400 classes. A referee can fairly answer
"those are classifier logits, not rankings".

This script runs the same measurement where the ranking is the product: embed a document
corpus and a query set with Qwen3-Embedding proper (last-token pooling + L2 normalize, no
classification head), rank by cosine similarity, and compare the top-k retrieved sets under
FP and PTQ. The theory transfers verbatim -- the similarity scores play the role of logits
and eps = ||s_PTQ - s_FP||_inf over the corpus.

No relevance judgements are needed. We measure FP-vs-PTQ *agreement*, not retrieval quality,
so qrels never enter and any corpus + query set works.

Two things come for free and the script exploits both:

  1. THE CORPUS-SIZE CURVE. Restricting the ranking to a corpus subset is a scoring-time
     operation, not an embedding-time one. So one pass over a corpus yields the result at
     every corpus size, and we can ask whether ranking fragility *worsens* as the corpus
     grows -- the prediction being that more documents packed into the same similarity
     range means smaller gaps, hence more flips.

  2. THE QUERY-ONLY VARIANT. Three ranking conditions come out of two encodings:
       FP            s = Q_fp @ D_fp.T
       both-quantized s = Q_q  @ D_q.T     (memory-motivated deployment)
       query-only     s = Q_q  @ D_fp.T    (realistic serving: corpus embedded offline in
                                            FP once, query encoder quantized for latency)
     FP is the reference for "did the ranking change" in both cases.

Writes one Parquet per (dataset, ptq-config), with one row per (query, corpus_size,
condition).
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

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.quantization import apply_ptq_
from src.gptq import apply_gptq_
from src.awq import apply_awq_
from src.vision.utils import sanitize_hf_model_name, set_seed


OmegaConf.register_new_resolver("sanitize_hf", sanitize_hf_model_name, replace=True)

IS_SLURM = "SLURM_JOB_ID" in os.environ

K_GAPS = 10

# name -> HF path; all expose "corpus" and "queries" configs
DATASET_TO_HF_PATH = {
    "NFCorpus":  "mteb/nfcorpus",
    "SciFact":   "mteb/scifact",
    "SCIDOCS":   "mteb/scidocs",
    "FiQA":      "mteb/fiqa",
    "TRECCOVID": "mteb/trec-covid",
}


def _load_corpus_and_queries(dataset_name, max_queries, seed, return_index=False):
    hf_path = DATASET_TO_HF_PATH[dataset_name]
    token = os.environ.get("HF_TOKEN")
    cache = os.environ.get("HF_DATASETS_CACHE")
    kw = dict(token=token, cache_dir=cache, trust_remote_code=True,
              verification_mode="no_checks")

    corpus = load_dataset(hf_path, "corpus", split="corpus", **kw)
    queries = load_dataset(hf_path, "queries", split="queries", **kw)

    docs = [((t or "") + " " + (x or "")).strip()
            for t, x in zip(corpus["title"], corpus["text"])]
    qs = [q or "" for q in queries["text"]]

    sel = np.arange(len(qs))
    if max_queries is not None and len(qs) > max_queries:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(qs), size=max_queries, replace=False)
        qs = [qs[i] for i in sel]

    if return_index:
        return docs, qs, sel
    return docs, qs


def _encode(model, texts, batch_size, is_query):
    """Qwen3-Embedding wants an instruction prefix on queries; the ST API applies it."""
    fn = None
    if is_query and hasattr(model, "encode_query"):
        fn = model.encode_query
    elif (not is_query) and hasattr(model, "encode_document"):
        fn = model.encode_document
    if fn is None:
        fn = model.encode
    emb = fn(texts, batch_size=batch_size, convert_to_numpy=True,
             normalize_embeddings=True, show_progress_bar=not IS_SLURM)
    return np.asarray(emb, dtype=np.float32)


def _profile(sims, k):
    """Top-k gap profile and top-k doc indices from a (B, N) similarity block."""
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(part, order, axis=1)
    gaps = vals[:, :1] - vals
    return gaps, idx


def _rows(sims_ref, sims_cmp, k, chunk):
    """Per-query gap profiles under both scorings, plus eps over the whole corpus."""
    out = []
    for a in range(0, sims_ref.shape[0], chunk):
        b = slice(a, a + chunk)
        gr, ir = _profile(sims_ref[b], k)
        gc, ic = _profile(sims_cmp[b], k)
        eps = np.abs(sims_cmp[b] - sims_ref[b]).max(axis=1)
        out.append((gr, ir, gc, ic, eps))
    return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
            np.concatenate([o[2] for o in out]), np.concatenate([o[3] for o in out]),
            np.concatenate([o[4] for o in out]))


@hydra.main(
    config_path="../../../../../config/experiments/text/sentence_transformers/004_input_fragility",
    config_name="retrieval_gap_profile",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    ############################################################################
    # BEGIN data
    ############################################################################

    docs, queries = _load_corpus_and_queries(
        cfg.dataset_name, cfg.max_queries, cfg.seed,
    )
    print(f"\n{cfg.dataset_name}: {len(docs)} documents, {len(queries)} queries")

    ############################################################################
    # END data
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP encoding, then PTQ encoding
    ############################################################################

    model = SentenceTransformer(cfg.model_name, device=str(device))
    model.max_seq_length = cfg.max_length
    model.eval()

    print("\nFP encoding:")
    d_fp = _encode(model, docs, cfg.batch_size, is_query=False)
    q_fp = _encode(model, queries, cfg.batch_size, is_query=True)

    inner = model[0].auto_model
    skip_modules = frozenset(cfg.ptq.skip_modules)
    method = str(getattr(cfg.ptq, "method", "rtn")).lower()
    if method == "rtn":
        quantized = apply_ptq_(model=inner, bits=cfg.ptq.bits,
                               granularity=cfg.ptq.granularity, skip_modules=skip_modules)
    elif method in ("gptq", "awq"):
        # No train split exists for a retrieval corpus, so GPTQ is calibrated on a random
        # sample of corpus documents. This is what a real deployment would do (calibration is
        # unlabeled), but it does mean the quantizer has seen corpus text — state it in the
        # writeup rather than implying a held-out calibration set.
        n_cal = int(getattr(cfg.ptq, "calib_batches", 4)) * cfg.batch_size
        rng_c = np.random.default_rng(cfg.seed + 1)
        idx_c = rng_c.choice(len(docs), size=min(n_cal, len(docs)), replace=False)
        cal_texts = [docs[i] for i in idx_c]
        calib = [cal_texts[i:i + cfg.batch_size]
                 for i in range(0, len(cal_texts), cfg.batch_size)]
        # Hooks sit on `inner`'s Linears; the forward must go through the outer
        # SentenceTransformer so tokenization and pooling happen as normal.
        _fn = apply_gptq_ if method == "gptq" else apply_awq_
        quantized = _fn(
            model=inner, bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
            skip_modules=skip_modules, calib_batches=calib,
            forward_fn=lambda _m, batch: _encode(model, batch, cfg.batch_size, is_query=False),
            **({"percdamp": float(getattr(cfg.ptq, "percdamp", 0.01))} if method == "gptq"
               else {"n_grid": int(getattr(cfg.ptq, "awq_grid", 20))}),
        )
    else:
        raise ValueError(f"ptq.method expected 'rtn', 'gptq' or 'awq', got '{method}'")
    print(f"\nPTQ[{method}]: bits={cfg.ptq.bits}, gran={cfg.ptq.granularity}; "
          f"quantised {len(quantized)} layers")

    print("\nPTQ encoding:")
    d_q = _encode(model, docs, cfg.batch_size, is_query=False)
    q_q = _encode(model, queries, cfg.batch_size, is_query=True)

    ############################################################################
    # END encoding
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN rank at every corpus size, under both quantization conditions
    ############################################################################

    n_docs = len(docs)
    sizes = sorted({s for s in cfg.corpus_sizes if s < n_docs} | {n_docs})
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_docs)          # fixed nested subsets across sizes
    k = min(int(getattr(cfg,'k_gaps',K_GAPS)), min(sizes))

    frames = []
    for size in sizes:
        sub = perm[:size]
        Dfp, Dq = d_fp[sub], d_q[sub]
        sims_fp = q_fp @ Dfp.T
        for cond, sims_cmp in (("both_quantized", q_q @ Dq.T),
                               ("query_only",     q_q @ Dfp.T)):
            gr, ir, gc, ic, eps = _rows(sims_fp, sims_cmp, k, cfg.sim_chunk)
            df = pd.DataFrame({
                "query_idx": np.arange(len(queries), dtype=np.int64),
                "corpus_size": np.int64(size),
                "condition": cond,
                "eps_linf": eps.astype(np.float32),
            })
            for j in range(k):
                df[f"fp_gap_{j+1}"] = gr[:, j].astype(np.float32)
                df[f"q_gap_{j+1}"] = gc[:, j].astype(np.float32)
                # doc ids are indices into the SUBSET; map back to global ids
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
    flip = (full.fp_top1.values != full.q_top1.values).mean()
    print(f"\n  full corpus, both-quantized: top-1 retrieval flip rate {flip*100:.2f}%")

    skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) else "none"
    out_dir = os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"], "text", "sentence_transformers",
        "retrieval_gap_profile", sanitize_hf_model_name(cfg.model_name), cfg.dataset_name,
        f"ml={cfg.max_length}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_tag}"
        + ("" if method == "rtn" else f"_m={method}"),
        f"seed={cfg.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)
    test_path = os.path.join(out_dir, "retrieval_gap_profile.parquet")
    out_df.to_parquet(test_path, index=False)

    meta = {
        "model_name": cfg.model_name, "dataset_name": cfg.dataset_name,
        "n_docs": n_docs, "n_queries": len(queries), "corpus_sizes": sizes,
        "k_gaps": k, "max_length": cfg.max_length,
        "ptq_bits": cfg.ptq.bits, "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "n_quantized_layers": len(quantized),
        "seed": cfg.seed, "test_path": test_path,
    }
    with open(os.path.join(out_dir, "retrieval_gap_profile_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nRetrieval gap profile saved: {test_path}")

    ############################################################################
    # END save
    ############################################################################


if __name__ == "__main__":
    main()
