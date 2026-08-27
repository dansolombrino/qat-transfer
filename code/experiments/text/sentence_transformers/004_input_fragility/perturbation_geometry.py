"""Why does quantizing the corpus HELP at W3? Decompose the embedding perturbation.

F37 established that the query-only reversal is not a magnitude effect: at W3 the realized gap
perturbations under query-only and both-quantized are statistically indistinguishable, yet the
flip rates differ at p<1e-10. The remaining explanation is the perturbation's *direction*.

For a query q and document d, the similarity is cos(q, d). Quantization moves each embedding by
some delta. Decompose the document-side move into the part parallel to the embedding and the part
orthogonal to it:

    delta_d = alpha * d_hat + delta_perp

The parallel part only rescales that document's norm. Under cosine similarity a pure rescaling is
invisible, and even under an inner product it scales every query's score for that document by the
same factor -- it cannot reorder documents *for a fixed query* unless alpha differs across
documents. The orthogonal part rotates the embedding and is what genuinely reorders.

So the hypothesis is: corpus-side perturbation is relatively more parallel (rank-preserving) than
query-side perturbation, at equal magnitude. This script measures that directly.

Outputs one parquet of per-embedding geometry plus a JSON summary.
"""
import json
import os
import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[5]   # repo root, derived not hardcoded
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
from dotenv import load_dotenv
load_dotenv(_R / ".env")

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint

from src.quantization import apply_ptq_
from src.gptq import apply_gptq_
from src.awq import apply_awq_


def _geometry(fp: np.ndarray, q: np.ndarray) -> dict:
    """Parallel/orthogonal split of the quantization move, per embedding row."""
    delta = q - fp
    norm = np.linalg.norm(fp, axis=1, keepdims=True)
    unit = fp / np.clip(norm, 1e-12, None)
    par = (delta * unit).sum(1)                       # signed component along the embedding
    perp = np.linalg.norm(delta - par[:, None] * unit, axis=1)
    dnorm = np.linalg.norm(delta, axis=1)
    return {
        "fp_norm": norm[:, 0],
        "delta_norm": dnorm,
        "par": par,
        "perp": perp,
        # the quantity that matters: what fraction of the move actually rotates the embedding
        "perp_frac": perp / np.clip(dnorm, 1e-12, None),
        "rel_delta": dnorm / np.clip(norm[:, 0], 1e-12, None),
    }


@hydra.main(
    config_path="../../../../../config/experiments/text/sentence_transformers/004_input_fragility",
    config_name="perturbation_geometry",
    version_base=None,
)
def main(cfg: DictConfig):
    rprint(OmegaConf.to_container(cfg, resolve=True))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(_R / "code" / "experiments" / "text" / "sentence_transformers"
                          / "004_input_fragility"))
    from retrieval_gap_profile import _load_corpus_and_queries, _encode  # reuse, do not duplicate

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.model_name, device=str(device))
    model.max_seq_length = cfg.max_length

    docs, queries = _load_corpus_and_queries(cfg.dataset_name, cfg.max_queries, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if len(docs) > cfg.max_docs:
        docs = [docs[i] for i in rng.choice(len(docs), size=cfg.max_docs, replace=False)]
    rprint(f"\n{cfg.dataset_name}: {len(docs)} documents, {len(queries)} queries")

    rprint("\nFP encoding:")
    d_fp = _encode(model, docs, cfg.batch_size, is_query=False)
    q_fp = _encode(model, queries, cfg.batch_size, is_query=True)

    inner = model[0].auto_model
    skip = frozenset(cfg.ptq.skip_modules)
    method = str(cfg.ptq.method).lower()
    if method == "rtn":
        n = apply_ptq_(model=inner, bits=cfg.ptq.bits,
                       granularity=cfg.ptq.granularity, skip_modules=skip)
    else:
        cal_idx = rng.choice(len(docs), size=min(cfg.ptq.calib_batches * cfg.batch_size, len(docs)),
                             replace=False)
        cal = [[docs[i] for i in cal_idx[j:j + cfg.batch_size]]
               for j in range(0, len(cal_idx), cfg.batch_size)]
        fn = apply_gptq_ if method == "gptq" else apply_awq_
        n = fn(model=inner, bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
               skip_modules=skip, calib_batches=cal,
               forward_fn=lambda _m, b: _encode(model, b, cfg.batch_size, is_query=False))
    rprint(f"\nPTQ[{method}]: bits={cfg.ptq.bits}, gran={cfg.ptq.granularity}; "
           f"quantised {len(n)} layers")

    rprint("\nPTQ encoding:")
    d_q = _encode(model, docs, cfg.batch_size, is_query=False)
    q_q = _encode(model, queries, cfg.batch_size, is_query=True)

    rows = []
    for side, fp, qq in (("document", d_fp, d_q), ("query", q_fp, q_q)):
        g = _geometry(np.asarray(fp, dtype=np.float64), np.asarray(qq, dtype=np.float64))
        rows.append(pd.DataFrame({"side": side, **g}))
    out = pd.concat(rows, ignore_index=True)

    summary = {}
    for side, sub in out.groupby("side"):
        summary[side] = {
            "n": int(len(sub)),
            "median_rel_delta": float(sub.rel_delta.median()),
            "median_perp_frac": float(sub.perp_frac.median()),
            "median_par_over_norm": float((sub.par / sub.fp_norm).median()),
        }
    dperp = summary["document"]["median_perp_frac"]
    qperp = summary["query"]["median_perp_frac"]
    summary["doc_minus_query_perp_frac"] = float(dperp - qperp)
    rprint("\n  fraction of the quantization move that ROTATES the embedding "
           "(lower = more rank-preserving):")
    rprint(f"    document side : {dperp:.4f}")
    rprint(f"    query side    : {qperp:.4f}")
    rprint(f"  -> {'documents rotate LESS (supports the direction hypothesis)' if dperp < qperp else 'documents rotate MORE or equally (refutes it)'}")

    tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) else "none"
    out_dir = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text" / "sentence_transformers" \
        / "perturbation_geometry" / cfg.model_name.replace("/", "_").replace(".", "_") \
        / cfg.dataset_name / f"ml={cfg.max_length}_bs={cfg.batch_size}" \
        / (f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={tag}"
           + ("" if method == "rtn" else f"_m={method}")) / f"seed={cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "perturbation_geometry.parquet", index=False)
    (out_dir / "perturbation_geometry_metadata.json").write_text(json.dumps(
        {"model_name": cfg.model_name, "dataset_name": cfg.dataset_name,
         "n_docs": len(docs), "n_queries": len(queries),
         "ptq": OmegaConf.to_container(cfg.ptq, resolve=True), "summary": summary}, indent=2))
    rprint(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
