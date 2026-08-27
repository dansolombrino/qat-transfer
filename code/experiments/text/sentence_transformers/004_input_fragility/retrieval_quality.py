"""Does quantization-induced ranking churn cost retrieval QUALITY, or is it neutral?

Everywhere else we measure FP-vs-PTQ *agreement*, which needs no labels and answers the
deployment question ("do my users see different results?").  It deliberately says nothing about
whether the new results are worse.  The BEIR corpora ship relevance judgements, so here we score
both rankings against ground truth and report nDCG@10 for each.

A flip is quality-neutral churn if nDCG is unchanged, and a real regression if it drops.  Either
outcome is informative: neutral churn is still a reproducibility problem for a deployed system,
and a drop is a stronger claim than the paper currently makes.
"""
import json, os, sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
from dotenv import load_dotenv
load_dotenv(_R / ".env")

import hydra, numpy as np, torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from src.quantization import apply_ptq_

sys.path.insert(0, str(_R / "code/experiments/text/sentence_transformers/004_input_fragility"))
from retrieval_gap_profile import DATASET_TO_HF_PATH, _encode


def _ndcg_at_k(order, rel, k=10):
    """order: (n_q, n_docs) doc indices ranked best-first. rel: dict q -> {doc: gain}."""
    out = []
    for qi, row in enumerate(order):
        gains = [rel.get(qi, {}).get(int(d), 0.0) for d in row[:k]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(rel.get(qi, {}).values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else np.nan)
    return float(np.nanmean(out)), int(np.sum(~np.isnan(out)))


@hydra.main(config_path="../../../../../config/experiments/text/sentence_transformers/004_input_fragility",
            config_name="retrieval_quality", version_base=None)
def main(cfg: DictConfig):
    rprint(OmegaConf.to_container(cfg, resolve=True))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    hf = DATASET_TO_HF_PATH[cfg.dataset_name]
    kw = dict(token=os.environ.get("HF_TOKEN"), cache_dir=os.environ.get("HF_DATASETS_CACHE"),
              trust_remote_code=True, verification_mode="no_checks")
    corpus = load_dataset(hf, "corpus", split="corpus", **kw)
    queries = load_dataset(hf, "queries", split="queries", **kw)
    qrels = load_dataset(hf, "default", split=cfg.qrels_split, **kw)

    docs = [((t or "") + " " + (x or "")).strip() for t, x in zip(corpus["title"], corpus["text"])]
    doc_pos = {str(i): n for n, i in enumerate(corpus["_id"])}
    qry_pos = {str(i): n for n, i in enumerate(queries["_id"])}

    # keep only queries that have at least one judged document
    rel = {}
    for r in qrels:
        qi, di, sc = qry_pos.get(str(r["query-id"])), doc_pos.get(str(r["corpus-id"])), float(r["score"])
        if qi is None or di is None or sc <= 0:
            continue
        rel.setdefault(qi, {})[di] = sc
    keep = sorted(rel)[: cfg.max_queries]
    rprint(f"\n{cfg.dataset_name}: {len(docs)} docs, {len(keep)} judged queries")
    qs = [queries["text"][i] for i in keep]
    rel = {n: rel[q] for n, q in enumerate(keep)}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.model_name, device=str(device))
    model.max_seq_length = cfg.max_length

    rprint("\nFP encoding:")
    d_fp, q_fp = _encode(model, docs, cfg.batch_size, False), _encode(model, qs, cfg.batch_size, True)
    inner = model[0].auto_model
    n = apply_ptq_(model=inner, bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
                   skip_modules=frozenset(cfg.ptq.skip_modules))
    rprint(f"\nPTQ: bits={cfg.ptq.bits}, gran={cfg.ptq.granularity}; quantised {len(n)} layers")
    rprint("\nPTQ encoding:")
    d_q, q_q = _encode(model, docs, cfg.batch_size, False), _encode(model, qs, cfg.batch_size, True)

    res = {}
    for tag, Q, D in (("fp", q_fp, d_fp), ("ptq", q_q, d_q)):
        sims = np.asarray(Q) @ np.asarray(D).T
        order = np.argsort(-sims, axis=1)[:, :10]
        nd, m = _ndcg_at_k(order, rel)
        res[tag] = dict(ndcg10=nd, n_scored=m)
        rprint(f"  nDCG@10 [{tag}] = {nd:.4f}  (n={m})")
    # agreement, for comparability with the rest of the paper
    o_fp = np.argsort(-(np.asarray(q_fp) @ np.asarray(d_fp).T), axis=1)[:, 0]
    o_q = np.argsort(-(np.asarray(q_q) @ np.asarray(d_q).T), axis=1)[:, 0]
    res["top1_flip"] = float((o_fp != o_q).mean())
    res["delta_ndcg"] = res["ptq"]["ndcg10"] - res["fp"]["ndcg10"]
    rprint(f"  top-1 flip = {res['top1_flip']:.1%}   delta nDCG@10 = {res['delta_ndcg']:+.4f}")

    out = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text" / "sentence_transformers" / \
        "retrieval_quality" / cfg.model_name.replace("/", "_").replace(".", "_") / cfg.dataset_name / \
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"seed={cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": cfg.model_name, "dataset_name": cfg.dataset_name,
               "ptq_config": OmegaConf.to_container(cfg.ptq, resolve=True)}
    payload.update(res)
    (out / "retrieval_quality.json").write_text(json.dumps(payload, indent=2))
    rprint(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
