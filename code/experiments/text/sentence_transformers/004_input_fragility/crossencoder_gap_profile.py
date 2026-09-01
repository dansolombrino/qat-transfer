"""Ranking fragility of a quantized CROSS-ENCODER reranker.

The bi-encoder results cover first-stage retrieval; deployed stacks rerank a shortlist with a
cross-encoder, and reranking is where the paper's motivation points but had no measurement.
Here the FP bi-encoder retrieves a top-K shortlist per query (fixed across conditions), and the
cross-encoder scores each (query, doc) pair; we quantize ONLY the cross-encoder and measure
what happens to the reranked ordering. Same schema as the other gap profiles: per-query gaps,
eps, flip of the reranked top-1, and the separation ratio.
"""
import copy, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from dotenv import load_dotenv
load_dotenv()

import hydra, numpy as np, torch
from omegaconf import DictConfig
from rich import print as rprint
from src.quantization import apply_ptq_

from retrieval_gap_profile import DATASET_TO_HF_PATH, _load_corpus_and_queries


@hydra.main(config_path="../../../../../config/experiments/text/sentence_transformers/004_input_fragility",
            config_name="crossencoder_gap_profile", version_base=None)
def main(cfg: DictConfig):
    rprint(dict(cfg))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    from sentence_transformers import SentenceTransformer, CrossEncoder
    docs, queries = _load_corpus_and_queries(cfg.dataset_name, cfg.max_queries, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if len(docs) > cfg.max_docs:
        docs = [docs[i] for i in rng.choice(len(docs), cfg.max_docs, replace=False)]
    rprint(f"{cfg.dataset_name}: {len(docs)} docs, {len(queries)} queries")

    # FP first-stage shortlist, FIXED across conditions (the bi-encoder is never quantized here)
    be = SentenceTransformer(cfg.biencoder_name, device=str(device))
    D = np.asarray(be.encode(docs, batch_size=cfg.batch_size, show_progress_bar=False,
                             normalize_embeddings=True))
    Q = np.asarray(be.encode(queries, batch_size=cfg.batch_size, show_progress_bar=False,
                             normalize_embeddings=True))
    shortlist = np.argsort(-(Q @ D.T), axis=1)[:, :cfg.top_k]
    del be
    torch.cuda.empty_cache()

    ce = CrossEncoder(cfg.model_name, device=str(device), max_length=cfg.max_length)
    inner = ce.model
    fp_state = copy.deepcopy(inner.state_dict())

    def score_all():
        out = np.zeros(shortlist.shape, dtype=np.float32)
        pairs, idx = [], []
        for qi, row in enumerate(shortlist):
            for r, di in enumerate(row):
                pairs.append((queries[qi], docs[di]))
                idx.append((qi, r))
        scores = ce.predict(pairs, batch_size=cfg.batch_size, show_progress_bar=False)
        for (qi, r), sc in zip(idx, scores):
            out[qi, r] = sc
        return out

    rprint("FP rerank pass:")
    s_fp = score_all()
    order_fp = np.argsort(-s_fp, axis=1)
    top1_fp = order_fp[:, 0]
    g = np.take_along_axis(s_fp, order_fp, axis=1)
    gaps_fp = g[:, :1] - g          # (n_q, K); column j = gap to rank-(j+1)

    n = apply_ptq_(model=inner, bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
                   skip_modules=frozenset(cfg.ptq.skip_modules))
    rprint(f"PTQ: bits={cfg.ptq.bits} gran={cfg.ptq.granularity}; quantized {len(n)} layers")
    s_q = score_all()
    top1_q = np.argsort(-s_q, axis=1)[:, 0]

    eps = np.abs(s_q - s_fp).max(1)
    flip = float((top1_q != top1_fp).mean())
    sep = float(np.median(gaps_fp[:, 1] / np.maximum(2 * eps, 1e-9)))
    rprint(f"\n  reranked top-1 flip rate : {flip:.2%}")
    rprint(f"  median separation ratio  : {sep:.4f}")
    inner.load_state_dict(fp_state)

    out_dir = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text" / "sentence_transformers" / \
        "crossencoder_gap_profile" / cfg.model_name.replace("/", "_") / cfg.dataset_name / \
        f"bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"seed={cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crossencoder_gap_profile.json").write_text(json.dumps(dict(
        model_name=cfg.model_name, biencoder_name=cfg.biencoder_name,
        dataset_name=cfg.dataset_name, top_k=int(cfg.top_k),
        bits=cfg.ptq.bits, granularity=cfg.ptq.granularity, seed=int(cfg.seed),
        n_queries=int(len(queries)), n_docs=int(len(docs)),
        flip_top1=flip, median_eps=float(np.median(eps)),
        median_gap2=float(np.median(gaps_fp[:, 1])), separation_ratio=sep,
        gap_profile_deciles={f"g{j}": [float(x) for x in
                             np.percentile(gaps_fp[:, j], [10, 50, 90])]
                             for j in (1, 2, 5)}), indent=2))
    rprint(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
