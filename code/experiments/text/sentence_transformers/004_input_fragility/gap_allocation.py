"""Does allocating bits by gap sensitivity beat allocating them by reconstruction error?

Existing weight-only PTQ spends its budget uniformly or to minimise per-layer reconstruction
error. Corollary 5 says the operative quantity is the perturbation of the single scalar
z_(1) - z_(2), not the layer's own output error. This compares three allocations of the SAME
budget -- same average scales per weight, hence same memory and inference cost:

  uniform   every layer at the base group size (what the rest of the paper uses)
  mse       finer granularity to layers with the largest reconstruction error
  gap       finer granularity to layers whose quantization most perturbs the top-1/top-2 gap

and reports the resulting top-1 retrieval change rate on the four text corpora.
"""
import json, os, sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
from dotenv import load_dotenv
load_dotenv(_R / ".env")

import copy, hydra, numpy as np, torch
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from src.allocation import (gap_sensitivity, mse_sensitivity, hawq_sensitivity, allocate, apply_allocation_,
                            allocate_bits, apply_bit_allocation_, _linears,
                            activation_stats, hessian_sensitivity, awq_salience_sensitivity,
                            relative_error_sensitivity, position_sensitivity,
                            fisher_only_sensitivity, act_norm_sensitivity)
from src.gptq import apply_gptq_
from src.awq import apply_awq_

sys.path.insert(0, str(_R / "code/experiments/text/sentence_transformers/004_input_fragility"))
from retrieval_gap_profile import _load_corpus_and_queries, _encode


@hydra.main(config_path="../../../../../config/experiments/text/sentence_transformers/004_input_fragility",
            config_name="gap_allocation", version_base=None)
def main(cfg: DictConfig):
    rprint(OmegaConf.to_container(cfg, resolve=True))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.model_name, device=str(device))
    model.max_seq_length = cfg.max_length
    inner = model[0].auto_model
    fp_state = copy.deepcopy(inner.state_dict())

    # Load the FULL query split; the selection happens below, once `rel` is known, so that the
    # budget is spent on queries that actually carry relevance judgements. Sampling blind from
    # the full split wastes it: FiQA judges 648 of 6648 queries, so a blind 1000-query sample
    # retains only ~97 of them.
    docs, queries_all, _ = _load_corpus_and_queries(
        cfg.dataset_name, None, cfg.seed, return_index=True)
    # relevance judgements, so allocations can be compared on retrieval QUALITY as well as on
    # agreement with the FP ranking. Agreement is the paper's framing; quality is what a
    # practitioner is judged on, and the two are not the same quantity.
    rel = {}
    try:
        from datasets import load_dataset
        from retrieval_gap_profile import DATASET_TO_HF_PATH
        kw = dict(token=os.environ.get("HF_TOKEN"), cache_dir=os.environ.get("HF_DATASETS_CACHE"),
                  trust_remote_code=True, verification_mode="no_checks")
        hf = DATASET_TO_HF_PATH[cfg.dataset_name]
        cor = load_dataset(hf, "corpus", split="corpus", **kw)
        qry = load_dataset(hf, "queries", split="queries", **kw)
        qr = load_dataset(hf, "default", split="test", **kw)
        dpos = {str(i): n for n, i in enumerate(cor["_id"])}
        qpos = {str(i): n for n, i in enumerate(qry["_id"])}
        for r_ in qr:
            qi, di = qpos.get(str(r_["query-id"])), dpos.get(str(r_["corpus-id"]))
            if qi is not None and di is not None and float(r_["score"]) > 0:
                rel.setdefault(qi, {})[di] = float(r_["score"])
    except Exception as e:
        rprint(f"  (no relevance judgements available: {type(e).__name__})")
    all_judged_docs = {d for ds_ in rel.values() for d in ds_}
    rng = np.random.default_rng(cfg.seed)
    judged = sorted(rel)
    unjudged = [i for i in range(len(queries_all)) if i not in rel]
    if len(judged) > cfg.max_queries:
        q_sel = np.sort(rng.choice(judged, cfg.max_queries, replace=False))
    else:
        n_fill = min(cfg.max_queries - len(judged), len(unjudged))
        fill = rng.choice(unjudged, n_fill, replace=False) if n_fill else np.array([], dtype=int)
        q_sel = np.sort(np.concatenate([np.array(judged, dtype=int), fill.astype(int)]))
    queries = [queries_all[i] for i in q_sel]
    rel = {int(q): rel[int(q)] for q in q_sel if int(q) in rel}   # only evaluated queries
    d_sel = np.arange(len(docs))
    if len(docs) > cfg.max_docs:
        # A uniform subsample throws away most gold documents (FiQA keeps ~14% of them),
        # which does not bias the comparison BETWEEN allocations but makes the absolute
        # Recall@1 meaningless. Keep every judged document, fill the rest with random
        # distractors. If the judgements alone are denser than half the budget (TREC-COVID),
        # restrict to a subset of queries first so distractors still dominate.
        # Deeply-judged collections (TREC-COVID averages ~350 relevant docs per query) cannot
        # fit every judgement AND enough distractors. Keep the top-scoring GOLD_PER_QUERY
        # judgements per query instead of dropping whole queries, and draw distractors only from
        # documents judged for no query -- otherwise a "distractor" may in fact be relevant and
        # would be scored as a miss.
        GOLD_PER_QUERY = 10
        rel = {q: dict(sorted(ds_.items(), key=lambda kv: -kv[1])[:GOLD_PER_QUERY])
               for q, ds_ in rel.items()}
        gold_arr = np.array(sorted({d for ds_ in rel.values() for d in ds_}), dtype=int)
        pool = np.setdiff1d(np.arange(len(docs)), np.array(sorted(all_judged_docs), dtype=int))
        n_fill = max(cfg.max_docs - len(gold_arr), 0)
        fill = rng.choice(pool, min(n_fill, len(pool)), replace=False)
        d_sel = np.concatenate([gold_arr, fill])
        docs = [docs[i] for i in d_sel]
    # `rel` is keyed by RAW HuggingFace positions, but the query and document lists have both
    # been subsampled and reordered above. Remap into evaluation index space, or every recall
    # number is computed against scrambled gold labels.
    q_new = {int(o): n for n, o in enumerate(q_sel)}
    d_new = {int(o): n for n, o in enumerate(d_sel)}
    rel = {q_new[q]: {d_new[d] for d in ds if d in d_new}
           for q, ds in rel.items() if q in q_new}
    rel = {q: ds for q, ds in rel.items() if ds}
    rprint(f"\n{cfg.dataset_name}: {len(docs)} documents, {len(queries)} queries, "
           f"{len(rel)} with relevance judgements")

    rprint("\nFP encoding:")
    D_fp = np.asarray(_encode(model, docs, cfg.batch_size, is_query=False))
    Q_fp = np.asarray(_encode(model, queries, cfg.batch_size, is_query=True))
    s_fp = Q_fp @ D_fp.T
    fp_top1 = s_fp.argmax(1)

    # Calibration queries must not be evaluated. The allocation is CHOSEN using them, so
    # scoring it on the same queries reports a fitted number rather than a held-out one.
    # BEIR ships no separate query pool for these corpora, so hold out a seeded random slice
    # and report every rate on the disjoint remainder. The budget is capped at a quarter of
    # the pool so the evaluation set stays the larger part on the small corpora.
    n_cal = min(cfg.calib_queries, len(queries) // 4)
    _perm = rng.permutation(len(queries))
    cal_idx = np.sort(_perm[:n_cal])
    eval_idx = np.sort(_perm[n_cal:])
    cal_q = [queries[int(i)] for i in cal_idx]
    cal_docs = docs[:cfg.calib_docs]
    D_cal = np.asarray(_encode(model, cal_docs, cfg.batch_size, is_query=False))

    def score_fn():
        Q = np.asarray(_encode(model, cal_q, cfg.batch_size, is_query=True))
        return Q @ D_cal.T

    skip = frozenset(cfg.ptq.skip_modules)
    numel = {n: m.weight.numel() for n, m in _linears(inner, skip)}
    rprint(f"\nmeasuring gap sensitivity over {len(numel)} layers "
           f"({n_cal} calibration queries, {len(cal_docs)} documents):")
    g_sens = gap_sensitivity(inner, skip, cfg.ptq.bits, f"group_{cfg.base_group}", score_fn)
    m_sens = mse_sensitivity(inner, skip, cfg.ptq.bits, f"group_{cfg.base_group}")
    h_sens = None
    if bool(cfg.get("add_hawq", False)):
        # pair each calibration query with its FP top-1 document among the calibration docs
        s_cal = np.asarray(_encode(model, cal_q, cfg.batch_size, is_query=True)) @ D_cal.T
        paired_docs = [cal_docs[i] for i in s_cal.argmax(1)]
        rprint("measuring HAWQ (Fisher-trace) sensitivity:")
        h_sens = hawq_sensitivity(model, inner, skip, cfg.ptq.bits,
                                  f"group_{cfg.base_group}", cal_q, paired_docs)
    # Additional published allocation criteria, all scored on the same calibration queries.
    extra_sens = {}
    if bool(cfg.get("add_baselines", False)):
        rprint("measuring activation-based criteria (GPTQ-Hessian, AWQ-salience):")
        stats_act = activation_stats(inner, skip, lambda: _encode(model, cal_q, cfg.batch_size,
                                                                 is_query=True))
        extra_sens["hess"] = hessian_sensitivity(inner, skip, cfg.ptq.bits,
                                                 f"group_{cfg.base_group}", stats_act)
        extra_sens["awqsal"] = awq_salience_sensitivity(inner, skip, cfg.ptq.bits,
                                                        f"group_{cfg.base_group}", stats_act)
        extra_sens["relerr"] = relative_error_sensitivity(inner, skip, cfg.ptq.bits,
                                                          f"group_{cfg.base_group}")
        extra_sens["position"] = position_sensitivity(inner, skip)
        extra_sens["actnorm"] = act_norm_sensitivity(stats_act)
        del stats_act
    if h_sens is not None and (bool(cfg.get("add_baselines", False))
                              or bool(cfg.get("baselines_lite", False))):
        extra_sens["fisheronly"] = fisher_only_sensitivity(h_sens, inner, skip, cfg.ptq.bits,
                                                           f"group_{cfg.base_group}")

    from scipy import stats
    common = [k for k in g_sens if k in m_sens]
    rho, pv = stats.spearmanr([g_sens[k] for k in common], [m_sens[k] for k in common])
    rprint(f"  Spearman(gap, mse) over layers = {rho:+.3f} (p={pv:.2g})")

    # At an integer average, uniform is the only allocation, so there is nothing to compare.
    # The question is posed at a fractional budget: given avg_bits between two integers, WHICH
    # layers get the extra bit? Uniform is unavailable; the honest control is a random split at
    # the same budget, averaged over several draws.
    gran = f"group_{cfg.base_group}"
    choices = list(cfg.bit_choices)
    target = float(cfg.avg_bits)
    rnd = {}
    for t in range(cfg.n_random):
        r = np.random.default_rng(1000 + t)
        rnd[f"random{t}"] = allocate_bits({k: float(r.random()) for k in numel},
                                          numel, choices, target)
    allocs = {"floor": {k: min(choices) for k in numel},
              "ceil": {k: max(choices) for k in numel},
              **rnd,
              "mse": allocate_bits(m_sens, numel, choices, target),
              "gap": allocate_bits(g_sens, numel, choices, target),
              **{k: allocate_bits(v, numel, choices, target)
                 for k, v in extra_sens.items() if v}}
    if h_sens is not None:
        allocs["hawq"] = allocate_bits(h_sens, numel, choices, target)
    res = {}
    if bool(cfg.get("sens_only", False)):
        rprint("sens_only: skipping policy evaluation, recording allocations only")
        allocs_iter = []
    else:
        allocs_iter = list(allocs.items())
    for name, alloc in allocs_iter:
        inner.load_state_dict(fp_state)
        method = str(getattr(cfg.ptq, "method", "rtn")).lower()
        if method == "rtn":
            n = apply_bit_allocation_(inner, alloc, gran)
        else:
            # Allocation and rounding are separate choices: quantize each bit-width group with
            # the chosen error-minimizing quantizer, so the comparison isolates the allocation.
            fn = apply_gptq_ if method == "gptq" else apply_awq_
            cal = [cal_docs[i:i + cfg.batch_size] for i in range(0, len(cal_docs), cfg.batch_size)]
            n = 0
            for b in sorted(set(alloc.values())):
                only = frozenset(k for k, v in alloc.items() if v == b)
                n += len(fn(model=inner, bits=b, granularity=gran, skip_modules=skip,
                            only=only, calib_batches=cal,
                            forward_fn=lambda _m, x: _encode(model, x, cfg.batch_size, is_query=False),
                            verbose=False))
        D_q = np.asarray(_encode(model, docs, cfg.batch_size, is_query=False))
        Q_q = np.asarray(_encode(model, queries, cfg.batch_size, is_query=True))
        s_q = Q_q @ D_q.T
        q_top1 = s_q.argmax(1)
        flipped = (q_top1[eval_idx] != fp_top1[eval_idx])
        flip = float(flipped.mean())
        # Recall@1 against ground truth, and how often a correct top-1 is lost
        r1_fp = r1_q = lost = float("nan")
        if rel:
            idx = [int(i) for i in eval_idx if int(i) in rel]
            if idx:
                ok_fp = np.array([fp_top1[i] in rel[i] for i in idx])
                ok_q = np.array([q_top1[i] in rel[i] for i in idx])
                r1_fp, r1_q = float(ok_fp.mean()), float(ok_q.mean())
                lost = float((ok_fp & ~ok_q).sum() / max(ok_fp.sum(), 1))
        eps = float(np.median(np.abs(s_q - s_fp).max(1)))
        scales = sum(numel[k] * alloc[k] for k in numel) / sum(numel.values())
        # Store the per-query indicator too: a scalar rate cannot be re-analysed on a
        # different subset later without paying for the whole sweep again.
        res[name] = dict(flip=flip, median_eps=eps, avg_bits=scales, n_layers=n,
                         recall1_fp=r1_fp, recall1=r1_q, gold_lost=lost,
                         flip_by_query=[int(b) for b in flipped])
        rprint(f"  {name:<8} flip {flip:6.2%}   R@1 {r1_q:6.2%}   gold lost {lost:6.2%}   "
               f"avg bits {scales:.4f}")
    inner.load_state_dict(fp_state)

    _meth = str(getattr(cfg.ptq, "method", "rtn"))
    out = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text" / "sentence_transformers" / \
        "gap_allocation" / cfg.model_name.replace("/", "_").replace(".", "_") / cfg.dataset_name / \
        f"bits={cfg.ptq.bits}_base={cfg.base_group}" / f"method={_meth}" / \
        f"avg={float(target):g}" / f"seed={cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_allocation.json").write_text(json.dumps(
        dict(model_name=cfg.model_name, dataset_name=cfg.dataset_name,
             bits=cfg.ptq.bits, base_group=cfg.base_group, bit_choices=choices,
             method=_meth, seed=int(cfg.seed), avg_bits_target=target,
             n_calib_queries=int(n_cal), n_eval_queries=int(len(eval_idx)),
             n_docs=len(docs), n_calib_docs=int(min(cfg.calib_docs, len(docs))),
             eval_query_idx=[int(i) for i in eval_idx],
             layer_rho=float(rho), results=res,
             alloc_gap={k: int(v) for k, v in allocs["gap"].items()},
             # every criterion's layer assignment, so agreement between criteria can be
             # measured later without recomputing any sensitivity
             allocs={n: {k: int(v) for k, v in a.items()} for n, a in allocs.items()}), indent=2))
    rprint(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
