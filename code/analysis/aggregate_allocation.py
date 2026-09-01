"""Aggregate the allocation grid into capture rates.

capture = (metric(alloc) - metric(floor)) / (metric(ceil) - metric(floor))
i.e. what fraction of the benefit of one extra bit does this allocation buy,
at a budget that spends only a fraction of that bit. Reported on flip rate
(lower better, sign-flipped), Recall@1 and gold retention (higher better).
"""
import json, glob, os
from collections import defaultdict
import numpy as np

BASE = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/"
                          "sentence_transformers/gap_allocation")
POLICIES = ["random0", "random1", "random2", "mse", "gap"]


def gain_and_headroom(res, pol, key, higher_better):
    """Return (gain, headroom) rather than their ratio.

    Per-config ratios are unusable here: when quantisation barely moves a metric the
    denominator approaches zero and the ratio explodes, so a mean over configs is dominated
    by the least informative ones. Summing numerator and denominator separately gives the
    pooled fraction of available benefit actually captured, weighted by real headroom.
    """
    try:
        lo, hi, v = res["floor"][key], res["ceil"][key], res[pol][key]
    except KeyError:
        return (np.nan, np.nan)
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (lo, hi, v)):
        return (np.nan, np.nan)
    if not higher_better:
        lo, hi, v = -lo, -hi, -v
    return (v - lo, hi - lo)


def capture(res, pol, key, higher_better):
    try:
        lo, hi, v = res["floor"][key], res["ceil"][key], res[pol][key]
    except KeyError:
        return np.nan
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (lo, hi, v)):
        return np.nan
    if not higher_better:                      # flip / gold_lost: lower is better
        lo, hi, v = -lo, -hi, -v
    if abs(hi - lo) < 1e-9:
        return np.nan                          # no headroom -> undefined, not 0
    return (v - lo) / (hi - lo)


def main():
    rows = []
    for f in glob.glob(os.path.join(BASE, "**", "gap_allocation.json"), recursive=True):
        j = json.load(open(f))
        if j.get("avg_bits_target") is None or "group_sizes" in j:
            continue          # superseded group-size-trading schema, not a bit allocation
        seg = {}
        for p in f.split(os.sep):
            if "=" in p:
                k, v = p.split("=", 1)
                seg[k] = v
        meth = j.get("method", seg.get("method", "rtn"))
        seed = int(j.get("seed", seg.get("seed", -1)))
        avg = float(j.get("avg_bits_target", seg.get("avg", "nan")))
        for pol in POLICIES:
            if pol not in j["results"]:
                continue
            rows.append(dict(
                model=j["model_name"].split("/")[-1], ds=j["dataset_name"],
                method=meth, seed=seed, avg=avg, policy=pol,
                cap_flip=capture(j["results"], pol, "flip", False),
                cap_r1=capture(j["results"], pol, "recall1", True),
                cap_gold=capture(j["results"], pol, "gold_lost", False),
                gh_flip=gain_and_headroom(j["results"], pol, "flip", False),
                gh_r1=gain_and_headroom(j["results"], pol, "recall1", True),
                gh_gold=gain_and_headroom(j["results"], pol, "gold_lost", False)))
    if not rows:
        print("no results found under", BASE)
        return

    def agg(rs, key):
        """Pooled capture: sum(gain) / sum(headroom), ignoring configs with no headroom."""
        g = np.array([r[key][0] for r in rs], float)
        h = np.array([r[key][1] for r in rs], float)
        # Drop configs with no headroom: if W4 is no better than W3 on this metric there is
        # nothing for an allocation to capture, and including them only adds noise (one config
        # even has W4 scoring BELOW W3 on Recall@1).
        m = ~(np.isnan(g) | np.isnan(h)) & (h > 1e-6)
        g, h = g[m], h[m]
        if not len(h):
            return (np.nan, 0)
        return (g.sum() / h.sum(), len(h))

    g = defaultdict(list)
    for r in rows:
        pol = "random" if r["policy"].startswith("random") else r["policy"]
        g[(r["method"], r["avg"], pol)].append(r)

    print("pooled capture = sum(gain) / sum(headroom) over configs with positive headroom\n")
    print(f"{'method':<6} {'avg':>5} {'policy':<7} {'cap(flip)':>11} "
          f"{'cap(R@1)':>10} {'cap(gold)':>11} {'n':>5}")
    fmt = lambda x: "   n/a" if np.isnan(x) else f"{x:6.1%}"
    for k in sorted(g, key=lambda x: (x[0], x[1], x[2])):
        rs = g[k]
        mf, nf = agg(rs, "gh_flip")
        mr, _ = agg(rs, "gh_r1")
        mg, _ = agg(rs, "gh_gold")
        print(f"{k[0]:<6} {k[1]:>5} {k[2]:<7} {fmt(mf):>11} {fmt(mr):>10} "
              f"{fmt(mg):>11} {nf:>5}")

    cfgs = set((r['model'], r['ds'], r['method'], r['seed'], r['avg']) for r in rows)
    print(f"\nconfigs: {len(cfgs)}   models: {sorted(set(r['model'] for r in rows))}")


if __name__ == "__main__":
    main()
