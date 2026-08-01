"""004 — Does QV similarity predict transfer gain? (timm supervised)

compute_qv_alignment.py measures how similar a donor's quantization vector is to
a receiver's.  This script answers the question that similarity was measured for,
which is the second half of reviewer 3HFP's objection:

    "Does it correlate with observed Delta(D,R) across the 22x22 vision matrix?
    If not, how should the theoretical account be interpreted?"

It joins the similarity of each donor-receiver pair against the accuracy outcome
of that same pair and reports the correlation.  Nothing about accuracy is
recomputed: Delta, Delta at the validation-selected lambda, and the recovery
ratio were all written by 998_rebuttal/001_zero_shot_reframing, and are read from
its JSON.

The quantity the reviewer is asking about
-----------------------------------------
Proposition 1 predicts that the fraction of the receiver's own QAT gain that a
donor patch recovers goes as cos^2 of the angle between the two QVs.  The
observable side of that is the recovery ratio Delta / Delta_ceiling, already
computed in 001.  So the single cell this whole sub-phase reduces to is

    recovery   versus   cosine^2   at model granularity,

and the printed summary leads with it.  Every other (metric, granularity,
operator) combination is reported underneath, because which aggregation is
defensible is exactly what was at issue and a result that holds under only one of
them is not a result.

Squaring happens here rather than in the compute script: storing cos and cos^2
side by side would be two copies of one number, free to disagree after an edit.

Why same-task pairs are dropped
-------------------------------
On the diagonal the donor is the receiver, so the similarity is 1 by algebra and
Delta is the receiver's own QAT ceiling rather than a transfer result.  Including
those 22 cells would inject a perfect-similarity, maximal-gain point into every
scatter and manufacture a correlation out of an identity.  They are excluded from
every statistic here, matching how 001 and 003 treat them.

The joined per-pair points are written alongside the coefficients, under
`points`.  They are what the visualization scripts scatter, and emitting them
here rather than letting a plot re-derive them is deliberate: the same-task
exclusion and the alignment-to-accuracy join would otherwise exist in two places,
and a figure whose population had drifted from the coefficient printed on it is
exactly the kind of error nobody catches by looking.

Pearson and Spearman are both reported.  Spearman is the more defensible of the
two: Proposition 1's functional form follows from a local quadratic model whose
validity across 22 heterogeneous tasks is exactly what is in question, whereas an
ordering claim -- better-aligned donors transfer better -- survives that
assumption failing.  n and a Fisher-z interval accompany every coefficient, so a
correlation over a handful of surviving pairs cannot be read as one over
hundreds.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from src.vision.utils import sanitize_timm_model_name

from qv_alignment_common import (
    GRANULARITY_LAYER,
    GRANULARITY_MODEL,
    GRANULARITY_NEURON,
    METRIC_COSINE,
    NEURON_MODES,
    pearson,
    spearman,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_timm_supervised"
MODALITY = "vision"

EVAL_ROOT_ALIGNMENT = "evaluations/vision/ilharco_timm_supervised/004_qv_alignment/vision/qv_alignment"
EVAL_ROOT_WIN_LOSS  = "evaluations/998_rebuttal/001_zero_shot_reframing"

IN_FILE  = "qv_alignment.json"
OUT_FILE = "qv_alignment_delta.json"

WIN_LOSS_FILE = f"win_loss_{FAMILY}.json"

# Accuracy outcomes to correlate against, as named in 001's per-pair records.
#   delta          transfer accuracy minus vanilla PTQ, at lambda = 1 (data-free)
#   delta_best     the same at the receiver-validation-selected lambda
#   recovery       delta / delta_ceiling, the observable side of the cos^2 law
#   recovery_best  the same at lambda_best
OUTCOMES = ("delta", "delta_best", "recovery", "recovery_best")

# The cell Proposition 1 is actually about.
HEADLINE = ("recovery", METRIC_COSINE, GRANULARITY_MODEL, None, "squared")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",   required=True, nargs="+")
    parser.add_argument("--seed",          required=True, type=int)

    parser.add_argument("--optim",         required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",            required=True, type=float)
    parser.add_argument("--wd",            required=True, type=float)
    parser.add_argument("--ls",            required=True, type=float)
    parser.add_argument("--wl",            required=True, type=int)
    parser.add_argument("--max-grad-norm", required=True, type=float)
    parser.add_argument("--batch-sizes",   required=True, type=int, nargs="+")

    parser.add_argument("--qat-bits",      required=True, type=int)
    parser.add_argument("--ptq-bits",      required=True, type=int)
    parser.add_argument("--granularity",   required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",  required=True, nargs="+")

    parser.add_argument("--metrics",       default=None, nargs="+",
                        help="Restrict which metrics are correlated. Default: every "
                             "metric present in the alignment JSON.")
    parser.add_argument("--granularities", default=None, nargs="+",
                        help="Restrict which granularities are correlated. Default: all present.")
    parser.add_argument("--operators",     default=None, nargs="+",
                        help="Restrict which operators are correlated. Default: all present.")

    parser.add_argument("--in-name",       default=IN_FILE,
                        help="Alignment filename to read, matching the compute "
                             "script's --out-name.")
    parser.add_argument("--out-name",      default=OUT_FILE)
    parser.add_argument("--drop-incomplete", action="store_true",
                        help="Skip models whose alignment and win/loss pair sets "
                             "disagree, instead of correlating over the intersection.")

    args = parser.parse_args()

    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")

    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _alignment_path(args, model_dir, optim_frag):
    return os.path.join(
        EVAL_ROOT_ALIGNMENT, model_dir,
        f"seed={args.seed}", optim_frag, _qat_frag(args), _ptq_frag(args),
        "alignment", args.in_name,
    )


def _win_loss_path(args):
    """001's grammar, which omits the skip tag from its qat=/ptq= fragments."""
    return os.path.join(
        EVAL_ROOT_WIN_LOSS,
        f"seed={args.seed}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        "split=test",
        WIN_LOSS_FILE,
    )


def _out_dir(args):
    return os.path.join(
        EVAL_ROOT_ALIGNMENT,
        f"seed={args.seed}", _qat_frag(args), _ptq_frag(args), "alignment",
    )


# ---------------------------------------------------------------------------
# Similarity series extraction
# ---------------------------------------------------------------------------
def similarity_series(pair, metric, granularity, mode, operator):
    """One similarity value out of a pair record, or None if it was not emitted.

    `mode` is the neuron aggregation mode, and is None for the layer and model
    granularities.  `operator` is None at model granularity, which has no
    aggregation step.
    """
    block = pair["metrics"].get(metric)
    if block is None:
        return None

    if granularity == GRANULARITY_MODEL:
        return block.get("model_wise")

    if granularity == GRANULARITY_LAYER:
        layer = block.get("layer_wise")
        return layer["agg"].get(operator) if layer else None

    neuron = block.get("neuron_wise")
    if not neuron or mode not in neuron:
        return None
    return neuron[mode].get(operator)


def enumerate_variants(alignment, args):
    """Every (metric, granularity, mode, operator) present and not filtered out.

    Read off an actual pair record rather than off `config`, so a variant is
    listed only if it is really in the file -- the two can differ when the JSON
    was written by an older run.
    """
    sample = alignment["pairs"][0]
    keep_m = set(args.metrics) if args.metrics else None
    keep_g = set(args.granularities) if args.granularities else None
    keep_o = set(args.operators) if args.operators else None

    variants = []
    for metric, block in sample["metrics"].items():
        if keep_m is not None and metric not in keep_m:
            continue

        if "model_wise" in block and (keep_g is None or GRANULARITY_MODEL in keep_g):
            variants.append((metric, GRANULARITY_MODEL, None, None))

        if "layer_wise" in block and (keep_g is None or GRANULARITY_LAYER in keep_g):
            for op in block["layer_wise"]["agg"]:
                if keep_o is None or op in keep_o:
                    variants.append((metric, GRANULARITY_LAYER, None, op))

        if "neuron_wise" in block and (keep_g is None or GRANULARITY_NEURON in keep_g):
            for mode in NEURON_MODES:
                if mode not in block["neuron_wise"]:
                    continue
                for op in block["neuron_wise"][mode]:
                    if keep_o is None or op in keep_o:
                        variants.append((metric, GRANULARITY_NEURON, mode, op))

    return variants


def variant_label(metric, granularity, mode, operator):
    parts = [metric, granularity]
    if mode is not None:
        parts.append(mode)
    if operator is not None:
        parts.append(operator)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def correlate(sims, outcomes_by_pair, variants, transform):
    """Pearson and Spearman of every variant against every outcome.

    `transform` is "raw" or "squared"; the squared form is what Proposition 1
    predicts the recovery ratio tracks.  Squaring is applied to the similarity,
    never to the outcome.
    """
    out = {}

    for variant in variants:
        xs_all = sims[variant]
        label = variant_label(*variant)
        block = {}

        for outcome in OUTCOMES:
            xs, ys = [], []
            for key, x in xs_all.items():
                y = outcomes_by_pair[key].get(outcome)
                if x is None or y is None:
                    continue
                xs.append(x * x if transform == "squared" else x)
                ys.append(y)

            block[outcome] = {"pearson": pearson(xs, ys), "spearman": spearman(xs, ys)}

        out[label] = block

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(stat):
    if stat["r"] is None:
        return f"    n/a (n={stat['n']})"
    ci = ""
    if stat["ci_lo"] is not None:
        ci = f" [{stat['ci_lo']:+.3f}, {stat['ci_hi']:+.3f}]"
    return f"{stat['r']:+.4f}{ci} (n={stat['n']})"


def _print_block(title, block, outcomes, limit=None):
    print(f"\n{title}")
    print("-" * len(title))

    rows = sorted(block.items(), key=lambda kv: -abs(
        kv[1][outcomes[0]]["spearman"]["r"] or 0.0
    ))
    if limit is not None:
        rows = rows[:limit]

    for label, per_outcome in rows:
        print(f"  {label}")
        for outcome in outcomes:
            s = per_outcome[outcome]
            print(f"    {outcome:<14} pearson {_fmt(s['pearson']):<34} "
                  f"spearman {_fmt(s['spearman'])}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    win_loss_path = _win_loss_path(args)
    if not os.path.exists(win_loss_path):
        sys.exit(
            f"missing {win_loss_path}\n"
            f"Run 998_rebuttal/001_zero_shot_reframing/compute_win_loss_timm_supervised.py "
            f"first -- this script never recomputes an accuracy quantity."
        )

    with open(win_loss_path) as f:
        win_loss = json.load(f)

    # 001 stores its models keyed by full timm name.
    outcomes_by_model = {}
    for model_name, block in win_loss["models"].items():
        outcomes_by_model[model_name] = {
            (p["donor"], p["receiver"]): p
            for p in block["pairs"]
            if not p["same_task"]
        }

    per_model = {}
    pooled_sims = {}
    pooled_outcomes = {}
    points = []
    variants = None
    missing = []

    for model_name, batch_size in zip(args.model_names, args.batch_sizes):
        model_dir  = sanitize_timm_model_name(model_name)
        path = _alignment_path(args, model_dir, _optim_frag(args, batch_size))

        if not os.path.exists(path):
            missing.append(path)
            print(f"[WARN] no alignment JSON for {model_name}: {path}", file=sys.stderr)
            continue

        with open(path) as f:
            alignment = json.load(f)

        outcomes = outcomes_by_model.get(model_name)
        if not outcomes:
            missing.append(f"win_loss:{model_name}")
            print(f"[WARN] {model_name} absent from {WIN_LOSS_FILE}", file=sys.stderr)
            continue

        model_variants = enumerate_variants(alignment, args)
        if variants is None:
            variants = model_variants
        elif set(variants) != set(model_variants):
            shared = sorted(set(variants) & set(model_variants))
            print(f"[WARN] {model_name} emits a different variant set; "
                  f"restricting to the {len(shared)} shared ones", file=sys.stderr)
            variants = shared

        cross = [p for p in alignment["pairs"] if not p["same_task"]]
        keys_align = {(p["donor"], p["receiver"]) for p in cross}
        keys_acc   = set(outcomes)

        if keys_align != keys_acc and args.drop_incomplete:
            print(f"[SKIP] {model_name}: {len(keys_align ^ keys_acc)} pairs differ "
                  f"between alignment and win/loss", file=sys.stderr)
            continue

        joined = keys_align & keys_acc
        if len(joined) < len(keys_align):
            print(f"[WARN] {model_name}: {len(keys_align) - len(joined)} alignment "
                  f"pairs have no accuracy counterpart", file=sys.stderr)

        sims = {
            v: {
                (p["donor"], p["receiver"]): similarity_series(p, *v)
                for p in cross if (p["donor"], p["receiver"]) in joined
            }
            for v in variants
        }
        outs = {k: outcomes[k] for k in joined}

        per_model[model_name] = {
            "display_name": alignment.get("display_name", model_name),
            "n_pairs":      len(joined),
            "raw":          correlate(sims, outs, variants, "raw"),
            "squared":      correlate(sims, outs, variants, "squared"),
        }

        # The individual points behind those coefficients, so a scatter plots
        # exactly the population the correlation was computed over.  Built from
        # the same `sims` and `outs` this loop already correlated -- no second
        # join and no second same-task filter, which is the whole point: two
        # implementations of "which pairs count" would eventually disagree, and
        # the figure would then be captioned with a rho it does not show.
        for key in sorted(joined):
            points.append({
                "model":         model_name,
                "display_name":  per_model[model_name]["display_name"],
                "donor":         key[0],
                "receiver":      key[1],
                **{o: outs[key].get(o) for o in OUTCOMES},
                "sims":          {variant_label(*v): sims[v][key] for v in variants},
            })

        for v in variants:
            pooled_sims.setdefault(v, {}).update(
                {(model_name, *k): x for k, x in sims[v].items()}
            )
        pooled_outcomes.update({(model_name, *k): o for k, o in outs.items()})

    if not per_model:
        sys.exit("no model produced a joinable alignment / win-loss pair; nothing to correlate")

    pooled = {
        "n_pairs": len(pooled_outcomes),
        "raw":     correlate(pooled_sims, pooled_outcomes, variants, "raw"),
        "squared": correlate(pooled_sims, pooled_outcomes, variants, "squared"),
    }

    # ---- report ------------------------------------------------------------
    outcome, metric, granularity, mode, transform = HEADLINE
    label = variant_label(metric, granularity, mode, None)

    print("=" * 78)
    print("Proposition 1's prediction: recovery ratio tracks cos^2 of the QV angle")
    print("=" * 78)
    if label in pooled[transform]:
        s = pooled[transform][label][outcome]
        print(f"  pooled over {len(per_model)} models, {pooled['n_pairs']} cross-task pairs")
        print(f"  {outcome} vs {label}^2")
        print(f"    pearson  {_fmt(s['pearson'])}")
        print(f"    spearman {_fmt(s['spearman'])}")
    else:
        print(f"  not available: {label} was not emitted "
              f"(re-run the compute script with --metrics {metric} "
              f"--granularities {granularity})")

    _print_block(
        f"Pooled, similarity squared — every variant, by |spearman| against {OUTCOMES[2]}",
        pooled["squared"], ["recovery", "delta"], limit=12,
    )
    _print_block(
        f"Pooled, similarity raw — every variant, by |spearman| against {OUTCOMES[0]}",
        pooled["raw"], ["delta", "delta_best"], limit=12,
    )

    for model_name, block in per_model.items():
        _print_block(
            f"{block['display_name']} ({block['n_pairs']} pairs), squared",
            block["squared"], ["recovery", "delta"], limit=5,
        )

    payload = {
        "family":   FAMILY,
        "modality": MODALITY,
        "config": {
            "seed":          args.seed,
            "optim":         args.optim,
            "lr":            args.lr,
            "wd":            args.wd,
            "ls":            args.ls,
            "wl":            args.wl,
            "max_grad_norm": args.max_grad_norm,
            "qat_bits":      args.qat_bits,
            "ptq_bits":      args.ptq_bits,
            "granularity":   args.granularity,
            "skip_modules":  sorted(args.skip_modules),
            "model_names":   list(args.model_names),
            "outcomes":      list(OUTCOMES),
            "delta_source":  win_loss_path,
            "same_task":     "excluded: similarity is 1 by algebra there",
        },
        "variants":  [variant_label(*v) for v in variants],
        "pooled":    pooled,
        "per_model": per_model,
        "points":    points,
        "missing":   missing,
    }

    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
