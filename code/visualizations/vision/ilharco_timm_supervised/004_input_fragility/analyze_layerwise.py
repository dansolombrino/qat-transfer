"""Analyse the layer-wise logit-lens dumps: where does PTQ fragility appear?

Consumes the parquets written by
`code/experiments/vision/ilharco_timm_supervised/004_input_fragility/layerwise_logit_lens.py`
and answers, per transformer block i:

  1. AUROC of `q_margin_l{i}` for predicting the final `bad` label, on FP-correct
     rows. This is the payoff: the layer at which this saturates is the earliest
     point a routing decision could be made.
  2. FP/PTQ top-1 agreement -- the layer where the two models start to disagree.
  3. Top-5 overlap between FP and PTQ rankings -- ranking drift, not just argmax.
  4. Commitment layer: the first block after which the model's top-1 no longer
     changes, reported separately for good / bad / lucky-Q inputs.

Writes a markdown report under plots/ and a plotly figure.

Intermediate logits are NOT calibrated (the head is trained for the last block),
so margin magnitudes are not comparable across layers. Everything reported here
is rank-based and therefore scale-free.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

from src.vision.utils import sanitize_timm_model_name

DUMP_ROOT = "layerwise_logit_lens"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default="vit_base_patch16_224.orig_in21k")
    p.add_argument("--batch-size", default="128")
    p.add_argument("--lr", default="1e-05")
    p.add_argument("--wd", default="0.1")
    p.add_argument("--ls", default="0.0")
    p.add_argument("--wl", default="500")
    p.add_argument("--max-grad-norm", default="1.0")
    p.add_argument("--seed", default="2038")
    p.add_argument("--skip-modules", nargs="+", default=["head"])
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--granularity", default="channel")
    p.add_argument("--min-bad", type=int, default=10,
                   help="skip tasks with fewer than this many bad test samples")
    p.add_argument("--out-path", default="plots/004_input_fragility/layerwise_report.md")
    p.add_argument("--fig-path", default="plots/004_input_fragility/layerwise_auroc.html")
    return p.parse_args()


def _load(args):
    """Returns {dataset: (df, n_blocks)} for every task with a dump on disk."""
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    sanitized = sanitize_timm_model_name(args.model_name)
    base = (Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision"
            / "ilharco_timm_supervised" / DUMP_ROOT / sanitized)
    if not base.exists():
        return {}
    out = {}
    for ds_dir in sorted(base.iterdir()):
        if not ds_dir.is_dir():
            continue
        d = (ds_dir
             / f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
               f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
             / f"ptq=bits={args.bits}_gran={args.granularity}_skip={skip_tag}"
             / f"seed={args.seed}")
        pq, mj = d / "layerwise_test.parquet", d / "layerwise_metadata.json"
        if not pq.exists():
            continue
        meta = json.loads(mj.read_text()) if mj.exists() else {}
        out[ds_dir.name] = (pd.read_parquet(pq), int(meta.get("n_blocks", 12)))
    return out


def _per_task_stats(df, n_blocks):
    """All per-layer statistics for one task."""
    fpc = df[df["fp_correct"]]
    y = fpc["bad"].values
    st = {"n_test": len(df), "n_bad": int(df["bad"].sum()),
          "n_lucky": int(df["lucky_q"].sum())}

    if y.sum() >= 1 and (~y.astype(bool)).sum() >= 1:
        st["auroc_q"] = [roc_auc_score(y, -fpc[f"q_margin_l{i}"].values) for i in range(n_blocks)]
        st["auroc_fp"] = [roc_auc_score(y, -fpc[f"fp_margin_l{i}"].values) for i in range(n_blocks)]
    else:
        st["auroc_q"] = st["auroc_fp"] = [np.nan] * n_blocks

    st["agree"] = [float((df[f"fp_top1_l{i}"] == df[f"q_top1_l{i}"]).mean()) for i in range(n_blocks)]
    st["overlap"] = [float(df[f"top5_overlap_l{i}"].mean()) for i in range(n_blocks)]

    # commitment layer: last block at which top-1 differs from the final top-1, +1
    last = n_blocks - 1
    for tag, mask in (("good", df["good"]), ("bad", df["bad"]), ("lucky", df["lucky_q"])):
        sub = df[mask]
        if len(sub) == 0:
            st[f"commit_{tag}"] = np.nan
            continue
        final = sub[f"q_top1_l{last}"].values
        commit = np.zeros(len(sub), dtype=int)
        for i in range(n_blocks):
            differs = sub[f"q_top1_l{i}"].values != final
            commit = np.where(differs, i + 1, commit)
        st[f"commit_{tag}"] = float(commit.mean())
    return st


def main():
    args = parse_args()
    tasks = _load(args)
    if not tasks:
        print("No layer-wise dumps found. Run layerwise_logit_lens.py first.")
        return

    eligible = {k: v for k, v in tasks.items() if int(v[0]["bad"].sum()) >= args.min_bad}
    n_blocks = max(v[1] for v in tasks.values())
    print(f"Loaded {len(tasks)} task(s); {len(eligible)} with >= {args.min_bad} bad samples.")

    per = {ds: _per_task_stats(df, nb) for ds, (df, nb) in eligible.items()}
    if not per:
        print("No eligible task has enough bad samples to compute AUROC.")
        return

    def mean_of(key):
        arr = np.array([per[ds][key] for ds in per], dtype=float)
        return np.nanmean(arr, axis=0)

    auroc_q, auroc_fp = mean_of("auroc_q"), mean_of("auroc_fp")
    agree, overlap = mean_of("agree"), mean_of("overlap")

    L = [];  L.append("# Layer-wise logit lens — where PTQ fragility appears\n")
    L.append(f"Model `{args.model_name}`, W{args.bits}-{args.granularity}, "
             f"{len(per)} eligible task(s) (>= {args.min_bad} bad test samples).\n")
    L.append("Intermediate logits are uncalibrated (the head is trained for the last "
             "block), so only rank-based quantities are meaningful.\n")

    L.append("\n## Per-layer means across tasks\n")
    L.append("| block | AUROC q_margin | AUROC fp_margin | FP/PTQ top-1 agree | top-5 overlap |")
    L.append("|---|---|---|---|---|")
    for i in range(n_blocks):
        L.append(f"| {i} | {auroc_q[i]:.3f} | {auroc_fp[i]:.3f} | "
                 f"{agree[i]*100:.1f}% | {overlap[i]:.2f}/5 |")

    final = auroc_q[-1]
    reach = next((i for i in range(n_blocks) if auroc_q[i] >= 0.95 * final), None)
    L.append(f"\n**Earliest block reaching 95% of the final AUROC ({final:.3f}): "
             f"block {reach} of {n_blocks-1}.**")
    if reach is not None and reach < n_blocks - 1:
        saved = (n_blocks - 1 - reach) / n_blocks
        L.append(f"A routing decision taken there would skip the last "
                 f"{n_blocks-1-reach} of {n_blocks} blocks (~{saved*100:.0f}% of the "
                 f"quantized forward pass) on routed inputs.")

    L.append("\n## Commitment layer (mean last block at which top-1 still changes)\n")
    L.append("| group | commitment layer |")
    L.append("|---|---|")
    for tag in ("good", "bad", "lucky"):
        v = np.nanmean([per[ds][f"commit_{tag}"] for ds in per])
        L.append(f"| {tag} | {v:.2f} |")
    L.append("\nLater commitment for `bad` than `good` would mean fragile inputs stay "
             "undecided deeper into the network.")

    L.append("\n## Per-task final-layer sanity\n")
    L.append("| task | n_test | n_bad | AUROC q_margin @ last block |")
    L.append("|---|---|---|---|")
    for ds in sorted(per):
        L.append(f"| {ds} | {per[ds]['n_test']} | {per[ds]['n_bad']} | {per[ds]['auroc_q'][-1]:.3f} |")

    out = Path(args.out_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"Report saved: {out}")

    fig = go.Figure()
    x = list(range(n_blocks))
    fig.add_trace(go.Scatter(x=x, y=auroc_q, name="q_margin", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=x, y=auroc_fp, name="fp_margin", mode="lines+markers"))
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="chance")
    fig.update_layout(title=f"Per-layer AUROC for predicting PTQ-broken inputs "
                            f"({args.model_name}, W{args.bits}-{args.granularity})",
                      xaxis_title="transformer block", yaxis_title="AUROC",
                      template="plotly_white")
    figp = Path(args.fig_path); figp.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(figp))
    print(f"Figure saved: {figp}")


if __name__ == "__main__":
    main()
