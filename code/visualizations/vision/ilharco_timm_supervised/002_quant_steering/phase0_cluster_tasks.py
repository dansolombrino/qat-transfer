"""Phase 0 follow-up: hierarchical clustering of tasks by quant-steering direction.

If `phase0_cross_task_cosine.py` reports near-zero *signed* mean cosine but
non-trivial *absolute* mean cosine, the data is consistent with task clusters
that share a direction internally but disagree on sign across clusters. This
script confirms or refutes that hypothesis directly.

For a chosen method and block (defaulting to block 0, where the absolute-cosine
signal was strongest in the W4-channel sweep), it:

  1. Loads each task's steering vector at that block.
  2. Builds the T×T sign-invariant similarity matrix |cos(v_i, v_j)|.
  3. Runs hierarchical clustering on the 1−|cos| distance.
  4. Cuts the tree at a user-specified number of clusters and prints memberships.
  5. Renders an HTML figure with the dendrogram + a heatmap reordered to the
     dendrogram leaf order (so blocks of high |cos| appear contiguous).

If the dendrogram shows tight clusters with high within-cluster |cos| and low
between-cluster |cos|, the right cross-task combiner is per-cluster averaging
(plus sign-alignment within each cluster) rather than global averaging.
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
load_dotenv()

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.vision.utils import sanitize_timm_model_name


METHODS = ("mean_diff", "contrastive_svd")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", required=True)
    p.add_argument("--lr", default="1e-05")
    p.add_argument("--wd", default="0.1")
    p.add_argument("--ls", default="0.0")
    p.add_argument("--wl", default="500")
    p.add_argument("--max-grad-norm", default="1.0")
    p.add_argument("--batch-size", default="128")
    p.add_argument("--seed", default="2038")
    p.add_argument("--bits", required=True, type=int)
    p.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    p.add_argument("--skip-modules", nargs="+", default=["head"])
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--method", default="mean_diff", choices=list(METHODS))
    p.add_argument(
        "--blocks", nargs="+", type=int, default=[0],
        help="Block index/indices to cluster on. Multiple blocks → concatenate then cluster.",
    )
    p.add_argument("--num-clusters", type=int, default=4)
    p.add_argument(
        "--linkage-method", default="average",
        choices=["single", "complete", "average", "weighted", "ward"],
    )
    p.add_argument("--min-bad", type=int, default=20)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _build_paths(args):
    checkpoint_base = Path(os.environ["CHECKPOINT_BASE_PATH"])
    sanitized = sanitize_timm_model_name(args.model_name)
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    base = (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "steering_vectors" / sanitized
    )
    optim_tag = (
        f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
        f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
    )
    ptq_tag = f"ptq=bits={args.bits}_gran={args.granularity}_skip={skip_tag}"
    seed_tag = f"seed={args.seed}"
    return base, optim_tag, ptq_tag, seed_tag, sanitized


def _load_vectors(base: Path, datasets, optim_tag, ptq_tag, seed_tag, method, min_bad):
    vectors_by_task: dict[str, np.ndarray] = {}
    skipped = []
    for ds in datasets:
        task_dir = base / ds / optim_tag / ptq_tag / seed_tag
        vec_path = task_dir / "steering_vectors.pt"
        meta_path = task_dir / "fit_metadata.json"
        if not vec_path.exists():
            skipped.append((ds, "no steering_vectors.pt"))
            continue
        payload = torch.load(vec_path, map_location="cpu", weights_only=True)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        num_bad = int(meta.get("num_bad", payload.get("num_bad", -1)))
        if 0 <= num_bad < min_bad:
            skipped.append((ds, f"num_bad={num_bad} < {min_bad}"))
            continue
        vectors_by_task[ds] = payload[method].numpy()  # (L, D)
    return vectors_by_task, skipped


def _build_feature_matrix(vectors_by_task, blocks):
    """For each task, concatenate vector slices at the requested blocks and
    unit-normalize the concatenation. Returns (T, len(blocks)*D)."""
    task_list = sorted(vectors_by_task.keys())
    rows = []
    for t in task_list:
        v = vectors_by_task[t][blocks].reshape(-1)  # (len(blocks) * D,)
        v = v / (np.linalg.norm(v) + 1e-12)
        rows.append(v)
    return np.stack(rows, axis=0), task_list


def _abs_cosine_distance(X):
    """|cos|-based distance: 1 - |X X^T|. X must already be row-normalized."""
    sim = X @ X.T
    abs_sim = np.abs(sim)
    np.fill_diagonal(abs_sim, 1.0)
    dist = 1.0 - abs_sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)  # guard against -0
    # Force symmetry (squareform is strict).
    dist = (dist + dist.T) * 0.5
    return dist, sim


def _print_clusters(task_list, cluster_ids, sim):
    print(f"\n=== Cluster memberships (k={cluster_ids.max()}) ===")
    for cid in sorted(set(cluster_ids)):
        members = [task_list[i] for i in range(len(task_list)) if cluster_ids[i] == cid]
        # Within-cluster mean |cos|
        idx = [i for i in range(len(task_list)) if cluster_ids[i] == cid]
        if len(idx) > 1:
            sub = sim[np.ix_(idx, idx)]
            off = sub[~np.eye(len(idx), dtype=bool)]
            within = float(np.abs(off).mean())
        else:
            within = float("nan")
        print(f"\n  cluster {cid}  (n={len(members)}, within-|cos|={within:.3f}):")
        for m in members:
            print(f"    - {m}")

    # Between-cluster mean |cos|
    print("\n  Between-cluster mean |cos| (lower = cleaner separation):")
    cluster_ids_arr = np.asarray(cluster_ids)
    uniq = sorted(set(cluster_ids))
    print("        " + "  ".join(f"  c{c}" for c in uniq))
    for ci in uniq:
        row = []
        for cj in uniq:
            if ci == cj:
                row.append("    -")
                continue
            i_idx = np.where(cluster_ids_arr == ci)[0]
            j_idx = np.where(cluster_ids_arr == cj)[0]
            sub = np.abs(sim[np.ix_(i_idx, j_idx)])
            row.append(f"{sub.mean():.2f}")
        print(f"  c{ci}    " + "  ".join(f"{v:>4s}" for v in row))


def _render_html(task_list, sim, link, cluster_ids, out_html: Path, title: str):
    """Side-by-side: dendrogram + heatmap reordered to match dendrogram leaves."""
    # Use scipy dendrogram to get the leaf order; render via plotly.
    dn = dendrogram(link, labels=task_list, no_plot=True)
    leaf_order = dn["leaves"]
    reordered_tasks = [task_list[i] for i in leaf_order]
    abs_sim_reordered = np.abs(sim)[np.ix_(leaf_order, leaf_order)]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.35, 0.65],
        subplot_titles=("Dendrogram (1 − |cos|)", "Pairwise |cos| reordered by dendrogram"),
    )

    # --- dendrogram traces ---
    icoord = np.asarray(dn["icoord"])
    dcoord = np.asarray(dn["dcoord"])
    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(
            go.Scatter(
                x=ys, y=xs, mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )
    # Y-axis ticks for dendrogram = leaf positions (scipy uses 5, 15, 25, …)
    leaf_y = 5 + 10 * np.arange(len(task_list))
    fig.update_yaxes(
        tickmode="array", tickvals=leaf_y, ticktext=reordered_tasks,
        row=1, col=1, autorange="reversed",
    )
    fig.update_xaxes(title_text="distance", row=1, col=1)

    # --- heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=abs_sim_reordered,
            x=reordered_tasks, y=reordered_tasks,
            colorscale="Viridis", zmin=0.0, zmax=1.0,
            colorbar=dict(title="|cos|"),
        ),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(title=title, height=600, width=1400)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))


def main():
    args = parse_args()
    base, optim_tag, ptq_tag, seed_tag, sanitized = _build_paths(args)

    datasets = args.datasets or sorted(d.name for d in base.iterdir() if d.is_dir())
    if not datasets:
        print(f"No datasets under {base}", file=sys.stderr)
        sys.exit(1)
    print(f"Method: {args.method}   Blocks: {args.blocks}   "
          f"PTQ: bits={args.bits} {args.granularity}")
    print(f"Discovered {len(datasets)} candidate dataset(s) under {base}\n")

    vectors_by_task, skipped = _load_vectors(
        base, datasets, optim_tag, ptq_tag, seed_tag, args.method, args.min_bad,
    )
    print(f"Loaded {len(vectors_by_task)} task(s); skipped {len(skipped)}:")
    for ds, reason in skipped:
        print(f"  - {ds}: {reason}")

    if len(vectors_by_task) < args.num_clusters:
        print(
            f"Need at least {args.num_clusters} usable tasks; got {len(vectors_by_task)}",
            file=sys.stderr,
        )
        sys.exit(1)

    X, task_list = _build_feature_matrix(vectors_by_task, args.blocks)
    dist, sim = _abs_cosine_distance(X)

    print(f"\nClustering {len(task_list)} tasks on |cos| at blocks {args.blocks} "
          f"via {args.linkage_method} linkage")
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method=args.linkage_method)
    cluster_ids = fcluster(link, t=args.num_clusters, criterion="maxclust")

    _print_clusters(task_list, cluster_ids, sim)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "002_quant_steering"
    blocks_tag = "-".join(str(b) for b in args.blocks)
    out_html = out_dir / (
        f"phase0_cluster_{sanitized}_bits{args.bits}_{args.granularity}"
        f"_method={args.method}_blocks={blocks_tag}_k={args.num_clusters}_seed{args.seed}.html"
    )
    title = (
        f"Task clustering — {args.model_name} W{args.bits} {args.granularity} | "
        f"{args.method} @ blocks {args.blocks} | k={args.num_clusters}"
    )
    _render_html(task_list, sim, link, cluster_ids, out_html, title)
    print(f"\nHTML report saved: {out_html}")


if __name__ == "__main__":
    main()
