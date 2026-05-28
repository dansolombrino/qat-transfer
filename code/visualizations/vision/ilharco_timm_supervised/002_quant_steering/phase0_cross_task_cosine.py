"""Phase 0 cross-task diagnostic for quantization-steering vectors.

Reads per-task `steering_vectors.pt` files produced by
`fit_steering_vector.py`, computes per-block cross-task cosine statistics for
both methods (mean_diff, contrastive_svd), and reports:

  - A markdown summary of mean signed cosine, mean |cosine|, and the
    fraction of task pairs with cosine > 0.3, per block per method.
  - An HTML line plot of these stats per block, with reference lines at the
    decision thresholds (0.2 partial / 0.6 strong) from CROSS_TASK_PLAN.md.
  - The pairwise cosine matrix at the best-aligned block, as an HTML
    heatmap, for sanity inspection.

No GPU. Auto-discovers datasets from disk.
"""

import argparse
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

import json
import numpy as np
import torch
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
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="If omitted, auto-discover every dataset that has a steering_vectors.pt at this config.",
    )
    p.add_argument(
        "--min-bad", type=int, default=20,
        help="Skip tasks whose bad-group has fewer than this many samples; their vectors are noisy garbage.",
    )
    p.add_argument(
        "--strong-threshold", type=float, default=0.6,
        help="Decision threshold for 'strong shared direction' per CROSS_TASK_PLAN.md.",
    )
    p.add_argument(
        "--partial-threshold", type=float, default=0.2,
        help="Decision threshold for 'partial overlap'.",
    )
    p.add_argument(
        "--shared-cos-threshold", type=float, default=0.3,
        help="Pair-cosine threshold for the 'fraction of pairs aligned' statistic.",
    )
    p.add_argument("--out-dir", default=None, help="Output dir for HTML. Defaults to plots/...")
    return p.parse_args()


def _build_paths(args: argparse.Namespace):
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


def _discover_datasets(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())


def _load_task(base: Path, dataset: str, optim_tag: str, ptq_tag: str, seed_tag: str):
    """Return (vectors_dict, metadata_dict) or (None, None) if files don't exist."""
    task_dir = base / dataset / optim_tag / ptq_tag / seed_tag
    vec_path = task_dir / "steering_vectors.pt"
    meta_path = task_dir / "fit_metadata.json"
    if not vec_path.exists():
        return None, None
    payload = torch.load(vec_path, map_location="cpu", weights_only=True)
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return payload, meta


def _pairwise_cosines(V: np.ndarray) -> np.ndarray:
    """V shape (T, L, D); returns (L, T, T) cosine matrix per block."""
    norms = np.linalg.norm(V, axis=2, keepdims=True)
    V_unit = V / (norms + 1e-12)
    return np.einsum("iLD,jLD->Lij", V_unit, V_unit)


def _off_diag_stats(cos_matrix: np.ndarray, shared_thr: float):
    """cos_matrix shape (L, T, T) -> dict of per-block stats (L,)."""
    L, T, _ = cos_matrix.shape
    eye = np.eye(T, dtype=bool)
    off = ~eye
    flat = cos_matrix[:, off]  # (L, T*(T-1))
    return dict(
        mean_signed=flat.mean(axis=1),
        mean_abs=np.abs(flat).mean(axis=1),
        median_signed=np.median(flat, axis=1),
        min_signed=flat.min(axis=1),
        max_signed=flat.max(axis=1),
        frac_above=np.mean(flat > shared_thr, axis=1),
    )


def _print_table(stats: dict, method: str, strong: float, partial: float):
    print(f"\n=== {method} ===")
    print(
        f"  block | mean(signed) | mean(|.|) | median | min   | max   | "
        f"frac>{stats.get('shared_thr', '?')}"
    )
    print(f"  {'-' * 6}-+-{'-' * 12}-+-{'-' * 9}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 9}")
    L = len(stats["mean_signed"])
    for l in range(L):
        ms = stats["mean_signed"][l]
        tag = ""
        if ms > strong:
            tag = "  ← strong shared"
        elif ms > partial:
            tag = "  ← partial overlap"
        elif ms < -partial:
            tag = "  ← anti-aligned (sign flips across tasks?)"
        print(
            f"  {l:>5d} | {ms:+12.3f} | {stats['mean_abs'][l]:9.3f} | "
            f"{stats['median_signed'][l]:+.3f} | {stats['min_signed'][l]:+.2f} | "
            f"{stats['max_signed'][l]:+.2f} | {stats['frac_above'][l]:.2%}{tag}"
        )


def _verdict(method_stats: dict, strong: float, partial: float) -> str:
    """Top-line decision rule per CROSS_TASK_PLAN.md."""
    lines = []
    for method, stats in method_stats.items():
        # Verdict driven by the BEST block (max mean_signed cosine).
        ms = stats["mean_signed"]
        best_block = int(np.argmax(ms))
        best_val = float(ms[best_block])
        # Also check mean_abs — diagnoses sign flipping
        ma = stats["mean_abs"]
        best_abs_block = int(np.argmax(ma))
        best_abs_val = float(ma[best_abs_block])
        if best_val > strong:
            verdict = "STRONG SHARED → Approach A (averaging) is enough"
        elif best_val > partial:
            verdict = "PARTIAL OVERLAP → Approach B (stacked SVD) recommended"
        elif best_abs_val > partial:
            verdict = (
                f"SIGN-FLIPPED SHARED → axes align but signs disagree "
                f"(|cos|={best_abs_val:.2f} at block {best_abs_block}); "
                f"re-align signs before combining"
            )
        else:
            verdict = "NO SHARED DIRECTION → cross-task transfer dead for this method"
        lines.append(
            f"  [{method}]  best signed cos = {best_val:+.3f} at block {best_block}  "
            f"|  best |cos| = {best_abs_val:.3f} at block {best_abs_block}"
        )
        lines.append(f"             → {verdict}")
    return "\n".join(lines)


def _render_html(
    blocks: range,
    method_stats: dict,
    cos_matrix_by_method: dict,
    task_list: list,
    out_html: Path,
    title: str,
    strong: float,
    partial: float,
):
    methods = list(method_stats.keys())
    n_methods = len(methods)
    fig = make_subplots(
        rows=n_methods + 1, cols=2,
        specs=[[{"colspan": 2}, None] for _ in range(n_methods)] + [[{}, {}]],
        subplot_titles=(
            [f"{m} — per-block cross-task cosine" for m in methods]
            + [f"{m} — pairwise cosine @ best block" for m in methods]
        ),
        row_heights=[0.3] * n_methods + [0.6 if n_methods >= 1 else 0.7],
        vertical_spacing=0.08,
    )

    for row_idx, method in enumerate(methods, start=1):
        s = method_stats[method]
        fig.add_trace(
            go.Scatter(
                x=list(blocks), y=s["mean_signed"], mode="lines+markers",
                name=f"{method}: mean(signed)",
                line=dict(width=2),
            ),
            row=row_idx, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(blocks), y=s["mean_abs"], mode="lines+markers",
                name=f"{method}: mean(|cos|)",
                line=dict(width=2, dash="dot"),
            ),
            row=row_idx, col=1,
        )
        fig.add_hline(
            y=strong, line_dash="dash", line_color="green", row=row_idx, col=1,
            annotation_text="strong", annotation_position="right",
        )
        fig.add_hline(
            y=partial, line_dash="dash", line_color="orange", row=row_idx, col=1,
            annotation_text="partial", annotation_position="right",
        )
        fig.add_hline(y=0, line_color="grey", row=row_idx, col=1)
        fig.update_yaxes(range=[-1.05, 1.05], row=row_idx, col=1)

    # Pairwise heatmap at the best-aligned block per method
    for col_idx, method in enumerate(methods, start=1):
        s = method_stats[method]
        best_block = int(np.argmax(s["mean_signed"]))
        mat = cos_matrix_by_method[method][best_block]
        fig.add_trace(
            go.Heatmap(
                z=mat, x=task_list, y=task_list,
                colorscale="RdBu", zmin=-1, zmax=1,
                colorbar=dict(title="cos"),
                showscale=(col_idx == 1),
                name=f"{method} pairwise (block {best_block})",
            ),
            row=n_methods + 1, col=col_idx,
        )

    fig.update_xaxes(title_text="Block index", row=n_methods, col=1)
    fig.update_layout(
        title=title, height=300 * n_methods + 700,
        showlegend=True,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))


def main() -> None:
    args = parse_args()
    base, optim_tag, ptq_tag, seed_tag, sanitized = _build_paths(args)

    datasets = args.datasets or _discover_datasets(base)
    if not datasets:
        print(f"No datasets found under {base}", file=sys.stderr)
        sys.exit(1)
    print(f"Searching for fits under: {base}")
    print(f"  optim_tag = {optim_tag}")
    print(f"  ptq_tag   = {ptq_tag}")
    print(f"  seed_tag  = {seed_tag}\n")

    vectors_by_task: dict[str, dict[str, np.ndarray]] = {}
    skipped: list[tuple[str, str]] = []
    for ds in datasets:
        payload, meta = _load_task(base, ds, optim_tag, ptq_tag, seed_tag)
        if payload is None:
            skipped.append((ds, "no steering_vectors.pt"))
            continue
        num_bad = int(meta.get("num_bad", payload.get("num_bad", -1)))
        if 0 <= num_bad < args.min_bad:
            skipped.append((ds, f"num_bad={num_bad} < {args.min_bad}"))
            continue
        vectors_by_task[ds] = {
            m: payload[m].numpy() for m in METHODS if m in payload
        }

    print(f"Loaded {len(vectors_by_task)} task(s); skipped {len(skipped)}:")
    for ds, reason in skipped:
        print(f"  - {ds}: {reason}")
    print()

    if len(vectors_by_task) < 2:
        print("Need at least 2 usable tasks for pairwise analysis", file=sys.stderr)
        sys.exit(1)

    task_list = sorted(vectors_by_task.keys())
    L, D = next(iter(vectors_by_task.values()))["mean_diff"].shape
    T = len(task_list)
    print(f"  {T} tasks × {L} blocks × {D}-D\n")

    method_stats: dict[str, dict] = {}
    cos_by_method: dict[str, np.ndarray] = {}
    for method in METHODS:
        V = np.stack(
            [vectors_by_task[t][method] for t in task_list], axis=0
        )  # (T, L, D)
        cos_matrix = _pairwise_cosines(V)  # (L, T, T)
        stats = _off_diag_stats(cos_matrix, args.shared_cos_threshold)
        stats["shared_thr"] = args.shared_cos_threshold
        method_stats[method] = stats
        cos_by_method[method] = cos_matrix
        _print_table(stats, method, args.strong_threshold, args.partial_threshold)

    print("\n=== Verdict (per CROSS_TASK_PLAN.md decision rule) ===")
    print(_verdict(method_stats, args.strong_threshold, args.partial_threshold))
    print()

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "002_quant_steering"
    out_html = out_dir / (
        f"phase0_cross_task_cosine_{sanitized}"
        f"_bits{args.bits}_{args.granularity}_seed{args.seed}.html"
    )
    title = (
        f"Phase 0 cross-task cosine — {args.model_name} "
        f"W{args.bits} {args.granularity} ({T} tasks)"
    )
    _render_html(
        range(L), method_stats, cos_by_method, task_list, out_html, title,
        strong=args.strong_threshold, partial=args.partial_threshold,
    )
    print(f"HTML report saved: {out_html}")


if __name__ == "__main__":
    main()
