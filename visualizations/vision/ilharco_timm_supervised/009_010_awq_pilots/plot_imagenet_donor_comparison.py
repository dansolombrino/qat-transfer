"""Export the ImageNet-donor AWQ pilot comparison for reviewer 3HFP.

The script is render-only: it reads the completed 009/010 evaluation JSONs and
the receiver-specific FP/AWQ baselines, then exports per-target data, aggregate
statistics, and reviewer-facing figures.  Receiver target is the only
aggregated run-id dimension.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qat-transfer-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_ROOT = (
    PROJECT_ROOT / "evaluations" / "vision" / "ilharco_timm_supervised"
)
DEFAULT_PLOT_ROOT = PROJECT_ROOT / "plots"
EXPERIMENT = Path("vision/ilharco_timm_supervised/009_010_awq_pilots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--sseed", required=True, type=int)
    parser.add_argument("--tseed", required=True, type=int)
    parser.add_argument("--optim", required=True)
    parser.add_argument("--qat", required=True)
    parser.add_argument("--awq", required=True)
    parser.add_argument("--qv", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def normalized_model_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def require_close(actual: float, expected: float, label: str, path: Path) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch in {path}: {actual} != {expected}")


def load_transfer_results(
    root: Path,
    *,
    model: str,
    source: str,
    source_seed: int,
    target_seed: int,
    alpha: float,
    split: str,
    metric_keys: Iterable[str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    paths = sorted(root.rglob("eval_results.json"))
    if not paths:
        raise FileNotFoundError(f"no transfer results found under {root}")
    for path in paths:
        data = load_json(path)
        if normalized_model_name(str(data["model_name"])) != model:
            continue
        if data["source"]["dataset_name"] != source:
            continue
        if int(data["source"]["seed"]) != source_seed:
            continue
        if int(data["target"]["seed"]) != target_seed:
            continue
        if data["eval_split"] != split:
            continue
        require_close(data["qv"]["alpha"], alpha, "alpha", path)
        target = str(data["target"]["dataset_name"])
        if target in records:
            raise ValueError(f"duplicate transfer result for {target}: {path}")
        for key in metric_keys:
            if not isinstance(data.get(key), (int, float)):
                raise ValueError(f"missing numeric {key} in {path}")
        records[target] = data
    if len(records) != 22:
        raise ValueError(f"expected 22 targets under {root}, found {len(records)}")
    return records


def legacy_optim_fragment(example: dict[str, Any]) -> str:
    return (
        f"optim=adamw_lr={example['lr']}_wd={example['wd']}_ls={example['ls']}"
        f"_wl={example['wl']}_mgn={example['max_grad_norm']}_bs={example['batch_size']}"
    )


def legacy_awq_fragment(example: dict[str, Any]) -> str:
    awq = example["awq"]
    skipped = "-".join(sorted(awq["skip_modules"])) or "none"
    return (
        f"awq=bits={awq['bits']}_gran={awq['granularity']}_skip={skipped}"
        f"_ncal={awq['num_calib_batches']}_ngrid={awq['n_grid']}_clip={awq['clip']}"
    )


def load_baselines(
    targets: Iterable[str],
    *,
    model: str,
    seed: int,
    example: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    fp: dict[str, float] = {}
    awq: dict[str, float] = {}
    optim_fragment = legacy_optim_fragment(example)
    awq_fragment = legacy_awq_fragment(example)
    baseline_root = EVALUATION_ROOT / "000_baselines" / "vision"
    for target in targets:
        fp_path = (
            baseline_root
            / "fp"
            / model
            / target
            / optim_fragment
            / f"seed={seed}"
            / "eval_results.json"
        )
        awq_path = (
            baseline_root
            / "fp_awq"
            / model
            / target
            / optim_fragment
            / awq_fragment
            / f"seed={seed}"
            / "eval_results.json"
        )
        fp[target] = float(load_json(fp_path)["test_accuracy"])
        awq[target] = float(load_json(awq_path)["test_accuracy"])
    return fp, awq


def output_directory(args: argparse.Namespace) -> Path:
    return (
        args.plot_root
        / EXPERIMENT
        / Path(__file__).stem
        / f"model={args.model}"
        / f"src={args.src}"
        / f"sseed={args.sseed}"
        / f"tseed={args.tseed}"
        / f"optim={args.optim}"
        / f"qat={args.qat}"
        / f"awq={args.awq}"
        / f"qv={args.qv}"
        / f"alpha={args.alpha}"
        / f"split={args.split}"
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def export_figure(
    interactive: go.Figure,
    static: Any,
    output_dir: Path,
    basename: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        path = output_dir / f"{basename}.{suffix}"
        temporary = output_dir / f".{basename}.{suffix}.tmp"
        static.savefig(temporary, format=suffix, bbox_inches="tight", **kwargs)
        temporary.replace(path)
        outputs.append(path)
    plt.close(static)
    html_path = output_dir / f"{basename}.html"
    html_tmp = output_dir / f".{basename}.html.tmp"
    interactive.write_html(html_tmp, include_plotlyjs=True, full_html=True)
    html_tmp.replace(html_path)
    outputs.append(html_path)
    return outputs


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "wins": sum(value > 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta_keys = (
        "qv_awq_fp_head_delta_vs_awq",
        "qv_awq_qat_head_delta_vs_awq",
        "awq_vector_delta_vs_awq",
        "awq_vector_delta_vs_fp",
    )
    subsets = {
        "all_targets": rows,
        "cross_task_only": [row for row in rows if not row["is_donor"]],
    }
    return {
        subset: {
            key: aggregate([float(row[key]) for row in selected])
            for key in delta_keys
        }
        for subset, selected in subsets.items()
    }


def accuracy_heatmap(rows: list[dict[str, Any]], title: str) -> go.Figure:
    columns = (
        ("fp_accuracy", "FP"),
        ("awq_accuracy", "AWQ(FP)"),
        ("qv_awq_fp_head_accuracy", "QV→AWQ\nFP head"),
        ("qv_awq_qat_head_accuracy", "QV→AWQ\nQAT head"),
        ("awq_vector_accuracy", "AWQ-vector"),
    )
    z = [[float(row[key]) for key, _ in columns] for row in rows]
    text = [[f"{value:.3f}" for value in row] for row in z]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=[label for _, label in columns],
            y=[row["target"] for row in rows],
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "Accuracy"},
            hovertemplate="target=%{y}<br>method=%{x}<br>accuracy=%{z:.5f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Raw accuracy — {title}",
        template="plotly_white",
        width=1050,
        height=1050,
        margin={"l": 135, "r": 100, "t": 110, "b": 100},
    )
    figure.update_yaxes(autorange="reversed", title="Receiver dataset")
    return figure


def static_accuracy_heatmap(rows: list[dict[str, Any]], title: str) -> Any:
    columns = (
        ("fp_accuracy", "FP"),
        ("awq_accuracy", "AWQ(FP)"),
        ("qv_awq_fp_head_accuracy", "QV→AWQ\nFP head"),
        ("qv_awq_qat_head_accuracy", "QV→AWQ\nQAT head"),
        ("awq_vector_accuracy", "AWQ-vector"),
    )
    values = np.asarray(
        [[float(row[key]) for key, _ in columns] for row in rows], dtype=np.float64
    )
    figure, axis = plt.subplots(figsize=(9.2, 11.0))
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            color = "white" if value < 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=color,
            )
    axis.set_xticks(np.arange(len(columns)), labels=[label for _, label in columns])
    axis.set_yticks(np.arange(len(rows)), labels=[row["target"] for row in rows])
    axis.set_ylabel("Receiver dataset")
    axis.set_title(f"Raw accuracy\n{title}")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Accuracy")
    figure.tight_layout()
    return figure


def delta_heatmap(rows: list[dict[str, Any]], title: str) -> go.Figure:
    columns = (
        ("qv_awq_fp_head_delta_vs_awq", "QV→AWQ FP head\n− AWQ"),
        ("qv_awq_qat_head_delta_vs_awq", "QV→AWQ QAT head\n− AWQ"),
        ("awq_vector_delta_vs_awq", "AWQ-vector\n− AWQ"),
        ("awq_vector_delta_vs_fp", "AWQ-vector\n− FP"),
    )
    z = [[100.0 * float(row[key]) for key, _ in columns] for row in rows]
    limit = max(abs(value) for row in z for value in row)
    text = [[f"{value:+.2f}" for value in row] for row in z]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=[label for _, label in columns],
            y=[row["target"] for row in rows],
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=-limit,
            zmax=limit,
            zmid=0.0,
            colorbar={"title": "Δ accuracy (pp)"},
            hovertemplate="target=%{y}<br>comparison=%{x}<br>delta=%{z:+.4f} pp<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Accuracy deltas — {title}",
        template="plotly_white",
        width=1050,
        height=1050,
        margin={"l": 135, "r": 120, "t": 110, "b": 120},
    )
    figure.update_yaxes(autorange="reversed", title="Receiver dataset")
    return figure


def static_delta_heatmap(rows: list[dict[str, Any]], title: str) -> Any:
    columns = (
        ("qv_awq_fp_head_delta_vs_awq", "QV→AWQ FP head\n− AWQ"),
        ("qv_awq_qat_head_delta_vs_awq", "QV→AWQ QAT head\n− AWQ"),
        ("awq_vector_delta_vs_awq", "AWQ-vector\n− AWQ"),
        ("awq_vector_delta_vs_fp", "AWQ-vector\n− FP"),
    )
    values = 100.0 * np.asarray(
        [[float(row[key]) for key, _ in columns] for row in rows], dtype=np.float64
    )
    limit = float(np.max(np.abs(values)))
    figure, axis = plt.subplots(figsize=(9.2, 11.0))
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:+.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
            )
    axis.set_xticks(np.arange(len(columns)), labels=[label for _, label in columns])
    axis.set_yticks(np.arange(len(rows)), labels=[row["target"] for row in rows])
    axis.set_ylabel("Receiver dataset")
    axis.set_title(f"Accuracy deltas\n{title}")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Δ accuracy (percentage points)")
    figure.tight_layout()
    return figure


def delta_bars(rows: list[dict[str, Any]], title: str) -> go.Figure:
    figure = go.Figure()
    for key, label, color in (
        ("qv_awq_fp_head_delta_vs_awq", "QV→AWQ (FP head) − AWQ", "#4c78a8"),
        ("qv_awq_qat_head_delta_vs_awq", "QV→AWQ (QAT head) − AWQ", "#f58518"),
        ("awq_vector_delta_vs_awq", "AWQ-vector − AWQ", "#54a24b"),
    ):
        figure.add_trace(
            go.Bar(
                name=label,
                orientation="h",
                y=[row["target"] for row in rows],
                x=[100.0 * float(row[key]) for row in rows],
                marker_color=color,
                hovertemplate="target=%{y}<br>delta=%{x:+.4f} pp<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        title=f"Transfer deltas against receiver AWQ — {title}",
        template="plotly_white",
        barmode="group",
        width=1200,
        height=1150,
        margin={"l": 135, "r": 80, "t": 110, "b": 90},
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
    )
    figure.add_vline(x=0.0, line_width=1.2, line_color="black")
    figure.update_xaxes(title="Accuracy delta versus AWQ(FP) (percentage points)")
    figure.update_yaxes(autorange="reversed", title="Receiver dataset")
    return figure


def static_delta_bars(rows: list[dict[str, Any]], title: str) -> Any:
    series = (
        ("qv_awq_fp_head_delta_vs_awq", "QV→AWQ (FP head) − AWQ", "#4c78a8"),
        ("qv_awq_qat_head_delta_vs_awq", "QV→AWQ (QAT head) − AWQ", "#f58518"),
        ("awq_vector_delta_vs_awq", "AWQ-vector − AWQ", "#54a24b"),
    )
    positions = np.arange(len(rows), dtype=np.float64)
    height = 0.24
    figure, axis = plt.subplots(figsize=(11.5, 12.0))
    for index, (key, label, color) in enumerate(series):
        offset = (index - 1) * height
        axis.barh(
            positions + offset,
            [100.0 * float(row[key]) for row in rows],
            height=height,
            label=label,
            color=color,
        )
    axis.axvline(0.0, linewidth=1.0, color="black")
    axis.set_yticks(positions, labels=[row["target"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Accuracy delta versus AWQ(FP) (percentage points)")
    axis.set_ylabel("Receiver dataset")
    axis.set_title(f"Transfer deltas against receiver AWQ\n{title}")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8)
    figure.tight_layout()
    return figure


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def markdown_summary(metadata: dict[str, Any], aggregates: dict[str, Any]) -> str:
    lines = [
        "# ImageNet-donor AWQ pilot summary",
        "",
        f"Model: `{metadata['model']}`; donor: `{metadata['src']}`; "
        f"lambda/alpha: `{metadata['alpha']}`; split: `{metadata['split']}`.",
        "",
        "All deltas below are accuracy percentage points. The reviewer-relevant "
        "summary excludes the ImageNet→ImageNet self-pair.",
        "",
        "| comparison | n | mean | median | min | max | wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "qv_awq_fp_head_delta_vs_awq": "QV→AWQ (FP head) − AWQ",
        "qv_awq_qat_head_delta_vs_awq": "QV→AWQ (QAT head) − AWQ",
        "awq_vector_delta_vs_awq": "AWQ-vector − AWQ",
        "awq_vector_delta_vs_fp": "AWQ-vector − FP",
    }
    for key, label in labels.items():
        values = aggregates["cross_task_only"][key]
        lines.append(
            f"| {label} | {values['n']} | {100 * values['mean']:+.3f} | "
            f"{100 * values['median']:+.3f} | {100 * values['min']:+.3f} | "
            f"{100 * values['max']:+.3f} | {values['wins']}/{values['n']} |"
        )
    return "\n".join(lines) + "\n"


def write_index(output_dir: Path, metadata: dict[str, Any]) -> None:
    figures = ("accuracy_heatmap", "delta_heatmap", "delta_bar")
    links = "\n".join(
        f'<li><a href="{name}.html">{html.escape(name)}</a> '
        f'(<a href="{name}.png">PNG</a>, <a href="{name}.pdf">PDF</a>, '
        f'<a href="{name}.svg">SVG</a>)</li>'
        for name in figures
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AWQ pilot exports</title></head>
<body><h1>ImageNet-donor AWQ pilot exports</h1>
<p>Model: <code>{html.escape(str(metadata['model']))}</code>; donor:
<code>{html.escape(str(metadata['src']))}</code>; alpha/lambda:
<code>{metadata['alpha']}</code>; split: <code>{metadata['split']}</code>.</p>
<ul>{links}
<li><a href="summary.csv">Per-target CSV</a></li>
<li><a href="summary.json">Full JSON with aggregates</a></li>
<li><a href="summary.md">Markdown aggregate summary</a></li></ul>
</body></html>
"""
    atomic_text(output_dir / "index.html", document)


def main() -> None:
    args = parse_args()
    results_009 = load_transfer_results(
        EVALUATION_ROOT / "009_qat_transfer_awq",
        model=args.model,
        source=args.src,
        source_seed=args.sseed,
        target_seed=args.tseed,
        alpha=args.alpha,
        split=args.split,
        metric_keys=(
            f"{args.split}_accuracy_fp_head_awq",
            f"{args.split}_accuracy_qat_head_awq",
        ),
    )
    results_010 = load_transfer_results(
        EVALUATION_ROOT / "010_awq_transfer" / "qv_transfer_awqv",
        model=args.model,
        source=args.src,
        source_seed=args.sseed,
        target_seed=args.tseed,
        alpha=args.alpha,
        split=args.split,
        metric_keys=(f"{args.split}_accuracy_fp_head",),
    )
    if set(results_009) != set(results_010):
        raise ValueError("009 and 010 target sets differ")

    targets = sorted(results_009, key=str.casefold)
    example = results_009[targets[0]]
    fp, awq = load_baselines(
        targets,
        model=args.model,
        seed=args.tseed,
        example=example,
    )

    fp_head_key = f"{args.split}_accuracy_fp_head_awq"
    qat_head_key = f"{args.split}_accuracy_qat_head_awq"
    vector_key = f"{args.split}_accuracy_fp_head"
    rows: list[dict[str, Any]] = []
    for target in targets:
        qv_fp = float(results_009[target][fp_head_key])
        qv_qat = float(results_009[target][qat_head_key])
        vector = float(results_010[target][vector_key])
        rows.append(
            {
                "target": target,
                "is_donor": target == args.src,
                "fp_accuracy": fp[target],
                "awq_accuracy": awq[target],
                "qv_awq_fp_head_accuracy": qv_fp,
                "qv_awq_fp_head_delta_vs_awq": qv_fp - awq[target],
                "qv_awq_qat_head_accuracy": qv_qat,
                "qv_awq_qat_head_delta_vs_awq": qv_qat - awq[target],
                "awq_vector_accuracy": vector,
                "awq_vector_delta_vs_awq": vector - awq[target],
                "awq_vector_delta_vs_fp": vector - fp[target],
            }
        )

    metadata = {
        "schema_version": "awq_pilot_comparison_v1",
        "model": args.model,
        "src": args.src,
        "sseed": args.sseed,
        "tseed": args.tseed,
        "optim": args.optim,
        "qat": args.qat,
        "awq": args.awq,
        "qv": args.qv,
        "alpha": args.alpha,
        "split": args.split,
        "target_order": targets,
        "units": {"accuracy": "fraction", "figure_deltas": "percentage_points"},
    }
    aggregates = aggregate_rows(rows)
    payload = {"metadata": metadata, "aggregates": aggregates, "rows": rows}
    output_dir = output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    atomic_text(output_dir / "summary.json", json.dumps(payload, indent=2) + "\n")
    atomic_text(output_dir / "summary.md", markdown_summary(metadata, aggregates))

    title = f"{args.src} donor, α/λ={args.alpha}, {args.model}, {args.split}"
    exported: list[Path] = []
    exported += export_figure(
        accuracy_heatmap(rows, title),
        static_accuracy_heatmap(rows, title),
        output_dir,
        "accuracy_heatmap",
    )
    exported += export_figure(
        delta_heatmap(rows, title),
        static_delta_heatmap(rows, title),
        output_dir,
        "delta_heatmap",
    )
    exported += export_figure(
        delta_bars(rows, title),
        static_delta_bars(rows, title),
        output_dir,
        "delta_bar",
    )
    write_index(output_dir, metadata)

    print(output_dir)
    for path in (
        output_dir / "index.html",
        output_dir / "summary.csv",
        output_dir / "summary.json",
        output_dir / "summary.md",
        *exported,
    ):
        print(path)


if __name__ == "__main__":
    main()
