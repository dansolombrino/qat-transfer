"""Leave-one-out orchestrator for cross-task quant-steering transfer.

For each candidate target task, this script:
  1. Resolves where its universal-vector .pt would live (excluding the target
     from the source pool).
  2. Resolves where its transfer-eval JSON would live.
  3. Either emits the shell command sequence to produce them (`--output commands`)
     or reads existing JSONs and summarizes the per-target transfer gain
     (`--output summary`).

Argparse, no Hydra. Does not launch GPU work itself — emits commands you can
pipe to `bash`, send to a Slurm queue, or run in the background. Mirrors the
`pick_best_alpha.py` convention from 001_qat_transfer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

_CODE_DIR = Path(__file__).resolve().parents[4]
_PROJECT_ROOT = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from dotenv import load_dotenv
load_dotenv()

from src.vision.utils import sanitize_timm_model_name


COMBINER_SCRIPT = "code/experiments/vision/ilharco_timm_supervised/003_quant_steering_transfer/combine_steering_vectors.py"
EVAL_SCRIPT = "code/experiments/vision/ilharco_timm_supervised/003_quant_steering_transfer/evaluate_steered_ptq_transfer.py"


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
    p.add_argument("--combiner", default="sign_align_average",
                   choices=["sign_align_average", "top_svd"])
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Target tasks to evaluate. If omitted, auto-discover from steering_vectors/.")
    p.add_argument("--methods", nargs="+", default=["mean_diff"],
                   help="Subset of steering methods to evaluate at transfer time.")
    p.add_argument("--block-sweep", default="all",
                   help='"all" or comma-separated list of block indices, e.g. "6" or "5,6,7".')
    p.add_argument("--alpha-grid", default="-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0",
                   help="Comma-separated floats for the alpha sweep.")
    p.add_argument("--gpu", default="0")
    p.add_argument("--output", choices=["commands", "commands-bg", "summary"], default="commands",
                   help="commands: serial shell pipeline. commands-bg: each task in background. "
                        "summary: read existing eval_results.json and tabulate.")
    return p.parse_args()


def _build_paths(args):
    checkpoint_base = Path(os.environ["CHECKPOINT_BASE_PATH"])
    evaluation_base = Path(os.environ["EVALUATION_BASE_PATH"])
    sanitized = sanitize_timm_model_name(args.model_name)
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    optim_tag = (
        f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
        f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
    )
    ptq_tag = f"ptq=bits={args.bits}_gran={args.granularity}_skip={skip_tag}"
    seed_tag = f"seed={args.seed}"
    return checkpoint_base, evaluation_base, sanitized, optim_tag, ptq_tag, seed_tag


def _discover_tasks(checkpoint_base: Path, sanitized: str, optim_tag: str, ptq_tag: str, seed_tag: str) -> list[str]:
    base = (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "steering_vectors" / sanitized
    )
    if not base.exists():
        return []
    out = []
    for ds_dir in sorted(base.iterdir()):
        if ds_dir.is_dir() and (ds_dir / optim_tag / ptq_tag / seed_tag / "steering_vectors.pt").exists():
            out.append(ds_dir.name)
    return out


def _universal_vector_path(checkpoint_base: Path, sanitized: str, optim_tag: str, ptq_tag: str, seed_tag: str, combiner: str, excluded: list[str]) -> Path:
    excl_tag = "-".join(sorted(excluded)) if excluded else "none"
    return (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "universal_steering_vectors" / sanitized / optim_tag / ptq_tag / seed_tag
        / f"combiner={combiner}_exclude={excl_tag}" / "universal_steering_vectors.pt"
    )


def _eval_results_path(evaluation_base: Path, sanitized: str, target: str, optim_tag: str, ptq_tag: str, combiner: str, excluded: list[str], seed_tag: str) -> Path:
    excl_tag = "-".join(sorted(excluded)) if excluded else "none"
    return (
        evaluation_base / "vision" / "ilharco_timm_supervised"
        / "003_quant_steering_transfer" / "vision" / "fp_ptq_steered_transfer"
        / sanitized / target / optim_tag / ptq_tag
        / f"combiner={combiner}_exclude={excl_tag}" / seed_tag
        / "eval_results.json"
    )


def _emit_commands(args, tasks: list[str], paths, background: bool):
    checkpoint_base, evaluation_base, sanitized, optim_tag, ptq_tag, seed_tag = paths
    skip_str = " ".join(args.skip_modules)
    methods_str = "[" + ",".join(args.methods) + "]"
    alpha_str = "[" + args.alpha_grid + "]"
    block_sweep = args.block_sweep
    if block_sweep != "all":
        block_sweep = "[" + block_sweep + "]"

    print(dedent(f"""\
        #!/usr/bin/env bash
        # LOO transfer sweep — {len(tasks)} target task(s)
        #   model      : {args.model_name}
        #   ptq        : W{args.bits} {args.granularity} skip={args.skip_modules}
        #   combiner   : {args.combiner}
        #   methods    : {args.methods}
        #   block_sweep: {args.block_sweep}
        #   alpha_grid : {args.alpha_grid}
        set -euo pipefail
        cd "$(dirname "$(readlink -f "${{BASH_SOURCE[0]:-$0}}")")/{os.path.relpath('.', _PROJECT_ROOT)}" 2>/dev/null || cd "{_PROJECT_ROOT}"
    """))

    for t in tasks:
        vec_path = _universal_vector_path(
            checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag,
            args.combiner, [t],
        )
        cmd = dedent(f"""\

            # ====== target: {t} ======
            uv run --active python {COMBINER_SCRIPT} \\
                --model-name {args.model_name} \\
                --lr {args.lr} --wd {args.wd} --ls {args.ls} --wl {args.wl} \\
                --max-grad-norm {args.max_grad_norm} --batch-size {args.batch_size} --seed {args.seed} \\
                --bits {args.bits} --granularity {args.granularity} \\
                --skip-modules {skip_str} \\
                --combiner {args.combiner} \\
                --exclude {t} \\
                --out-path "{vec_path}" \\
              && uv run --active python {EVAL_SCRIPT} \\
                model_name={args.model_name} dataset_name={t} \\
                batch_size={args.batch_size} lr={args.lr} wd={args.wd} ls={args.ls} wl={args.wl} \\
                max_grad_norm={args.max_grad_norm} seed={args.seed} gpu={args.gpu} \\
                ptq.bits={args.bits} ptq.granularity={args.granularity} \\
                'ptq.skip_modules=[{",".join(args.skip_modules)}]' \\
                'steering.methods={methods_str}' \\
                steering.block_sweep={block_sweep} \\
                'steering.alpha_grid={alpha_str}' \\
                steering.vectors_path="{vec_path}"\
        """)
        if background:
            cmd = cmd + " &"
        print(cmd)

    if background:
        print("\nwait")
    print()


def _summary(args, tasks: list[str], paths):
    checkpoint_base, evaluation_base, sanitized, optim_tag, ptq_tag, seed_tag = paths

    rows = []
    for t in tasks:
        p = _eval_results_path(
            evaluation_base, sanitized, t, optim_tag, ptq_tag,
            args.combiner, [t], seed_tag,
        )
        if not p.exists():
            rows.append((t, None, None, None, None, None, None))
            continue
        r = json.loads(p.read_text())
        best = r["best_by_val"]
        rows.append((
            t,
            r["fp_test_accuracy"],
            r["plain_ptq_test_accuracy"],
            r["test_accuracy"],
            r["test_gain_over_plain_ptq"],
            best["method"],
            (best["block"], best["alpha"]),
        ))

    print(f"\n# LOO transfer summary — {args.model_name} | W{args.bits} {args.granularity} | combiner={args.combiner}\n")
    print(f"| target        | FP test | plain PTQ | best steered | Δ vs PTQ | method        | (block, α) |")
    print(f"|---------------|---------|-----------|--------------|----------|---------------|------------|")
    total_gain = 0.0
    n_done = 0
    n_pos = 0
    for t, fp, p_ptq, st, dlt, m, blka in rows:
        if fp is None:
            print(f"| {t:<13} |   ---   |    ---    |     ---      |   ---    |     ---       |    ---     |")
            continue
        n_done += 1
        if dlt > 0:
            n_pos += 1
        total_gain += dlt
        blk = blka[0] if blka else "-"
        alp = f"{blka[1]:+.2f}" if blka else "-"
        print(f"| {t:<13} | {fp * 100:6.2f}% | {p_ptq * 100:8.2f}% | {st * 100:11.2f}% | {dlt * 100:+7.2f}pp | {m:<13} | ({blk:>2}, {alp:>5}) |")

    if n_done:
        print(f"\nCompleted: {n_done}/{len(rows)} | mean Δ vs PTQ: {(total_gain / n_done) * 100:+.2f}pp | "
              f"transfer helped: {n_pos}/{n_done}\n")


def main() -> None:
    args = parse_args()
    paths = _build_paths(args)
    checkpoint_base, evaluation_base, sanitized, optim_tag, ptq_tag, seed_tag = paths

    tasks = args.tasks or _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
    if not tasks:
        print("No tasks discovered. Did you run 002 fit_steering_vector.py at this PTQ config?", file=sys.stderr)
        sys.exit(1)

    if args.output == "summary":
        _summary(args, tasks, paths)
    else:
        _emit_commands(args, tasks, paths, background=(args.output == "commands-bg"))


if __name__ == "__main__":
    main()
