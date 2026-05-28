"""Combine per-task steering vectors into a single cross-task universal vector.

Reads the `steering_vectors.pt` files produced by 002's `fit_steering_vector.py`
for a list of source tasks, sign-aligns them (or computes the top-SVD direction
across them), and saves a universal vector at the same .pt schema so 002's
evaluation script can load it.

Argparse, no Hydra (the input space is naturally a flat set of file paths).
No GPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[4]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch

from src.vision.utils import sanitize_timm_model_name


METHODS_INPUT = ("mean_diff", "contrastive_svd")
COMBINERS = ("sign_align_average", "top_svd")


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
        "--exclude", nargs="*", default=[],
        help="Task names to exclude from the source pool (for LOO this is the held-out target).",
    )
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help="Explicit source-task list. If omitted, auto-discover all tasks with a fitted vector "
             "at this PTQ config, minus --exclude.",
    )
    p.add_argument("--min-bad", type=int, default=20)
    p.add_argument(
        "--combiner", default="sign_align_average", choices=COMBINERS,
        help="sign_align_average: pick a reference task per block, flip signs of others to align, "
             "then average and unit-normalize. top_svd: stack unit vectors as rows of a (T, D) "
             "matrix per block and take the top right-singular-vector.",
    )
    p.add_argument(
        "--sign-reference", default=None,
        help="Task name to use as sign reference for sign_align_average. Defaults to the first "
             "task alphabetically in the source pool.",
    )
    p.add_argument(
        "--out-path", default=None,
        help="Absolute path where the universal_steering_vectors.pt is saved. If omitted, derived "
             "from cfg under CHECKPOINT_BASE_PATH/.../universal_steering_vectors/.",
    )
    return p.parse_args()


def _per_task_dir(checkpoint_base: Path, sanitized: str, dataset: str, optim_tag: str, ptq_tag: str, seed_tag: str) -> Path:
    return (
        checkpoint_base / "vision" / "ilharco_timm_supervised" / "steering_vectors"
        / sanitized / dataset / optim_tag / ptq_tag / seed_tag
    )


def _build_paths(args):
    checkpoint_base = Path(os.environ["CHECKPOINT_BASE_PATH"])
    sanitized = sanitize_timm_model_name(args.model_name)
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    optim_tag = (
        f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
        f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
    )
    ptq_tag = f"ptq=bits={args.bits}_gran={args.granularity}_skip={skip_tag}"
    seed_tag = f"seed={args.seed}"
    return checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag


def _discover_tasks(checkpoint_base: Path, sanitized: str, optim_tag: str, ptq_tag: str, seed_tag: str) -> list[str]:
    base = (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "steering_vectors" / sanitized
    )
    if not base.exists():
        return []
    out = []
    for ds_dir in sorted(base.iterdir()):
        if not ds_dir.is_dir():
            continue
        if (ds_dir / optim_tag / ptq_tag / seed_tag / "steering_vectors.pt").exists():
            out.append(ds_dir.name)
    return out


def _load_source_vectors(args, datasets, paths):
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = paths
    loaded: dict[str, dict[str, np.ndarray]] = {}
    skipped: list[tuple[str, str]] = []
    for ds in datasets:
        d = _per_task_dir(checkpoint_base, sanitized, ds, optim_tag, ptq_tag, seed_tag)
        vec_path = d / "steering_vectors.pt"
        meta_path = d / "fit_metadata.json"
        if not vec_path.exists():
            skipped.append((ds, "no steering_vectors.pt"))
            continue
        payload = torch.load(vec_path, map_location="cpu", weights_only=True)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        num_bad = int(meta.get("num_bad", payload.get("num_bad", -1)))
        if 0 <= num_bad < args.min_bad:
            skipped.append((ds, f"num_bad={num_bad} < {args.min_bad}"))
            continue
        loaded[ds] = {m: payload[m].numpy() for m in METHODS_INPUT if m in payload}
    return loaded, skipped


def _sign_align_average(V: np.ndarray, ref_idx: int) -> np.ndarray:
    """V shape (T, L, D), returns (L, D). Reference is V[ref_idx]; each task is
    flipped per-block so its dot with the reference is non-negative. Returns
    the unit-normalized per-block mean."""
    L, D = V.shape[1], V.shape[2]
    out = np.zeros((L, D), dtype=np.float64)
    for l in range(L):
        ref = V[ref_idx, l]
        ref_norm = ref / (np.linalg.norm(ref) + 1e-12)
        block_stack = V[:, l, :]
        norms = np.linalg.norm(block_stack, axis=1, keepdims=True) + 1e-12
        unit = block_stack / norms
        dots = unit @ ref_norm
        signs = np.where(dots >= 0, 1.0, -1.0)
        aligned = unit * signs[:, None]
        mean_v = aligned.mean(axis=0)
        out[l] = mean_v / (np.linalg.norm(mean_v) + 1e-12)
    return out


def _top_svd(V: np.ndarray) -> np.ndarray:
    """V shape (T, L, D), returns (L, D). Per block, stack tasks' unit vectors
    as rows and return the top right-singular-vector. Sign is then aligned to
    the per-block mean of the unit vectors so the direction is reproducible."""
    L, D = V.shape[1], V.shape[2]
    out = np.zeros((L, D), dtype=np.float64)
    for l in range(L):
        block_stack = V[:, l, :]
        norms = np.linalg.norm(block_stack, axis=1, keepdims=True) + 1e-12
        unit = block_stack / norms
        # Stable top right-singular-vector via SVD of (T, D) — economy SVD.
        _, _, vh = np.linalg.svd(unit, full_matrices=False)
        v = vh[0]
        # Sign-align to mean of unit vectors (any nonzero reference works).
        mean_ref = unit.mean(axis=0)
        if np.dot(v, mean_ref) < 0:
            v = -v
        out[l] = v / (np.linalg.norm(v) + 1e-12)
    return out


def _combine(args, loaded: dict[str, dict[str, np.ndarray]]):
    task_list = sorted(loaded.keys())
    T = len(task_list)
    sample = next(iter(loaded.values()))
    methods_present = [m for m in METHODS_INPUT if m in sample]
    out: dict[str, torch.Tensor] = {}
    per_method_meta = {}
    for method in methods_present:
        # Stack tasks: (T, L, D)
        V = np.stack([loaded[t][method] for t in task_list], axis=0)
        if args.combiner == "sign_align_average":
            ref_name = args.sign_reference or task_list[0]
            if ref_name not in task_list:
                raise SystemExit(
                    f"--sign-reference={ref_name!r} not in source pool "
                    f"(available: {task_list})"
                )
            ref_idx = task_list.index(ref_name)
            universal = _sign_align_average(V, ref_idx)
            per_method_meta[method] = {"sign_reference": ref_name, "ref_idx": ref_idx}
        elif args.combiner == "top_svd":
            universal = _top_svd(V)
            per_method_meta[method] = {"combiner": "top_svd"}
        else:
            raise SystemExit(f"unknown combiner {args.combiner!r}")
        out[method] = torch.from_numpy(universal).float()
    return out, task_list, methods_present, per_method_meta


def _exclude_tag(excluded: list[str]) -> str:
    return "-".join(sorted(excluded)) if excluded else "none"


def main() -> None:
    args = parse_args()
    paths = _build_paths(args)
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = paths

    if args.datasets is None:
        discovered = _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
        datasets = [d for d in discovered if d not in set(args.exclude)]
    else:
        datasets = [d for d in args.datasets if d not in set(args.exclude)]

    if not datasets:
        print(f"No source tasks available after applying --exclude={args.exclude}", file=sys.stderr)
        sys.exit(1)
    print(f"Combiner: {args.combiner}   excluded: {sorted(args.exclude) or 'none'}")
    print(f"Source pool ({len(datasets)} candidates): {datasets}\n")

    loaded, skipped = _load_source_vectors(args, datasets, paths)
    print(f"Loaded {len(loaded)} task(s); skipped {len(skipped)}:")
    for ds, reason in skipped:
        print(f"  - {ds}: {reason}")
    if len(loaded) < 2:
        print("Need at least 2 source tasks to combine.", file=sys.stderr)
        sys.exit(1)

    universal, task_list, methods_present, meta = _combine(args, loaded)
    L, D = universal[methods_present[0]].shape
    print(f"\nCombined universal vectors: {len(methods_present)} methods × {L} blocks × {D}-D")
    print(f"  Source tasks used ({len(task_list)}): {task_list}")

    # Per-block sanity: mean |cos| of each source task's vector with the universal.
    print(f"\nSanity — mean |cos| of source tasks to the universal vector, per block:")
    for method in methods_present:
        univ = universal[method].numpy()  # (L, D)
        print(f"  [{method}]")
        for l in range(L):
            tvecs = np.stack([loaded[t][method][l] for t in task_list], axis=0)
            tnorms = np.linalg.norm(tvecs, axis=1, keepdims=True) + 1e-12
            tunit = tvecs / tnorms
            unorm = np.linalg.norm(univ[l]) + 1e-12
            uunit = univ[l] / unorm
            cosines = tunit @ uunit
            mean_abs = float(np.abs(cosines).mean())
            print(f"    block {l:>2}:  mean |cos| = {mean_abs:.3f}  (min {np.abs(cosines).min():.3f}, max {np.abs(cosines).max():.3f})")

    # Output path
    if args.out_path is not None:
        out_path = Path(args.out_path)
    else:
        excl_tag = _exclude_tag(args.exclude)
        out_dir = (
            checkpoint_base / "vision" / "ilharco_timm_supervised" / "universal_steering_vectors"
            / sanitized / optim_tag / ptq_tag / seed_tag
            / f"combiner={args.combiner}_exclude={excl_tag}"
        )
        out_path = out_dir / "universal_steering_vectors.pt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_blocks": L,
        "embed_dim": D,
        # Carry through the source-task counts so the downstream eval JSON has
        # context. num_good/num_bad/num_total are aggregated from sources.
        "num_good": -1,
        "num_bad": -1,
        "num_fp_correct": -1,
        "num_total": -1,
    }
    payload.update(universal)
    torch.save(payload, out_path)

    metadata_path = out_path.with_name("combine_metadata.json")
    metadata = {
        "model_name": args.model_name,
        "lr": args.lr, "wd": args.wd, "ls": args.ls, "wl": args.wl,
        "max_grad_norm": args.max_grad_norm, "batch_size": args.batch_size,
        "seed": args.seed,
        "ptq_bits": args.bits, "ptq_granularity": args.granularity,
        "ptq_skip_modules": args.skip_modules,
        "combiner": args.combiner,
        "source_tasks": task_list,
        "excluded": sorted(args.exclude),
        "min_bad": args.min_bad,
        "methods": methods_present,
        "per_method_meta": meta,
        "vectors_path": str(out_path),
        "num_blocks": L,
        "embed_dim": D,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nUniversal vectors saved:  {out_path}")
    print(f"Metadata saved:           {metadata_path}")


if __name__ == "__main__":
    main()
