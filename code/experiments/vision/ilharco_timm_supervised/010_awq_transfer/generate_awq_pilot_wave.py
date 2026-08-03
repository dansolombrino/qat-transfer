"""Generate the approved ImageNet-donor 009 + 010 wave for one GPU lane."""

import argparse
import shlex
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.run_id import hydra_override_arg, run_id_flat, run_id_path


DATASETS = [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397",
    "SVHN", "CIFAR10", "CIFAR100", "STL10", "Food101", "Flowers102",
    "FER2013", "PCAM", "OxfordIIITPet", "RenderedSST2", "EMNIST",
    "FashionMNIST", "KMNIST", "TinyImageNet", "ImageNet",
]
MODEL = "vit_base_patch16_224.orig_in21k"
MODEL_ID = "vit_base_patch16_224_orig_in21k"
SEED = 2038
OPTIM = "lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128"
QAT = "b3-gchannel-shead"
AWQ = "b3-gchannel-shead-n4-grid20-clip1"
WAVE_TOTAL = 45

RUN_ID_009 = ["model", "src", "tgt", "optim", "qat", "awq", "qv", "split"]
RUN_ID_010_MAT = ["model", "donor", "seed", "optim", "awq"]
RUN_ID_010 = [
    "model", "src", "tgt", "sseed", "tseed", "optim", "awq", "alpha", "split"
]


def identity_009(target: str) -> dict:
    return {
        "model": MODEL_ID,
        "src": f"ImageNet-seed{SEED}",
        "tgt": f"{target}-seed{SEED}",
        "optim": OPTIM,
        "qat": QAT,
        "awq": AWQ,
        "qv": "a1.0",
        "split": "test",
    }


def identity_materialize() -> dict:
    return {
        "model": MODEL_ID,
        "donor": "ImageNet",
        "seed": SEED,
        "optim": OPTIM,
        "awq": AWQ,
    }


def identity_010(target: str) -> dict:
    return {
        "model": MODEL_ID,
        "src": "ImageNet",
        "tgt": target,
        "sseed": SEED,
        "tseed": SEED,
        "optim": OPTIM,
        "awq": AWQ,
        "alpha": 1.0,
        "split": "test",
    }


def quoted_overrides(items: list[tuple[str, object]]) -> str:
    return " ".join(shlex.quote(hydra_override_arg(key, value)) for key, value in items)


def write_run(
    *,
    wave_id: str,
    rig: str,
    gpu: str,
    script_group: Path,
    flat: str,
    nested: Path,
    eval_group: Path,
    artifact: str,
    python_script: str,
    overrides: list[tuple[str, object]],
    purpose: str,
    this_run: str,
) -> None:
    wave_dir = PROJECT_ROOT / script_group / flat / f"wave_{wave_id}"
    wave_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = eval_group / nested
    log_dir = Path(str(script_group).replace("scripts/", "logs/", 1)) / flat / f"wave_{wave_id}"
    script_path = wave_dir / f"wave_{rig}_gpu{gpu}.sh"
    body = f'''#!/usr/bin/env bash
# run: {flat}   wave: {wave_id}   rig: {rig}   gpu: {gpu}
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || exit 1
cd "$PROJECT_ROOT" || exit 1

RUN_ID_FLAT={shlex.quote(flat)}
EVAL_DIR={shlex.quote(str(eval_dir))}
LOG_DIR={shlex.quote(str(log_dir))}
ARTIFACT={shlex.quote(artifact)}

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}
export WAVE_ID={shlex.quote(wave_id)}
export HYDRA_FULL_ERROR=1
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

HYDRA_ARGS=({quoted_overrides(overrides)})
.venv/bin/python {shlex.quote(python_script)} "${{HYDRA_ARGS[@]}}" 2>&1 \\
  | tee "$LOG_DIR/wave_{rig}_gpu{gpu}-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${{PIPESTATUS[@]}}")
python_rc=${{pipeline_rc[0]}}
tee_rc=${{pipeline_rc[1]}}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi
if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'EOF'
import datetime, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text()) if p.exists() else {{}}
s.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"))
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(s, sort_keys=True) + "\\n")
t.replace(p)
EOF
fi
exit "$rc"
'''
    script_path.write_text(body)
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    readme = wave_dir / "README.md"
    readme.write_text(
        f"# wave {wave_id} — {purpose}\n\n"
        f"Dispatched from rig-4090 to {rig} GPU {gpu}.\n\n"
        f"Why this wave: reviewer-3HFP AWQ pilots with ImageNet as the sole donor.\n\n"
        f"This run: `{this_run}` → {rig}, gpu {gpu}.\n\n"
        f"Full wave: {WAVE_TOTAL} runs on {rig}, gpu {gpu}, sequentially.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--rig", default="rig-3090-ti")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint-base", type=Path, required=True)
    parser.add_argument("--tracking-out", type=Path, required=True)
    args = parser.parse_args()

    common = [
        ("model_name", MODEL), ("batch_size", 128), ("lr", 1e-5),
        ("wd", 0.1), ("ls", 0.0), ("wl", 500), ("max_grad_norm", 1.0),
        ("gpu", 0),
    ]
    awq = [
        ("awq.bits", 3), ("awq.granularity", "channel"),
        ("awq.skip_modules", ["head"]), ("awq.num_calib_batches", 4),
        ("awq.n_grid", 20), ("awq.clip", True),
    ]

    mat = identity_materialize()
    mat_nested = run_id_path(mat, RUN_ID_010_MAT)
    write_run(
        wave_id=args.wave_id, rig=args.rig, gpu=args.gpu,
        script_group=Path("scripts/vision/ilharco_timm_supervised/010_awq_transfer/materialize_awq_checkpoint"),
        flat=run_id_flat(mat, RUN_ID_010_MAT), nested=mat_nested,
        eval_group=Path("evaluations/vision/ilharco_timm_supervised/010_awq_transfer/materialize_awq_checkpoint"),
        artifact=str(args.checkpoint_base / "vision" / "ilharco_timm_supervised" / "awq_transfer" / mat_nested / "classifier_epoch_1.pt"),
        python_script="code/experiments/vision/ilharco_timm_supervised/010_awq_transfer/materialize_awq_checkpoint.py",
        overrides=common + [("dataset_name", "ImageNet"), ("seed", SEED)] + awq,
        purpose="materialize the ImageNet AWQ donor displacement",
        this_run=run_id_flat(mat, RUN_ID_010_MAT),
    )

    rows = []
    for target in DATASETS:
        ident = identity_009(target)
        nested = run_id_path(ident, RUN_ID_009)
        write_run(
            wave_id=args.wave_id, rig=args.rig, gpu=args.gpu,
            script_group=Path("scripts/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq"),
            flat=run_id_flat(ident, RUN_ID_009), nested=nested,
            eval_group=Path("evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq"),
            artifact=str(Path("evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq") / nested / "eval_results.json"),
            python_script="code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq.py",
            overrides=common + [("eval_split", "test"), ("source.dataset_names", ["ImageNet"]),
                ("source.seed", SEED), ("target.dataset_names", [target]),
                ("target.seed", SEED), ("qat.bits", 3),
                ("qat.granularity", "channel"), ("qat.skip_modules", ["head"]),
                ("qv.alphas", [1.0])] + awq,
            purpose="test QAT-QV patching on top of AWQ at lambda=1",
            this_run=run_id_flat(ident, RUN_ID_009),
        )
        rows.append(("009", ident))

    rows.insert(0, ("010-materialize", mat))
    for target in DATASETS:
        ident = identity_010(target)
        nested = run_id_path(ident, RUN_ID_010)
        write_run(
            wave_id=args.wave_id, rig=args.rig, gpu=args.gpu,
            script_group=Path("scripts/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv"),
            flat=run_id_flat(ident, RUN_ID_010), nested=nested,
            eval_group=Path("evaluations/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv"),
            artifact=str(Path("evaluations/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv") / nested / "eval_results.json"),
            python_script="code/experiments/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv.py",
            overrides=common + [("eval_split", "test"), ("source.dataset_name", "ImageNet"),
                ("source.seed", SEED), ("target.dataset_name", target),
                ("target.seed", SEED), ("qv.alpha", 1.0)] + awq,
            purpose="test transfer of the ImageNet AWQ checkpoint displacement",
            this_run=run_id_flat(ident, RUN_ID_010),
        )
        rows.append(("010", ident))

    args.tracking_out.write_text("\n".join(f"{kind}\t{values}" for kind, values in rows) + "\n")
    print(f"generated {len(rows)} runs for wave {args.wave_id}")


if __name__ == "__main__":
    main()
