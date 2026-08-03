"""Generate the approved unit-alpha phase-009 full-grid completion wave.

The existing phase-009 artifacts contain the complete ImageNet-donor row. This
generator verifies that 22-cell baseline, materializes the remaining 21 x 22
cells as canonical self-guarded wave scripts, and emits EXPERIMENTS.md rows.
It never launches a GPU process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import stat
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.run_id import hydra_override_arg, run_id_flat, run_id_path


DATASETS = (
    "Cars",
    "CIFAR10",
    "CIFAR100",
    "DTD",
    "EMNIST",
    "EuroSAT",
    "FashionMNIST",
    "FER2013",
    "Flowers102",
    "Food101",
    "GTSRB",
    "ImageNet",
    "KMNIST",
    "MNIST",
    "OxfordIIITPet",
    "PCAM",
    "RenderedSST2",
    "RESISC45",
    "STL10",
    "SUN397",
    "SVHN",
    "TinyImageNet",
)
DONORS_3090 = frozenset(
    {
        "CIFAR100",
        "EuroSAT",
        "Flowers102",
        "KMNIST",
        "PCAM",
        "STL10",
        "TinyImageNet",
    }
)
DONORS_4090 = frozenset(set(DATASETS) - DONORS_3090 - {"ImageNet"})

MODEL = "vit_base_patch16_224.orig_in21k"
MODEL_ID = "vit_base_patch16_224_orig_in21k"
SEED = 2038
OPTIM = "lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128"
QAT = "b3-gchannel-shead"
AWQ = "b3-gchannel-shead-n4-grid20-clip1"
RUN_ID_PARAMS = ["model", "src", "tgt", "optim", "qat", "awq", "qv", "split"]

SCRIPT_GROUP = Path(
    "scripts/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq"
)
EVAL_GROUP = Path(
    "evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq"
)
PYTHON_SCRIPT = (
    "code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq/"
    "qv_transfer_awq.py"
)
EXPECTED_EXISTING = 22
EXPECTED_GENERATED = 462


def identity(source: str, target: str) -> dict[str, str]:
    return {
        "model": MODEL_ID,
        "src": f"{source}-seed{SEED}",
        "tgt": f"{target}-seed{SEED}",
        "optim": OPTIM,
        "qat": QAT,
        "awq": AWQ,
        "qv": "a1.0",
        "split": "test",
    }


def placement(source: str) -> tuple[str, str]:
    if source in DONORS_3090:
        return "rig-3090-ti", "0"
    if source in DONORS_4090:
        return "rig-4090", "0"
    raise ValueError(f"source is not assigned to a production lane: {source}")


def quoted_overrides(items: list[tuple[str, object]]) -> str:
    return " ".join(shlex.quote(hydra_override_arg(key, value)) for key, value in items)


def artifact_path(config: dict[str, str]) -> Path:
    return PROJECT_ROOT / EVAL_GROUP / run_id_path(config, RUN_ID_PARAMS) / "eval_results.json"


def validate_existing_grid() -> None:
    present: list[tuple[str, str]] = []
    missing_imagenet: list[str] = []
    unexpected_present: list[tuple[str, str]] = []
    for source in DATASETS:
        for target in DATASETS:
            exists = artifact_path(identity(source, target)).is_file()
            if exists:
                present.append((source, target))
                if source != "ImageNet":
                    unexpected_present.append((source, target))
            elif source == "ImageNet":
                missing_imagenet.append(target)

    if missing_imagenet:
        raise RuntimeError(f"existing ImageNet-donor row is incomplete: {missing_imagenet}")
    if unexpected_present:
        raise RuntimeError(
            "phase 009 already contains non-ImageNet unit-alpha test cells; "
            f"refusing the locked 462-cell wave: {unexpected_present[:10]}"
        )
    if len(present) != EXPECTED_EXISTING:
        raise RuntimeError(
            f"expected {EXPECTED_EXISTING} existing cells, found {len(present)}"
        )


def overrides(source: str, target: str) -> list[tuple[str, object]]:
    return [
        ("model_name", MODEL),
        ("batch_size", 128),
        ("eval_split", "test"),
        ("limit_num_batches", None),
        ("log_to_file", False),
        ("skip_existing", True),
        ("lr", 1e-5),
        ("wd", 0.1),
        ("ls", 0.0),
        ("wl", 500),
        ("max_grad_norm", 1.0),
        ("gpu", 0),
        ("source.dataset_names", [source]),
        ("source.seed", SEED),
        ("source.limit_num_epochs", None),
        ("target.dataset_names", [target]),
        ("target.seed", SEED),
        ("target.limit_num_epochs", None),
        ("qat.bits", 3),
        ("qat.granularity", "channel"),
        ("qat.skip_modules", ["head"]),
        ("qv.alphas", [1.0]),
        ("awq.skip_modules", ["head"]),
        ("awq.bits", 3),
        ("awq.granularity", "channel"),
        ("awq.num_calib_batches", 4),
        ("awq.n_grid", 20),
        ("awq.clip", True),
    ]


def wave_script(
    *, wave_id: str, rig: str, gpu: str, flat: str, nested: Path, args: str
) -> str:
    eval_dir = EVAL_GROUP / nested
    log_dir = Path(str(SCRIPT_GROUP).replace("scripts/", "logs/", 1)) / flat / f"wave_{wave_id}"
    artifact = eval_dir / "eval_results.json"
    return f'''#!/usr/bin/env bash
# run: {flat}   experiment: vision/ilharco_timm_supervised/009_qat_transfer_awq
# wave: {wave_id}   rig: {rig}   gpu: {gpu}
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || exit 1
cd "$PROJECT_ROOT" || exit 1

RUN_ID_FLAT={shlex.quote(flat)}
EVAL_DIR={shlex.quote(str(eval_dir))}
LOG_DIR={shlex.quote(str(log_dir))}
ARTIFACT={shlex.quote(str(artifact))}
export WAVE_ID={shlex.quote(wave_id)}

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

SOURCE_TAG="wave--$WAVE_ID"
SOURCE_REVISION=$(git rev-parse "refs/tags/$SOURCE_TAG^{{commit}}" 2>/dev/null) || {{
  echo "[source-drift] missing $SOURCE_TAG" >&2; exit 86;
}}
ACTUAL_REVISION=$(git rev-parse HEAD 2>/dev/null) || {{
  echo "[source-drift] project is not a Git working tree" >&2; exit 86;
}}
if [ "$ACTUAL_REVISION" != "$SOURCE_REVISION" ]; then
  echo "[source-drift] HEAD=$ACTUAL_REVISION expected=$SOURCE_REVISION ($SOURCE_TAG)" >&2
  exit 86
fi
SOURCE_PATHS=(code config scripts pyproject.toml uv.lock poetry.lock setup.cfg setup.py Pipfile Pipfile.lock requirements*.txt environment*.yml environment*.yaml Dockerfile*)
if ! git diff --quiet -- "${{SOURCE_PATHS[@]}}" || \
   ! git diff --cached --quiet -- "${{SOURCE_PATHS[@]}}" || \
   [ -n "$(git ls-files --others --exclude-standard -- "${{SOURCE_PATHS[@]}}")" ]; then
  echo "[source-drift] execution files differ from $SOURCE_REVISION" >&2
  exit 86
fi
export SOURCE_REVISION SOURCE_TAG

export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}
export HYDRA_FULL_ERROR=1
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

HYDRA_ARGS=({args})
.venv/bin/python {shlex.quote(PYTHON_SCRIPT)} "${{HYDRA_ARGS[@]}}" 2>&1 \
  | tee "$LOG_DIR/wave_{rig}_gpu{gpu}-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${{PIPESTATUS[@]}}")
python_rc=${{pipeline_rc[0]}}
tee_rc=${{pipeline_rc[1]}}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc; the required run log is incomplete" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi

if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'EOF'
import datetime, json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text()) if p.exists() else {{}}
s.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"),
         wave_id=os.environ.get("WAVE_ID"), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
         source_revision=os.environ.get("SOURCE_REVISION"),
         source_tag=os.environ.get("SOURCE_TAG"))
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(s, sort_keys=True) + "\\n")
t.replace(p)
EOF
fi
exit "$rc"
'''


def wave_readme(
    *, wave_id: str, prepared_at: str, rig: str, gpu: str, flat: str
) -> str:
    return (
        f"# wave {wave_id} — complete the phase-009 unit-alpha AWQ grid\n\n"
        f"Prepared {prepared_at} on rig-4090; production launch remains manual.\n\n"
        f"Source tag: `wave--{wave_id}` (the feature branch and annotated tag must "
        "resolve to the same commit on both isolated execution clones).\n\n"
        "Why this wave: add the 462 missing donor-receiver cells required for the "
        "full 22×22 `001_009` PTQ-versus-AWQ comparison. The existing 22-cell "
        "ImageNet-donor row is reused.\n\n"
        f"This run: `{flat}` → {rig}, gpu {gpu}.\n\n"
        "Full wave: 462 runs — 308 on rig-4090 gpu0 and 154 on "
        "rig-3090-ti gpu0, one sequential lane per rig.\n"
    )


def tracking_row(config: dict[str, str], wave_id: str, rig: str, gpu: str) -> str:
    values = [config[key] for key in RUN_ID_PARAMS]
    cells = values + [
        wave_id,
        rig,
        gpu,
        "todo",
        "",
        "",
        "",
        "",
        "",
        "prepared; manual launch",
    ]
    return "| " + " | ".join(str(value) for value in cells) + " |"


def insert_tracking_rows(path: Path, wave_id: str, rows: list[str]) -> None:
    text = path.read_text()
    if f"| {wave_id} |" in text:
        raise RuntimeError(f"EXPERIMENTS.md already contains wave {wave_id}")
    section = "## vision/ilharco_timm_supervised/009_qat_transfer_awq"
    start = text.index(section)
    next_section = text.index("\n## ", start + len(section))
    chunk = text[start:next_section]
    if chunk.count("| model | src | tgt | optim | qat | awq | qv | split |") != 1:
        raise RuntimeError("could not identify the unique phase-009 tracking table")
    updated_chunk = chunk.rstrip() + "\n" + "\n".join(rows) + "\n"
    path.write_text(text[:start] + updated_chunk + text[next_section:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-id", required=True)
    parser.add_argument(
        "--experiments-file", type=Path, default=PROJECT_ROOT / "EXPERIMENTS.md"
    )
    parser.add_argument(
        "--check", action="store_true", help="validate inputs and report the locked grid only"
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}-\d{6}", args.wave_id):
        raise ValueError("wave id must use YYYYMMDD-HHMMSS")
    if len(DONORS_4090) != 14 or len(DONORS_3090) != 7:
        raise RuntimeError("approved 14/7 donor assignment drifted")
    validate_existing_grid()

    cells = [
        (source, target)
        for source in DATASETS
        if source != "ImageNet"
        for target in DATASETS
    ]
    if len(cells) != EXPECTED_GENERATED:
        raise RuntimeError(f"expected {EXPECTED_GENERATED} generated cells, found {len(cells)}")

    counts = {"rig-4090": 0, "rig-3090-ti": 0}
    for source, _ in cells:
        counts[placement(source)[0]] += 1
    if counts != {"rig-4090": 308, "rig-3090-ti": 154}:
        raise RuntimeError(f"approved placement counts drifted: {counts}")

    print(
        f"validated {EXPECTED_EXISTING} existing cells and {len(cells)} missing cells; "
        f"assignment={counts}"
    )
    if args.check:
        return

    prepared_at = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    tracking_rows: list[str] = []
    for source, target in cells:
        config = identity(source, target)
        flat = run_id_flat(config, RUN_ID_PARAMS)
        nested = run_id_path(config, RUN_ID_PARAMS)
        rig, gpu = placement(source)
        wave_dir = PROJECT_ROOT / SCRIPT_GROUP / flat / f"wave_{args.wave_id}"
        if wave_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing wave directory: {wave_dir}")
        wave_dir.mkdir(parents=True)
        script_path = wave_dir / f"wave_{rig}_gpu{gpu}.sh"
        script_path.write_text(
            wave_script(
                wave_id=args.wave_id,
                rig=rig,
                gpu=gpu,
                flat=flat,
                nested=nested,
                args=quoted_overrides(overrides(source, target)),
            )
        )
        script_path.chmod(
            script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        (wave_dir / "README.md").write_text(
            wave_readme(
                wave_id=args.wave_id,
                prepared_at=prepared_at,
                rig=rig,
                gpu=gpu,
                flat=flat,
            )
        )
        tracking_rows.append(tracking_row(config, args.wave_id, rig, gpu))

    insert_tracking_rows(args.experiments_file, args.wave_id, tracking_rows)
    print(f"generated {len(cells)} wave scripts, READMEs, and tracking rows")


if __name__ == "__main__":
    main()
