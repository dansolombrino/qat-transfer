"""Generate the wave that completes phase 009's donor axis from 15 to 22.

Phase `009_qat_transfer_awq` was dispatched as wave 20260803-140339 in two lanes: 14
donors on rig-4090 and 7 donors on rig-3090-ti. Only the first lane ever ran. Together
with the 22-cell ImageNet donor row from wave 20260802-160930 that leaves 330 of the
484 unit-alpha test cells on disk, so every aggregate computed over the phase spans 15
donors while the GPTQ phase it is compared against spans all 22. The two strong-PTQ arms
of the reviewer response are therefore not evaluated on the same grid, and the response
has to carry an explicit "restricted to the same 15 donors" caveat to stay honest.

This generator emits the 154 missing cells -- the 7 donors CIFAR100, EuroSAT, Flowers102,
KMNIST, PCAM, STL10 and TinyImageNet crossed with all 22 receivers, at `qv.alphas=[1.0]`
on `split=test`. Nothing else about the phase changes: same backbone, same qat and awq
fragments, same runner, so the new cells are drop-in comparable with the 330 already
present.

Three choices deserve justification.

We do not reuse the wrappers already generated for the dead rig-3090-ti lane. Those carry
a Git source-drift gate that requires HEAD to equal `refs/tags/wave--20260803-140339` with
a clean `code/ config/ scripts/` tree. The rig this wave targets, behemoth, holds an rsync
mirror rather than a Git working tree, so that gate can only ever `exit 86` there. The
wrappers emitted here follow the behemoth-compatible shape already proven by wave
20260802-212527 instead: a hardcoded project root, an authorized-GPU guard, and artifact
verification after the run in place of revision pinning.

We do not re-run `generate_full_grid_wave.py` either. Its `validate_existing_grid` refuses
to emit when any non-ImageNet unit-alpha test cell already exists, which is now 308 of them.

Lane assignment is round-robin over a receiver-major cell ordering. Runtime is dominated by
the receiver -- the donor side only loads a checkpoint, and the AWQ calibration batches come
from the receiver's train split -- so striping this way spreads the expensive receivers
(ImageNet, SUN397, Food101) evenly instead of letting one lane inherit all of them. Seven
donors against four lanes are coprime, so the rotation stays even across receivers too.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "vit_base_patch16_224.orig_in21k"
MODEL_TAG = "vit_base_patch16_224_orig_in21k"

DONORS = [
    "CIFAR100",
    "EuroSAT",
    "Flowers102",
    "KMNIST",
    "PCAM",
    "STL10",
    "TinyImageNet",
]

RECEIVERS = [
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
]

SEED = 2038
OPTIM_FRAG = "optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128"
QAT_FRAG = "qat=b3-gchannel-shead"
AWQ_FRAG = "awq=b3-gchannel-shead-n4-grid20-clip1"
QV_FRAG = "qv=a1.0"
SPLIT_FRAG = "split=test"

EVAL_ROOT = (
    "evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq"
    "/vision/qv_transfer_awq"
)
LOG_ROOT = (
    "logs/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq"
)
SCRIPT_ROOT = (
    "scripts/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq"
)
WAVE_ROOT = "scripts/vision/ilharco_timm_supervised/009_qat_transfer_awq/missing_donors"

RUNNER = (
    "code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq"
    "/qv_transfer_awq.py"
)

# Per-rig project roots and measured throughput. `seconds_per_cell` comes from the
# `.status.json` markers of the earlier 009 waves (rig-3090-ti 196 s, rig-4090 117 s)
# and from this wave's own first 34 cells on behemoth. It is used only to size each
# lane's share so the lanes finish together instead of leaving one straggler; it never
# affects what is computed.
RIGS = {
    "behemoth": {
        "project_root": "/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer",
        "seconds_per_cell": 80.0,
    },
    "rig-3090-ti": {
        "project_root": "/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer",
        "seconds_per_cell": 196.0,
    },
}

EXPECTED_CELLS = len(DONORS) * len(RECEIVERS)


# ---------------------------------------------------------------------------
# Path fragments
# ---------------------------------------------------------------------------


def run_id_flat(donor: str, receiver: str) -> str:
    return ",".join(
        [
            f"model={MODEL_TAG}",
            f"src={donor}-seed{SEED}",
            f"tgt={receiver}-seed{SEED}",
            OPTIM_FRAG,
            QAT_FRAG,
            AWQ_FRAG,
            QV_FRAG,
            SPLIT_FRAG,
        ]
    )


def eval_dir(donor: str, receiver: str) -> str:
    return "/".join(
        [
            EVAL_ROOT,
            f"model={MODEL_TAG}",
            f"src={donor}-seed{SEED}",
            f"tgt={receiver}-seed{SEED}",
            OPTIM_FRAG,
            QAT_FRAG,
            AWQ_FRAG,
            QV_FRAG,
            SPLIT_FRAG,
        ]
    )


def hydra_command(donor: str, receiver: str) -> str:
    return (
        f".venv/bin/python {RUNNER} model_name={MODEL_NAME} batch_size=128 "
        "lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 eval_split=test "
        "limit_num_batches=null log_to_file=false skip_existing=true "
        f"'source.dataset_names=[\"{donor}\"]' source.seed={SEED} "
        f"'target.dataset_names=[\"{receiver}\"]' target.seed={SEED} "
        "qat.bits=3 qat.granularity=channel 'qat.skip_modules=[\"head\"]' "
        "'qv.alphas=[1.0]' "
        "awq.bits=3 awq.granularity=channel 'awq.skip_modules=[\"head\"]' "
        "awq.num_calib_batches=4 awq.n_grid=20 awq.clip=true"
    )


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def cell_script(
    donor: str, receiver: str, rig: str, gpu: int, authorized: str, wave_id: str
) -> str:
    flat = run_id_flat(donor, receiver)
    edir = eval_dir(donor, receiver)
    artifact = f"{edir}/eval_results.json"
    ldir = f"{LOG_ROOT}/{flat}/wave_{wave_id}"
    lane = f"{rig}_gpu{gpu}"
    return f"""#!/usr/bin/env bash
# run: {flat}   experiment: vision/ilharco_timm_supervised/009_qat_transfer_awq
# wave: {wave_id}   rig: {rig}   gpu: {gpu}
set -uo pipefail
PROJECT_ROOT="{RIGS[rig]["project_root"]}"
cd "$PROJECT_ROOT" || exit 1

# Authorization provenance: user approved behemoth GPUs 0,2,4,5,6,7 and rig-3090-ti
# gpu0 for this donor-completion wave; rig-4090 was unreachable at dispatch time.
PHYSICAL_GPU={gpu}
AUTHORIZED_GPUS="{authorized}"
case ",$AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] GPU $PHYSICAL_GPU is outside the authorized {rig} set" >&2; exit 64 ;;
esac

RUN_ID_FLAT='{flat}'
EVAL_DIR='{edir}'
ARTIFACT='{artifact}'
LOG_DIR='{ldir}'

if [ -s "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES={gpu}
export WAVE_ID={wave_id}
export HYDRA_FULL_ERROR=1
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

{hydra_command(donor, receiver)} 2>&1 \\
  | tee "$LOG_DIR/wave_{lane}-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${{PIPESTATUS[@]}}")
python_rc=${{pipeline_rc[0]}}
tee_rc=${{pipeline_rc[1]}}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc; the required run log is incomplete" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi

if [ ! -s "$ARTIFACT" ]; then
  echo "[error] missing golden artifact: $ARTIFACT" >&2
  if [ "$rc" -eq 0 ]; then rc=70; fi
fi
exit "$rc"
"""


def lane_runner(rig: str, gpu: int, authorized: str, wave_id: str, total: int) -> str:
    lane = f"{rig}_gpu{gpu}"
    return f"""#!/usr/bin/env bash
# wave {wave_id}; {rig} GPU {gpu}; user-approved {rig} set {authorized}
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="{RIGS[rig]["project_root"]}"
cd "$PROJECT_ROOT" || exit 1
PHYSICAL_GPU={gpu}
AUTHORIZED_GPUS="{authorized}"
case ",$AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized {rig} GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
MANIFEST="$SCRIPT_DIR/{lane}.manifest"
FAILURES="$SCRIPT_DIR/{lane}.failures"
: > "$FAILURES"
failures=0
completed=0
total={total}
while IFS= read -r job; do
  [ -n "$job" ] || continue
  echo "[$(date --iso-8601=seconds)] lane {lane}: $((completed + 1))/$total $job"
  bash "$job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] FAILED rc=$rc: $job" >&2
    echo "$job" >> "$FAILURES"
    failures=$((failures + 1))
  fi
  completed=$((completed + 1))
done < "$MANIFEST"
echo "[$(date --iso-8601=seconds)] lane {lane}: completed=$completed failures=$failures"
if [ "$failures" -ne 0 ]; then exit 1; fi
"""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def validate_existing_grid() -> None:
    """Refuse to emit a cell that already exists, and refuse if the gap moved."""
    present = []
    for donor in DONORS:
        for receiver in RECEIVERS:
            if os.path.exists(f"{eval_dir(donor, receiver)}/eval_results.json"):
                present.append((donor, receiver))
    if present:
        raise SystemExit(
            f"refusing to emit: {len(present)} of the {EXPECTED_CELLS} target cells "
            f"already exist, first={present[0]}"
        )

    root = f"{EVAL_ROOT}/model={MODEL_TAG}"
    if not os.path.isdir(root):
        raise SystemExit(f"phase 009 evaluation root missing: {root}")
    covered = sorted(
        name[len("src=") : -len(f"-seed{SEED}")]
        for name in os.listdir(root)
        if name.startswith("src=") and name.endswith(f"-seed{SEED}")
    )
    missing = sorted(set(RECEIVERS) - set(covered))
    if missing != sorted(DONORS):
        raise SystemExit(
            f"donor gap has moved: expected {sorted(DONORS)}, found missing {missing}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_lanes(specs: list[str]) -> list[tuple[str, int]]:
    """Parse `rig:gpu,gpu,...` specs into an ordered lane list."""
    lanes: list[tuple[str, int]] = []
    for spec in specs:
        rig, _, gpu_field = spec.partition(":")
        if rig not in RIGS:
            raise SystemExit(f"unknown rig {rig!r}; known: {sorted(RIGS)}")
        if not gpu_field:
            raise SystemExit(f"lane spec {spec!r} names no GPU")
        for token in gpu_field.split(","):
            lanes.append((rig, int(token)))
    if len(set(lanes)) != len(lanes):
        raise SystemExit(f"duplicate lane in {specs}")
    return lanes


def load_exclusions(path: str | None) -> set[tuple[str, str]]:
    """Read `<donor> <receiver>` lines naming cells that already have an artifact.

    Completion is only ever read off artifacts, never off a runner's exit status, so
    this file is produced by globbing `eval_results.json` on the executing rig. Cells
    listed here are dropped from the partition rather than merely skipped at run time,
    which keeps the lane sizes honest and the ETA meaningful.
    """
    if path is None:
        return set()
    done: set[tuple[str, str]] = set()
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise SystemExit(f"malformed exclusion line: {line!r}")
            done.add((fields[0], fields[1]))
    unknown = {cell for cell in done if cell[0] not in DONORS or cell[1] not in RECEIVERS}
    if unknown:
        raise SystemExit(f"exclusion file names cells outside this wave: {sorted(unknown)[:5]}")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-id", required=True)
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["behemoth:4,5,6,7"],
        help="lane specs, e.g. behemoth:0,2,4,5,6,7 rig-3090-ti:0",
    )
    parser.add_argument(
        "--exclude-done",
        help="file of '<donor> <receiver>' lines whose artifact already exists",
    )
    parser.add_argument("--force", action="store_true", help="skip the coverage guards")
    args = parser.parse_args()

    if not args.force:
        validate_existing_grid()

    lane_keys = parse_lanes(args.lanes)
    authorized = {
        rig: ",".join(str(gpu) for rig_name, gpu in lane_keys if rig_name == rig)
        for rig, _ in lane_keys
    }

    done = load_exclusions(args.exclude_done)
    # Receiver-major ordering: the receiver dominates runtime -- the donor side only
    # loads a checkpoint, and AWQ calibrates on the receiver's train split -- so walking
    # receivers outermost and dealing to the hungriest lane spreads the expensive
    # receivers (ImageNet, SUN397, Food101) rather than bunching them on one lane.
    cells = [
        (donor, receiver)
        for receiver in RECEIVERS
        for donor in DONORS
        if (donor, receiver) not in done
    ]
    if len(cells) + len(done) != EXPECTED_CELLS:
        raise SystemExit(
            f"expected {EXPECTED_CELLS} cells, built {len(cells)} + {len(done)} excluded"
        )

    # Greedy least-projected-finish assignment, weighting each lane by its rig's
    # measured seconds-per-cell so a slow rig gets proportionally fewer cells and every
    # lane finishes at about the same time.
    lanes: dict[tuple[str, int], list[str]] = {key: [] for key in lane_keys}
    projected: dict[tuple[str, int], float] = {key: 0.0 for key in lane_keys}
    for donor, receiver in cells:
        key = min(lane_keys, key=lambda candidate: projected[candidate])
        rig, gpu = key
        flat = run_id_flat(donor, receiver)
        job_dir = os.path.join(SCRIPT_ROOT, flat, f"wave_{args.wave_id}")
        os.makedirs(job_dir, exist_ok=True)
        job_path = os.path.join(job_dir, f"wave_{rig}_gpu{gpu}.sh")
        with open(job_path, "w") as handle:
            handle.write(
                cell_script(donor, receiver, rig, gpu, authorized[rig], args.wave_id)
            )
        os.chmod(job_path, 0o755)
        lanes[key].append(job_path)
        projected[key] += float(RIGS[rig]["seconds_per_cell"])

    wave_dir = os.path.join(WAVE_ROOT, f"wave_{args.wave_id}")
    os.makedirs(wave_dir, exist_ok=True)
    for (rig, gpu), jobs in lanes.items():
        lane = f"{rig}_gpu{gpu}"
        with open(os.path.join(wave_dir, f"{lane}.manifest"), "w") as handle:
            handle.write("\n".join(jobs) + "\n")
        runner_path = os.path.join(wave_dir, f"run_{lane}.sh")
        with open(runner_path, "w") as handle:
            handle.write(
                lane_runner(rig, gpu, authorized[rig], args.wave_id, len(jobs))
            )
        os.chmod(runner_path, 0o755)

    manifest = {
        "wave_id": args.wave_id,
        "phase": "vision/ilharco_timm_supervised/009_qat_transfer_awq",
        "purpose": "complete the donor axis from 15 to 22 at alpha=1.0, split=test",
        "model_name": MODEL_NAME,
        "donors": DONORS,
        "receivers": RECEIVERS,
        "seed": SEED,
        "alphas": [1.0],
        "eval_split": "test",
        "lanes": [f"{rig}_gpu{gpu}" for rig, gpu in lane_keys],
        "total_cells": EXPECTED_CELLS,
        "excluded_done": len(done),
        "dispatched_cells": len(cells),
        "lane_sizes": {f"{rig}_gpu{gpu}": len(jobs) for (rig, gpu), jobs in lanes.items()},
        "projected_seconds": {
            f"{rig}_gpu{gpu}": projected[(rig, gpu)] for rig, gpu in lane_keys
        },
    }
    with open(os.path.join(wave_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(
        f"wave {args.wave_id}: {len(cells)} cells dispatched "
        f"({len(done)} already done of {EXPECTED_CELLS})"
    )
    for rig, gpu in lane_keys:
        key = (rig, gpu)
        print(f"  {rig}_gpu{gpu}: {len(lanes[key])} cells, ~{projected[key] / 60:.0f} min")
    print(f"wave dir: {wave_dir}")


if __name__ == "__main__":
    main()
