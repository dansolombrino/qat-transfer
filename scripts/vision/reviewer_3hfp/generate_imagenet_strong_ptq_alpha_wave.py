#!/usr/bin/env python3
"""Generate a two-method ImageNet-donor validation alpha-sweep wave.

Each generated job owns one (method, receiver) row and evaluates the complete
paper alpha grid in one process.  This keeps receiver calibration batches
frozen and amortizes dataset/checkpoint setup while still producing one golden
artifact and status marker per alpha.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


ALPHAS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)
MODEL = "vit_base_patch16_224.orig_in21k"
MODEL_PATH = "vit_base_patch16_224_orig_in21k"
OPTIM = "lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128"
QAT = "b3-gchannel-shead"
AWQ = "b3-gchannel-shead-n4-grid20-clip1"
GPTQ = "b3-gchannel-shead-n4-damp0.01-act0"
BEHEMOTH_AUTHORIZED_GPUS = "0,2,4,5,6,7"
PROJECT_ROOTS = {
    "behemoth": "/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer",
    "rig-3090-ti": "/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer",
}

# Measured 009 alpha=1 test runtimes on the 3090 Ti, seconds.  They are used
# only as relative weights for longest-processing-time placement.
TARGET_SECONDS = {
    "Cars": 138,
    "DTD": 79,
    "EuroSAT": 85,
    "GTSRB": 173,
    "MNIST": 149,
    "RESISC45": 118,
    "SUN397": 467,
    "SVHN": 287,
    "CIFAR10": 147,
    "CIFAR100": 147,
    "STL10": 132,
    "Food101": 286,
    "Flowers102": 151,
    "FER2013": 124,
    "PCAM": 349,
    "OxfordIIITPet": 96,
    "RenderedSST2": 79,
    "EMNIST": 243,
    "FashionMNIST": 149,
    "KMNIST": 148,
    "TinyImageNet": 150,
}


@dataclass
class Lane:
    rig: str
    gpu: int
    speed: float
    jobs: list[tuple[str, str, float]] = field(default_factory=list)
    load: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.rig}_gpu{self.gpu}"


def alpha_text(alpha: float) -> str:
    return str(alpha)


def awq_artifact(target: str, alpha: float, split: str) -> str:
    return (
        "evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/"
        f"qv_transfer_awq/model={MODEL_PATH}/src=ImageNet-seed2038/"
        f"tgt={target}-seed2038/optim={OPTIM}/qat={QAT}/awq={AWQ}/"
        f"qv=a{alpha_text(alpha)}/split={split}/eval_results.json"
    )


def gptq_artifact(target: str, alpha: float, split: str) -> str:
    return (
        "evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/"
        f"qv_transfer_gptq/{MODEL_PATH}/src=ImageNet_seed=2038/"
        f"tgt={target}_seed=2038/"
        "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/"
        "qat=bits=3_gran=channel_skip=head/"
        "gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/"
        f"qv=alpha={alpha_text(alpha)}/split={split}/eval_results.json"
    )


def assign_jobs() -> list[Lane]:
    lanes = [Lane("behemoth", gpu, 2.0) for gpu in (0, 2, 4, 5, 6, 7)]
    lanes.append(Lane("rig-3090-ti", 0, 0.5))
    jobs = [
        (method, target, float(seconds))
        for method in ("awq", "gptq")
        for target, seconds in TARGET_SECONDS.items()
    ]
    for method, target, weight in sorted(jobs, key=lambda item: (-item[2], item[0], item[1])):
        lane = min(lanes, key=lambda item: (item.load / item.speed, item.key))
        lane.jobs.append((method, target, weight))
        lane.load += weight
    return lanes


def job_dir(method: str, target: str, wave: str, rig: str, gpu: int) -> Path:
    quant = f"awq={AWQ}" if method == "awq" else f"gptq={GPTQ}"
    experiment = "009_qat_transfer_awq" if method == "awq" else "005_qat_transfer_gptq"
    program = "qv_transfer_awq" if method == "awq" else "qv_transfer_gptq"
    run_id = ",".join(
        (
            f"model={MODEL_PATH}",
            "src=ImageNet-seed2038",
            f"tgt={target}-seed2038",
            f"optim={OPTIM}",
            f"qat={QAT}",
            quant,
            "qv=agrid11",
            "split=val",
        )
    )
    return Path("scripts/vision/ilharco_timm_supervised") / experiment / program / run_id / f"wave_{wave}"


def hydra_command(method: str, target: str) -> tuple[str, str]:
    alpha_list = ",".join(alpha_text(alpha) for alpha in ALPHAS)
    common = (
        f"model_name={MODEL} batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 "
        "max_grad_norm=1.0 gpu=0 eval_split=val "
        "'source.dataset_names=[\"ImageNet\"]' source.seed=2038 "
        f"'target.dataset_names=[\"{target}\"]' target.seed=2038 "
        "qat.bits=3 qat.granularity=channel 'qat.skip_modules=[\"head\"]' "
        f"'qv.alphas=[{alpha_list}]'"
    )
    if method == "awq":
        script = "code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq.py"
        method_args = (
            "awq.bits=3 awq.granularity=channel 'awq.skip_modules=[\"head\"]' "
            "awq.num_calib_batches=4 awq.n_grid=20 awq.clip=true"
        )
    else:
        script = "code/experiments/vision/ilharco_timm_supervised/005_qat_transfer_gptq/qv_transfer_gptq.py"
        method_args = (
            "gptq.bits=3 gptq.granularity=channel 'gptq.skip_modules=[\"head\"]' "
            "gptq.num_calib_batches=4 gptq.percdamp=0.01 gptq.actorder=false "
            "gptq.block_size=128"
        )
    return script, f"{common} {method_args}"


def write_job(root: Path, wave: str, lane: Lane, method: str, target: str) -> str:
    rel_dir = job_dir(method, target, wave, lane.rig, lane.gpu)
    out_dir = root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    script_name = f"wave_{lane.rig}_gpu{lane.gpu}.sh"
    rel_script = rel_dir / script_name
    artifacts = [
        awq_artifact(target, alpha, "val") if method == "awq"
        else gptq_artifact(target, alpha, "val")
        for alpha in ALPHAS
    ]
    artifact_lines = "\n".join(f"  '{path}'" for path in artifacts)
    py_script, hydra_args = hydra_command(method, target)
    run_id = f"method={method},src=ImageNet,tgt={target},qv=agrid11,split=val"
    log_dir = f"logs/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/{run_id}/wave_{wave}"
    behemoth_guard = ""
    if lane.rig == "behemoth":
        behemoth_guard = f'''\n# Authorization provenance: user explicitly approved behemoth GPUs 0,2,4,5,6,7
# for this ImageNet-only AWQ/GPTQ alpha-sweep wave.
PHYSICAL_GPU={lane.gpu}
BEHEMOTH_AUTHORIZED_GPUS="{BEHEMOTH_AUTHORIZED_GPUS}"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] GPU $PHYSICAL_GPU is outside the authorized behemoth set" >&2; exit 64 ;;
esac
'''
    body = f'''#!/usr/bin/env bash
# run: {run_id}   wave: {wave}   rig: {lane.rig}   gpu: {lane.gpu}
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="{PROJECT_ROOTS[lane.rig]}"
cd "$PROJECT_ROOT" || exit 1
{behemoth_guard}
ARTIFACTS=(
{artifact_lines}
)
all_done=true
for artifact in "${{ARTIFACTS[@]}}"; do
  if [ ! -s "$artifact" ]; then all_done=false; break; fi
done
if "$all_done"; then
  echo "[skip] all 11 validation artifacts already exist for {method}/{target}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES={lane.gpu}
export WAVE_ID={wave}
export HYDRA_FULL_ERROR=1
LOG_DIR='{log_dir}'
mkdir -p "$LOG_DIR" || exit 1

.venv/bin/python {py_script} {hydra_args} 2>&1 \\
  | tee "$LOG_DIR/wave_{lane.rig}_gpu{lane.gpu}-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${{PIPESTATUS[@]}}")
python_rc=${{pipeline_rc[0]}}
tee_rc=${{pipeline_rc[1]}}
rc=$python_rc
if [ "$tee_rc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=$tee_rc; fi

missing=0
for artifact in "${{ARTIFACTS[@]}}"; do
  if [ ! -s "$artifact" ]; then
    echo "[error] missing golden artifact: $artifact" >&2
    missing=$((missing + 1))
  fi
done
if [ "$missing" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=70; fi
exit "$rc"
'''
    (out_dir / script_name).write_text(body)
    (out_dir / script_name).chmod(0o755)
    readme = f'''# ImageNet-only strong-PTQ alpha sweep

- Wave: `{wave}`
- Stage: validation alpha grid
- Method: `{method.upper()}`
- Donor: `ImageNet`, seed 2038
- Receiver: `{target}`, seed 2038
- Alpha grid: `{list(ALPHAS)}`
- Placement: `{lane.rig}` GPU `{lane.gpu}`
- Golden outputs: 11 `eval_results.json` files, one per alpha
- Resume: the producer skips existing cells; the wrapper succeeds only when all 11 artifacts are nonempty
- Selection metric: `val_accuracy_fp_head_{method}`
- Follow-up: evaluate only the validation-selected alpha on test

The process evaluates the full row so all alphas reuse the same materialized
receiver calibration batches.  Behemoth placement is covered by the user's
explicit authorization for GPUs `{BEHEMOTH_AUTHORIZED_GPUS}` in this wave.
'''
    (out_dir / "README.md").write_text(readme)
    return str(rel_script)


def write_lane(root: Path, wave_root: Path, wave: str, lane: Lane, jobs: list[str]) -> None:
    manifest = wave_root / f"{lane.key}.manifest"
    manifest.write_text("\n".join(jobs) + "\n")
    guard = ""
    if lane.rig == "behemoth":
        guard = f'''PHYSICAL_GPU={lane.gpu}
BEHEMOTH_AUTHORIZED_GPUS="{BEHEMOTH_AUTHORIZED_GPUS}"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
'''
    runner = wave_root / f"run_{lane.key}.sh"
    runner.write_text(f'''#!/usr/bin/env bash
# wave {wave}; {lane.rig} GPU {lane.gpu}; user-approved behemoth set {BEHEMOTH_AUTHORIZED_GPUS}
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="{PROJECT_ROOTS[lane.rig]}"
cd "$PROJECT_ROOT" || exit 1
{guard}failures=0
completed=0
total=$(wc -l < "$SCRIPT_DIR/{manifest.name}")
while IFS= read -r job; do
  [ -n "$job" ] || continue
  echo "[$(date --iso-8601=seconds)] lane {lane.key}: $((completed + 1))/$total $job"
  bash "$job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] FAILED rc=$rc: $job" >&2
    failures=$((failures + 1))
  fi
  completed=$((completed + 1))
done < "$SCRIPT_DIR/{manifest.name}"
echo "[$(date --iso-8601=seconds)] lane {lane.key}: completed=$completed failures=$failures"
if [ "$failures" -ne 0 ]; then exit 1; fi
''')
    runner.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    wave_root = root / "scripts/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep" / f"wave_{args.wave}"
    wave_root.mkdir(parents=True, exist_ok=True)

    lanes = assign_jobs()
    manifest = {"wave": args.wave, "alphas": ALPHAS, "lanes": {}}
    for lane in lanes:
        scripts = [write_job(root, args.wave, lane, method, target) for method, target, _ in lane.jobs]
        write_lane(root, wave_root, args.wave, lane, scripts)
        manifest["lanes"][lane.key] = {
            "rig": lane.rig,
            "gpu": lane.gpu,
            "speed_weight": lane.speed,
            "estimated_reference_seconds": lane.load,
            "jobs": [{"method": method, "target": target} for method, target, _ in lane.jobs],
        }
    (wave_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (wave_root / "README.md").write_text(
        f"""# Reviewer 3HFP ImageNet-only strong-PTQ alpha sweep

Wave `{args.wave}` contains 42 validation jobs: 21 receivers times AWQ/GPTQ.
Each job evaluates the paper grid `{list(ALPHAS)}`, giving 462 validation cells.
After validation, choose the best alpha independently for each method/receiver
using the corresponding strong-PTQ FP-head metric, then evaluate only those 42
frozen choices on test. ImageNet is donor-only; the self-pair is excluded.

Placement uses rig-3090-ti GPU 0 and behemoth GPUs `{BEHEMOTH_AUTHORIZED_GPUS}`.
Rig-4090 was excluded because its NVIDIA driver was unavailable at preflight.
The user explicitly authorized the six named behemoth GPUs for this wave.
"""
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
