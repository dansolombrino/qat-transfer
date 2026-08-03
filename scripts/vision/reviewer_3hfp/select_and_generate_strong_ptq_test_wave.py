#!/usr/bin/env python3
"""Select validation-best strong-PTQ alphas and generate frozen test jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_imagenet_strong_ptq_alpha_wave import (
    ALPHAS,
    AWQ,
    BEHEMOTH_AUTHORIZED_GPUS,
    GPTQ,
    MODEL,
    MODEL_PATH,
    OPTIM,
    PROJECT_ROOTS,
    QAT,
    alpha_text,
    awq_artifact,
    gptq_artifact,
)


def val_artifact(method: str, target: str, alpha: float) -> str:
    return awq_artifact(target, alpha, "val") if method == "awq" else gptq_artifact(target, alpha, "val")


def test_artifact(method: str, target: str, alpha: float) -> str:
    return awq_artifact(target, alpha, "test") if method == "awq" else gptq_artifact(target, alpha, "test")


def select(root: Path, method: str, target: str) -> dict:
    metric = f"val_accuracy_fp_head_{method}"
    candidates = []
    for alpha in ALPHAS:
        rel = val_artifact(method, target, alpha)
        path = root / rel
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing validation artifact: {path}")
        payload = json.loads(path.read_text())
        value = payload.get(metric)
        if value is None:
            raise KeyError(f"missing {metric!r} in {path}")
        candidates.append({"alpha": alpha, "accuracy": float(value), "artifact": rel})
    # Conservative deterministic tie break: prefer the smaller patch magnitude
    # among exactly tied validation accuracies.
    best = max(candidates, key=lambda item: (item["accuracy"], -item["alpha"]))
    return {"metric": metric, "tie_break": "smallest_alpha", "selected": best, "candidates": candidates}


def hydra_command(method: str, target: str, alpha: float) -> tuple[str, str]:
    common = (
        f"model_name={MODEL} batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 "
        "max_grad_norm=1.0 gpu=0 eval_split=test "
        "'source.dataset_names=[\"ImageNet\"]' source.seed=2038 "
        f"'target.dataset_names=[\"{target}\"]' target.seed=2038 "
        "qat.bits=3 qat.granularity=channel 'qat.skip_modules=[\"head\"]' "
        f"'qv.alphas=[{alpha_text(alpha)}]'"
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


def write_job(root: Path, wave: str, lane_key: str, rig: str, gpu: int, method: str, target: str, selection: dict) -> str:
    alpha = float(selection["selected"]["alpha"])
    quant = f"awq={AWQ}" if method == "awq" else f"gptq={GPTQ}"
    experiment = "009_qat_transfer_awq" if method == "awq" else "005_qat_transfer_gptq"
    program = "qv_transfer_awq" if method == "awq" else "qv_transfer_gptq"
    run_id = ",".join((
        f"model={MODEL_PATH}", "src=ImageNet-seed2038", f"tgt={target}-seed2038",
        f"optim={OPTIM}", f"qat={QAT}", quant, f"qv=a{alpha_text(alpha)}", "split=test",
    ))
    rel_dir = Path("scripts/vision/ilharco_timm_supervised") / experiment / program / run_id / f"wave_{wave}"
    out_dir = root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    script_name = f"wave_{rig}_gpu{gpu}.sh"
    artifact = test_artifact(method, target, alpha)
    py_script, args = hydra_command(method, target, alpha)
    guard = ""
    if rig == "behemoth":
        guard = f'''PHYSICAL_GPU={gpu}
BEHEMOTH_AUTHORIZED_GPUS="{BEHEMOTH_AUTHORIZED_GPUS}"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
'''
    log_dir = f"logs/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/method={method},src=ImageNet,tgt={target},qv=a{alpha_text(alpha)},split=test/wave_{wave}"
    (out_dir / script_name).write_text(f'''#!/usr/bin/env bash
# frozen validation selection; wave {wave}; lane {lane_key}
set -uo pipefail
PROJECT_ROOT="{PROJECT_ROOTS[rig]}"
cd "$PROJECT_ROOT" || exit 1
{guard}ARTIFACT='{artifact}'
if [ -s "$ARTIFACT" ]; then echo "[skip] selected test artifact exists: $ARTIFACT"; exit 0; fi
export CUDA_VISIBLE_DEVICES={gpu}
export WAVE_ID={wave}
export HYDRA_FULL_ERROR=1
LOG_DIR='{log_dir}'
mkdir -p "$LOG_DIR" || exit 1
.venv/bin/python {py_script} {args} 2>&1 \\
  | tee "$LOG_DIR/wave_{rig}_gpu{gpu}-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${{PIPESTATUS[@]}}")
rc=${{pipeline_rc[0]}}
if [ "${{pipeline_rc[1]}}" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=${{pipeline_rc[1]}}; fi
if [ ! -s "$ARTIFACT" ] && [ "$rc" -eq 0 ]; then rc=70; fi
exit "$rc"
''')
    (out_dir / script_name).chmod(0o755)
    (out_dir / "README.md").write_text(
        f"# Frozen selected-alpha test\n\nMethod `{method.upper()}`, ImageNet → `{target}`, "
        f"selected alpha `{alpha_text(alpha)}` from `{selection['metric']}` = "
        f"`{selection['selected']['accuracy']}` with smallest-alpha tie-break. "
        f"Wave `{wave}`, placement `{rig}` GPU `{gpu}`.\n"
    )
    return str(rel_dir / script_name)


def write_lane(root: Path, wave_root: Path, wave: str, lane_key: str, rig: str, gpu: int, jobs: list[str]) -> None:
    manifest = wave_root / f"test_{lane_key}.manifest"
    manifest.write_text("\n".join(jobs) + "\n")
    guard = ""
    if rig == "behemoth":
        guard = f'''PHYSICAL_GPU={gpu}
BEHEMOTH_AUTHORIZED_GPUS="{BEHEMOTH_AUTHORIZED_GPUS}"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
'''
    runner = wave_root / f"run_test_{lane_key}.sh"
    runner.write_text(f'''#!/usr/bin/env bash
set -u
PROJECT_ROOT="{PROJECT_ROOTS[rig]}"
cd "$PROJECT_ROOT" || exit 1
{guard}failures=0
while IFS= read -r job; do
  [ -n "$job" ] || continue
  bash "$job" || failures=$((failures + 1))
done < "{wave_root.relative_to(root)}/{manifest.name}"
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
    manifest = json.loads((wave_root / "manifest.json").read_text())
    selections = {"wave": args.wave, "protocol": "validation_best_then_test", "lanes": {}}
    for lane_key, lane in manifest["lanes"].items():
        jobs = []
        lane_selected = []
        for job in lane["jobs"]:
            method, target = job["method"], job["target"]
            chosen = select(root, method, target)
            lane_selected.append({"method": method, "target": target, **chosen})
            jobs.append(write_job(root, args.wave, lane_key, lane["rig"], lane["gpu"], method, target, chosen))
        write_lane(root, wave_root, args.wave, lane_key, lane["rig"], lane["gpu"], jobs)
        selections["lanes"][lane_key] = lane_selected
    (wave_root / "selected_alphas.json").write_text(json.dumps(selections, indent=2) + "\n")
    print(json.dumps(selections, indent=2))


if __name__ == "__main__":
    main()
