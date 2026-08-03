"""Generate the approved three-lane behemoth wave from the canonical config."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "code"))
from common.run_id import hydra_override_arg, run_id_flat, run_id_path  # noqa: E402

RUN_ID_PARAMS = ("model", "task", "mode", "seed", "data_spec", "train_spec", "qv_source", "alpha", "quantizer", "eval_spec")
PLACEMENT = {"gsm8k": "0,2", "samsum": "4,5", "e2e_nlg": "6,7"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--source-tag", required=True)
    args = parser.parse_args()
    base = OmegaConf.to_container(OmegaConf.load(ROOT / "config/experiments/text/google_gemma3_causallm/001_qat_transfer/run_task.yaml"), resolve=True)
    wave_records = []
    for task, gpus in PLACEMENT.items():
        cfg = dict(base)
        cfg.update(task=task, mode="full")
        flat = run_id_flat(cfg, RUN_ID_PARAMS)
        relative_run = run_id_path(cfg, RUN_ID_PARAMS)
        directory = ROOT / "scripts/text/google_gemma3_causallm/001_qat_transfer" / flat / f"wave_{args.wave_id}"
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / f"wave_behemoth_gpu{gpus.replace(',', '_')}.sh"
        log = ROOT / "logs/text/google_gemma3_causallm/001_qat_transfer" / relative_run / f"wave_{args.wave_id}" / f"behemoth_gpu{gpus.replace(',', '_')}.log"
        content = f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
PROJECT_ROOT=\"$(git -C \"$SCRIPT_DIR\" rev-parse --show-toplevel)\"
cd \"$PROJECT_ROOT\"
export WAVE_ID={args.wave_id}
export SOURCE_TAG={args.source_tag}
export SOURCE_REVISION=\"$(git rev-list -n 1 \"$SOURCE_TAG\")\"
test \"$(git rev-parse HEAD)\" = \"$SOURCE_REVISION\"
export CUDA_VISIBLE_DEVICES={gpus}
mkdir -p \"$(dirname \"{log.relative_to(ROOT)}\")\"
exec ./.venv/bin/torchrun --standalone --nproc_per_node=2 \\
  code/experiments/text/google_gemma3_causallm/001_qat_transfer/run_task.py \\
  {hydra_override_arg('task', task)} {hydra_override_arg('mode', 'full')} \\
  > \"{log.relative_to(ROOT)}\" 2>&1
"""
        script.write_text(content)
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        readme = directory / "README.md"
        readme.write_text(
            f"# Gemma 3 QV transfer — {task}\n\n"
            f"Wave: `{args.wave_id}`\n\nSource: `{args.source_tag}` "
            "(resolved and verified at launch)\n\n"
            f"Rig/GPU: `behemoth` / `{gpus}`\n\nRun ID: `{flat}`\n"
        )
        wave_records.append({"task": task, "gpus": gpus, "run_id": flat, "run_path": str(relative_run), "script": str(script.relative_to(ROOT)), "log": str(log.relative_to(ROOT))})
    manifest_dir = ROOT / "scripts/text/google_gemma3_causallm/001_qat_transfer" / f"wave_{args.wave_id}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({"wave_id": args.wave_id, "source_revision": "resolved-and-verified-from-tag-at-launch", "source_tag": args.source_tag, "one_wave_gpu_authorization": [0, 2, 4, 5, 6, 7], "runs": wave_records}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(wave_records, indent=2))


if __name__ == "__main__":
    main()
