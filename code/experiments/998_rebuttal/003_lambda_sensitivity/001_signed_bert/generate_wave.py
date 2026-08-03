"""Generate the approved behemoth wave and optional selected-test followups."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import stat
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.run_id import hydra_override_arg, run_id_flat, run_id_path
from src.text.data.common import DATASET_NAME_TO_EPOCHS


EXPERIMENT = "998_rebuttal/003_lambda_sensitivity/001_signed_bert"
RUN_ID_PARAMS = ["model", "split", "alpha", "receiver", "donors"]
RUN_SCRIPT = (
    "code/experiments/998_rebuttal/003_lambda_sensitivity/"
    "001_signed_bert/run_row.py"
)
GPUS = ["0", "2", "4", "5", "6", "7"]
AUTHORIZED_GPUS = "0,2,4,5,6,7"
NEGATIVE_GRID = [
    -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40,
    -0.45, -0.50, -0.75, -1.00, -1.25, -1.50, -1.75, -2.00,
]


def initial_runs() -> list[dict]:
    datasets = sorted(DATASET_NAME_TO_EPOCHS)
    runs = []
    for alpha in [0.0] + NEGATIVE_GRID:
        for receiver in datasets:
            runs.append(
                dict(
                    model="bert-large",
                    split="val",
                    alpha=alpha,
                    receiver=receiver,
                    donors="all",
                )
            )
    for receiver in datasets:
        runs.append(
            dict(
                model="bert-large",
                split="test",
                alpha=-1.0,
                receiver=receiver,
                donors="all",
            )
        )
    return runs


def followup_runs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [
        dict(
            model=run.get("model", "bert-large"),
            split=run.get("split", "test"),
            alpha=float(run["alpha"]),
            receiver=run["receiver"],
            donors=run.get("donors", run.get("donor")),
        )
        for run in payload["runs"]
    ]


def atomic_write(path: Path, value: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def shell_script(run: dict, wave: str, gpu: str) -> str:
    flat = run_id_flat(run, RUN_ID_PARAMS)
    nested = run_id_path(run, RUN_ID_PARAMS)
    eval_dir = Path("evaluations") / EXPERIMENT / nested
    artifact = eval_dir / "complete.json"
    log_dir = Path("logs") / EXPERIMENT / flat / f"wave_{wave}"
    args = " ".join(
        shlex.quote(hydra_override_arg(key, run[key])) for key in RUN_ID_PARAMS
    )
    return f'''#!/usr/bin/env bash
# run: {flat}   experiment: {EXPERIMENT}
# wave: {wave}   rig: behemoth   gpu: {gpu}
# GPU auth: user-granted 2026-08-02 for wave {wave}: gpu {AUTHORIZED_GPUS}
set -uo pipefail
cd "$(dirname "$0")/../../../../../.." || exit 1

RUN_ID_FLAT={shlex.quote(flat)}
EVAL_DIR={shlex.quote(str(eval_dir))}
LOG_DIR={shlex.quote(str(log_dir))}
ARTIFACT={shlex.quote(str(artifact))}

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}
BEHEMOTH_AUTHORIZED_GPUS={shlex.quote(AUTHORIZED_GPUS)}
for d in ${{CUDA_VISIBLE_DEVICES//,/ }}; do
  case ",$BEHEMOTH_AUTHORIZED_GPUS," in
    *",$d,"*) ;;
    *) echo "[abort] gpu $d is not authorized (authorized: $BEHEMOTH_AUTHORIZED_GPUS)" >&2; exit 1 ;;
  esac
done
export WAVE_ID={shlex.quote(wave)}
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

.venv/bin/python {shlex.quote(RUN_SCRIPT)} {args} 2>&1 \
  | tee "$LOG_DIR/wave_behemoth_gpu{gpu}-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${{PIPESTATUS[@]}}")
python_rc=${{pipeline_rc[0]}}
tee_rc=${{pipeline_rc[1]}}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi
if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'PYEOF'
import datetime, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text()) if p.exists() else {{}}
s.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"))
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(s) + "\\n")
t.replace(p)
PYEOF
fi
exit "$rc"
'''


def readme(run: dict, wave: str, gpu: str, n_runs: int) -> str:
    flat = run_id_flat(run, RUN_ID_PARAMS)
    dispatched = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# wave {wave} — signed BERT-large lambda rebuttal experiment

Dispatched {dispatched} from rig-4090.

Why this wave: measure the negative and zero lambda arms for BERT-large, then
confirm negative validation selections on test for reviewer 3HFP.

This run: `{flat}` → behemoth, gpu {gpu}.

Full wave: {n_runs} currently materialized runs on behemoth GPUs
{AUTHORIZED_GPUS}; GPUs 2,4,5,6,7 were user-authorized on 2026-08-02 for this
wave only. Conditional selected-test runs may be added under this same approved
wave after validation aggregation.
"""


def append_tracking(rows: list[tuple[dict, str]], wave: str) -> None:
    path = PROJECT_ROOT / "EXPERIMENTS.md"
    header = "## 998_rebuttal/003_lambda_sensitivity/001_signed_bert"
    if path.exists():
        text = path.read_text()
    else:
        text = "# Experiments\n\n"
    if header not in text:
        text += (
            f"{header}\n"
            "run_id params: model, split, alpha, receiver, donors "
            "(mirrors RUN_ID_PARAMS in run_row.py)\n"
            "expected final artifact: evaluations/.../<run_id path>/complete.json\n\n"
            "| model | split | alpha | receiver | donors | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |\n"
            "|---|---|---:|---|---|---|---|---:|---|---|---|---|---|---|---|\n"
        )
    existing = set()
    for line in text.splitlines():
        if line.startswith("| bert-"):
            fields = [field.strip() for field in line.strip("|").split("|")]
            if len(fields) >= 8:
                existing.add(tuple(fields[:8]))
    additions = []
    for run, gpu in rows:
        identity = (
            run["model"], run["split"], str(run["alpha"]), run["receiver"],
            run["donors"], wave, "behemoth", gpu,
        )
        if identity in existing:
            continue
        additions.append(
            "| " + " | ".join(identity) + " | todo |  |  |  |  |  |  |"
        )
    if additions:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n".join(additions) + "\n"
    atomic_write(path, text)


def generate(runs: list[dict], wave: str) -> None:
    rows = []
    total = len(runs)
    for index, run in enumerate(runs):
        gpu = GPUS[index % len(GPUS)]
        flat = run_id_flat(run, RUN_ID_PARAMS)
        wave_dir = (
            PROJECT_ROOT / "scripts" / EXPERIMENT / flat / f"wave_{wave}"
        )
        atomic_write(
            wave_dir / f"wave_behemoth_gpu{gpu}.sh",
            shell_script(run, wave, gpu),
            executable=True,
        )
        atomic_write(wave_dir / "README.md", readme(run, wave, gpu, total))
        rows.append((run, gpu))
    append_tracking(rows, wave)
    print(f"generated {total} runs across {len(GPUS)} lanes for wave {wave}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", required=True)
    parser.add_argument("--followup-manifest", type=Path)
    args = parser.parse_args()
    runs = (
        followup_runs(args.followup_manifest)
        if args.followup_manifest else initial_runs()
    )
    generate(runs, args.wave)


if __name__ == "__main__":
    main()
