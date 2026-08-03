"""Run one resumable signed-lambda donor row and certify its raw artifacts."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[5]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(PROJECT_ROOT)

from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter
from src.text.data.common import DATASET_NAME_TO_EPOCHS


EXPERIMENT = "998_rebuttal/003_lambda_sensitivity/001_signed_bert"
RUN_ID_PARAMS = ["model", "split", "alpha", "receiver", "donors"]
RAW_ROOT = Path(
    "evaluations/text/ilharco_automodelforsequenceclassification/"
    "001_qat_transfer/text/qv_transfer"
)
QV_SCRIPT = Path(
    "code/experiments/text/ilharco_automodelforsequenceclassification/"
    "001_qat_transfer/qv_transfer.py"
)

MODEL_NAMES = {
    "bert-base": "google-bert/bert-base-uncased",
    "bert-large": "google-bert/bert-large-uncased",
}
MODEL_DIRS = {
    "bert-base": "google_bert_bert_base_uncased",
    "bert-large": "google_bert_bert_large_uncased",
}


def alpha_text(value: float) -> str:
    return str(float(value))


def all_datasets() -> list[str]:
    return sorted(DATASET_NAME_TO_EPOCHS)


def expand_donors(value: str) -> list[str]:
    donors = all_datasets() if value == "all" else [value]
    unknown = sorted(set(donors) - set(all_datasets()))
    if unknown:
        raise ValueError(f"unknown donors: {unknown}")
    return donors


def raw_cell_path(cfg: DictConfig, donor: str) -> Path:
    model_dir = MODEL_DIRS[cfg.model]
    optim = (
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}"
    )
    quant = f"bits={cfg.qat_bits}_gran={cfg.granularity}_skip={cfg.skip_module}"
    ptq = f"bits={cfg.ptq_bits}_gran={cfg.granularity}_skip={cfg.skip_module}"
    return (
        RAW_ROOT
        / model_dir
        / f"src={donor}_seed={cfg.seed}"
        / f"tgt={cfg.receiver}_seed={cfg.seed}"
        / optim
        / f"qat={quant}"
        / f"ptq={ptq}"
        / f"qv=alpha={alpha_text(cfg.alpha)}"
        / f"split={cfg.split}"
        / "eval_results.json"
    )


def validate_cell(path: Path, cfg: DictConfig, donor: str) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "missing"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable: {exc}"
    expected = {
        "model_name": MODEL_NAMES[cfg.model],
        "eval_split": cfg.split,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            return False, f"{key}={data.get(key)!r}, expected {value!r}"
    if data.get("source", {}).get("dataset_name") != donor:
        return False, "source mismatch"
    if data.get("target", {}).get("dataset_name") != cfg.receiver:
        return False, "target mismatch"
    if float(data.get("qv", {}).get("alpha")) != float(cfg.alpha):
        return False, "alpha mismatch"
    metric = data.get(f"{cfg.split}_accuracy_fp_head_ptq")
    if not isinstance(metric, (int, float)):
        return False, "required metric missing"
    return True, None


def qv_command(cfg: DictConfig, donors: list[str]) -> list[str]:
    donor_list = ",".join(donors)
    return [
        sys.executable,
        str(QV_SCRIPT),
        f"model_name={MODEL_NAMES[cfg.model]}",
        f"batch_size={cfg.batch_size}",
        f"max_length={cfg.max_length}",
        f"eval_split={cfg.split}",
        f"eval_mode={cfg.eval_mode}",
        f"lr={cfg.lr}",
        f"wd={cfg.wd}",
        f"ls={cfg.ls}",
        f"max_grad_norm={cfg.max_grad_norm}",
        f"gpu={cfg.gpu}",
        f"source.dataset_names=[{donor_list}]",
        f"source.seed={cfg.seed}",
        f"target.dataset_names=[{cfg.receiver}]",
        f"target.seed={cfg.seed}",
        f"qat.bits={cfg.qat_bits}",
        f"qat.granularity={cfg.granularity}",
        f"qat.skip_modules=[{cfg.skip_module}]",
        f"qv.alpha={alpha_text(cfg.alpha)}",
        f"ptq.bits={cfg.ptq_bits}",
        f"ptq.granularity={cfg.granularity}",
        f"ptq.skip_modules=[{cfg.skip_module}]",
    ]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@hydra.main(
    config_path=(
        "../../../../../config/experiments/998_rebuttal/"
        "003_lambda_sensitivity/001_signed_bert"
    ),
    config_name="run_row",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"unsupported model: {cfg.model}")
    if cfg.split not in ("val", "test"):
        raise ValueError(f"unsupported split: {cfg.split}")
    if cfg.eval_mode != "fp_head_ptq_only":
        raise ValueError("signed-lambda rows require eval_mode=fp_head_ptq_only")
    if cfg.receiver not in all_datasets():
        raise ValueError(f"unknown receiver: {cfg.receiver}")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    eval_dir = PROJECT_ROOT / "evaluations" / EXPERIMENT / run_id_path(
        resolved, RUN_ID_PARAMS
    )
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)
    completion = eval_dir / "complete.json"
    donors = expand_donors(cfg.donors)

    with StatusWriter(eval_dir) as status:
        invalid = []
        for donor in donors:
            ok, reason = validate_cell(PROJECT_ROOT / raw_cell_path(cfg, donor), cfg, donor)
            if not ok:
                invalid.append((donor, reason))
        status.heartbeat(progress=f"cells {len(donors) - len(invalid)}/{len(donors)}")

        if invalid:
            print(
                "[resume] evaluating missing/invalid donors: "
                + ", ".join(f"{d} ({reason})" for d, reason in invalid),
                flush=True,
            )
            env = os.environ.copy()
            env.setdefault("SLURM_JOB_ID", "signed-lambda")
            process = subprocess.Popen(
                qv_command(cfg, [donor for donor, _ in invalid]),
                cwd=PROJECT_ROOT,
                env=env,
            )
            while process.poll() is None:
                time.sleep(max(5, int(cfg.poll_interval_s)))
                complete_count = sum(
                    validate_cell(PROJECT_ROOT / raw_cell_path(cfg, donor), cfg, donor)[0]
                    for donor in donors
                )
                status.heartbeat(progress=f"cells {complete_count}/{len(donors)}")
            if process.returncode:
                raise RuntimeError(f"qv_transfer exited with {process.returncode}")

        artifacts = []
        for donor in donors:
            path = PROJECT_ROOT / raw_cell_path(cfg, donor)
            ok, reason = validate_cell(path, cfg, donor)
            if not ok:
                raise RuntimeError(f"invalid final artifact for {donor}: {reason}: {path}")
            data = json.loads(path.read_text())
            artifacts.append(
                {
                    "donor": donor,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "metric": data[f"{cfg.split}_accuracy_fp_head_ptq"],
                }
            )

        status.heartbeat(progress=f"cells {len(donors)}/{len(donors)}")
        atomic_json(
            completion,
            {
                "completed": datetime.datetime.now().isoformat(timespec="seconds"),
                "model": cfg.model,
                "split": cfg.split,
                "alpha": float(cfg.alpha),
                "receiver": cfg.receiver,
                "donors": donors,
                "artifacts": artifacts,
            },
        )
        print(f"[complete] {completion.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
