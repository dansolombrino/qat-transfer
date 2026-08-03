"""Materialize one unfolded AWQ(FP) donor checkpoint for displacement transfer."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv

load_dotenv()

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter
from src.awq import apply_awq_
from src.vision.data.common import DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import sanitize_timm_model_name, set_seed


RUN_ID_PARAMS = ["model", "donor", "seed", "optim", "awq"]


def _optim_tag(cfg: DictConfig) -> str:
    return (
        f"lr{cfg.lr}-wd{cfg.wd}-ls{cfg.ls}-wl{cfg.wl}"
        f"-mgn{cfg.max_grad_norm}-bs{cfg.batch_size}"
    )


def _awq_tag(cfg: DictConfig) -> str:
    skip = "-".join(sorted(cfg.awq.skip_modules)) or "none"
    return (
        f"b{cfg.awq.bits}-g{cfg.awq.granularity}-s{skip}"
        f"-n{cfg.awq.num_calib_batches}-grid{cfg.awq.n_grid}"
        f"-clip{int(cfg.awq.clip)}"
    )


def run_identity(cfg: DictConfig) -> dict:
    return {
        "model": sanitize_timm_model_name(cfg.model_name),
        "donor": cfg.dataset_name,
        "seed": cfg.seed,
        "optim": _optim_tag(cfg),
        "awq": _awq_tag(cfg),
    }


def fp_checkpoint_path(cfg: DictConfig, dataset_name: str, seed: int) -> Path:
    epochs = (
        DATASET_NAME_TO_EPOCHS[dataset_name]
        if cfg.limit_num_epochs is None
        else cfg.limit_num_epochs
    )
    return Path(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        (
            f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
            f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}"
        ),
        f"seed={seed}",
        f"classifier_epoch_{epochs}.pt",
    )


def awq_checkpoint_dir(cfg: DictConfig) -> Path:
    return Path(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "awq_transfer",
    ) / run_id_path(run_identity(cfg), RUN_ID_PARAMS)


def evaluation_dir(cfg: DictConfig) -> Path:
    return Path(
        os.environ["EVALUATION_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "010_awq_transfer",
        "materialize_awq_checkpoint",
    ) / run_id_path(run_identity(cfg), RUN_ID_PARAMS)


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/010_awq_transfer",
    config_name="materialize_awq_checkpoint",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    identity = run_identity(cfg)
    eval_dir = evaluation_dir(cfg)
    resolved = {**identity, "config": OmegaConf.to_container(cfg, resolve=True)}
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)

    output_dir = awq_checkpoint_dir(cfg)
    output_path = output_dir / "classifier_epoch_1.pt"
    manifest_path = output_dir / "awq_manifest.json"
    fp_path = fp_checkpoint_path(cfg, cfg.dataset_name, cfg.seed)
    if not fp_path.exists():
        raise FileNotFoundError(f"missing donor FP checkpoint: {fp_path}")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    with StatusWriter(eval_dir) as status:
        status.heartbeat(progress="0/3")
        classifier = ImageClassifier.load(
            model_name=cfg.model_name,
            num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
            filename=str(fp_path),
        ).to(device)
        classifier.eval()

        dataset = get_dataset(
            dataset_name=cfg.dataset_name,
            preprocess_train=classifier.train_preprocess,
            preprocess_inference=classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
            seed=cfg.seed,
        )
        calib_batches = []
        for batch in dataset.train_loader:
            if len(calib_batches) >= cfg.awq.num_calib_batches:
                break
            calib_batches.append(batch)
        if len(calib_batches) != cfg.awq.num_calib_batches:
            raise RuntimeError(
                f"expected {cfg.awq.num_calib_batches} calibration batches, "
                f"got {len(calib_batches)}"
            )
        status.heartbeat(progress="1/3")

        all_linear_names = [
            name for name, module in classifier.named_modules()
            if isinstance(module, nn.Linear)
        ]
        quantized_names = apply_awq_(
            model=classifier,
            bits=cfg.awq.bits,
            granularity=cfg.awq.granularity,
            skip_modules=frozenset(cfg.awq.skip_modules),
            calib_loader=calib_batches,
            device=device,
            num_calib_batches=cfg.awq.num_calib_batches,
            n_grid=cfg.awq.n_grid,
            clip=cfg.awq.clip,
        )
        skipped_names = sorted(set(all_linear_names) - set(quantized_names))
        status.heartbeat(progress="2/3")

        classifier.to("cpu")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment": "010_awq_transfer/materialize_awq_checkpoint",
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "seed": cfg.seed,
            "fp_classifier_path": str(fp_path),
            "awq_classifier_path": str(output_path),
            "awq": OmegaConf.to_container(cfg.awq, resolve=True),
            "calibration_split": "train",
            "quantized_layers": quantized_names,
            "skipped_layers": skipped_names,
            "checkpoint_contents": "weights-only unfolded AWQ state_dict",
            "resume_supported": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        # The checkpoint is the golden completion signal, so write it last.
        classifier.save(str(output_path))
        status.heartbeat(progress="3/3")
        print(f"Saved AWQ donor checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
