"""Transfer an unfolded AWQ donor displacement to a receiver and evaluate raw."""

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
from tqdm import tqdm

from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    maybe_dictionarize,
)
from src.duration import checkpoint_epochs, mult_path_frag, mult_tag
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import accuracy, sanitize_timm_model_name, set_seed


HEAD_PREFIX = "model.head."
MATERIALIZE_RUN_ID_PARAMS = ["model", "donor", "seed", "mult", "optim", "awq"]
RUN_ID_PARAMS = [
    "model",
    "src",
    "tgt",
    "sseed",
    "tseed",
    "smult",
    "tmult",
    "optim",
    "awq",
    "alpha",
    "split",
]


def _is_head_key(key: str) -> bool:
    return key.startswith(HEAD_PREFIX)


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


def materialize_identity(cfg: DictConfig) -> dict:
    return {
        "model": sanitize_timm_model_name(cfg.model_name),
        "donor": cfg.source.dataset_name,
        "seed": cfg.source.seed,
        "mult": mult_tag(cfg.source.epoch_mult),
        "optim": _optim_tag(cfg),
        "awq": _awq_tag(cfg),
    }


def run_identity(cfg: DictConfig) -> dict:
    return {
        "model": sanitize_timm_model_name(cfg.model_name),
        "src": cfg.source.dataset_name,
        "tgt": cfg.target.dataset_name,
        "sseed": cfg.source.seed,
        "tseed": cfg.target.seed,
        "smult": mult_tag(cfg.source.epoch_mult),
        "tmult": mult_tag(cfg.target.epoch_mult),
        "optim": _optim_tag(cfg),
        "awq": _awq_tag(cfg),
        "alpha": float(cfg.qv.alpha),
        "split": cfg.eval_split,
    }


def fp_checkpoint_path(cfg: DictConfig, dataset_name: str, seed: int) -> Path:
    is_source = dataset_name == cfg.source.dataset_name
    limit = cfg.source.limit_num_epochs if is_source else cfg.target.limit_num_epochs
    epoch_mult = cfg.source.epoch_mult if is_source else cfg.target.epoch_mult
    epochs = checkpoint_epochs(dataset_name, DATASET_NAME_TO_EPOCHS, limit)
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
        mult_path_frag(epoch_mult),
        f"seed={seed}",
        f"classifier_epoch_{epochs}.pt",
    )


def awq_checkpoint_path(cfg: DictConfig) -> Path:
    return (
        Path(
            os.environ["CHECKPOINT_BASE_PATH"],
            "vision",
            "ilharco_timm_supervised",
            "awq_transfer",
        )
        / run_id_path(materialize_identity(cfg), MATERIALIZE_RUN_ID_PARAMS)
        / "classifier_epoch_1.pt"
    )


def evaluation_dir(cfg: DictConfig) -> Path:
    return Path(
        os.environ["EVALUATION_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "010_awq_transfer",
        "qv_transfer_awqv",
    ) / run_id_path(run_identity(cfg), RUN_ID_PARAMS)


def evaluate(dataset, model, device: torch.device, split: str) -> float:
    if split == "test":
        loader = dataset.test_loader
    elif split == "val":
        loader = dataset.val_loader
    else:
        raise ValueError(f"unsupported eval split: {split!r}")
    model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({split})", leave=False):
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            top1, = accuracy(model(images), labels, topk=(1,))
            correct += top1
            total += labels.size(0)
    return correct / total


def build_awq_vector(fp_source: dict, awq_source: dict) -> tuple[dict, int, int]:
    fp_backbone = {key for key in fp_source if not _is_head_key(key)}
    awq_backbone = {key for key in awq_source if not _is_head_key(key)}
    if fp_backbone != awq_backbone:
        raise ValueError("FP and AWQ donor backbone key sets differ")
    vector = {}
    head_filtered = 0
    dtype_filtered = 0
    for key, fp_value in fp_source.items():
        if _is_head_key(key):
            head_filtered += 1
            continue
        if not fp_value.is_floating_point():
            dtype_filtered += 1
            continue
        if fp_value.shape != awq_source[key].shape:
            raise ValueError(f"shape mismatch for donor key {key}")
        vector[key] = awq_source[key] - fp_value
    return vector, head_filtered, dtype_filtered


def apply_vector(fp_target: dict, vector: dict, alpha: float) -> dict:
    patched = {}
    for key, target_value in fp_target.items():
        if _is_head_key(key) or key not in vector:
            patched[key] = target_value
            continue
        if target_value.shape != vector[key].shape:
            raise ValueError(f"shape mismatch for receiver key {key}")
        patched[key] = target_value + alpha * vector[key]
    missing = sorted(set(vector) - set(fp_target))
    if missing:
        raise ValueError(f"AWQ vector keys missing from receiver: {missing[:5]}")
    return patched


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/010_awq_transfer",
    config_name="qv_transfer_awqv",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.eval_split not in ("val", "test"):
        raise ValueError("eval_split must be 'val' or 'test'")
    set_seed(cfg.target.seed)
    identity = run_identity(cfg)
    eval_dir = evaluation_dir(cfg)
    resolved = {**identity, "config": OmegaConf.to_container(cfg, resolve=True)}
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)

    fp_source_path = fp_checkpoint_path(cfg, cfg.source.dataset_name, cfg.source.seed)
    awq_source_path = awq_checkpoint_path(cfg)
    fp_target_path = fp_checkpoint_path(cfg, cfg.target.dataset_name, cfg.target.seed)
    for path in (fp_source_path, awq_source_path, fp_target_path):
        if not path.exists():
            raise FileNotFoundError(f"missing required checkpoint: {path}")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    with StatusWriter(eval_dir) as status:
        status.heartbeat(progress="0/2")
        fp_source = torch.load(fp_source_path, map_location="cpu")
        awq_source = torch.load(awq_source_path, map_location="cpu")
        fp_target = torch.load(fp_target_path, map_location="cpu")
        vector, head_filtered, dtype_filtered = build_awq_vector(
            fp_source, awq_source
        )
        patched = apply_vector(fp_target, vector, float(cfg.qv.alpha))
        del fp_source, awq_source, fp_target

        classifier = ImageClassifier(
            model_name=cfg.model_name,
            num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.target.dataset_name],
        )
        classifier.load_state_dict(patched)
        del patched
        status.heartbeat(progress="1/2")

        dataset = get_dataset(
            dataset_name=cfg.target.dataset_name,
            preprocess_train=classifier.train_preprocess,
            preprocess_inference=classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
            seed=cfg.target.seed,
        )
        accuracy_value = evaluate(
            dataset, classifier, device=device, split=cfg.eval_split
        )
        status.heartbeat(progress="2/2")

        results = {
            "experiment": "010_awq_transfer/qv_transfer_awqv",
            "model_name": cfg.model_name,
            "eval_split": cfg.eval_split,
            "source": {
                "dataset_name": cfg.source.dataset_name,
                "seed": cfg.source.seed,
                "fp_classifier_path": str(fp_source_path),
                "awq_classifier_path": str(awq_source_path),
            },
            "target": {
                "dataset_name": cfg.target.dataset_name,
                "seed": cfg.target.seed,
                "fp_classifier_path": str(fp_target_path),
            },
            "qv": {
                "alpha": float(cfg.qv.alpha),
                "num_keys_in_vector": len(vector),
                "num_head_keys_excluded": head_filtered,
                "num_dtype_filtered": dtype_filtered,
            },
            "awq": OmegaConf.to_container(cfg.awq, resolve=True),
            f"{cfg.eval_split}_accuracy_fp_head": accuracy_value,
            "num_classes": len(dataset.class_names),
            "random_chance": 1.0 / len(dataset.class_names),
            "evaluation_mode": "raw; no quantizer after AWQ-displacement patching",
        }
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / "eval_results.json"
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Results saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
