"""Test whether QV transfer remains useful on top of RepQ-ViT (WP4).

The paper's original transfer result applies vanilla RTN after constructing
``FP_target + alpha * (QAT_source - FP_source)``. This phase replaces RTN with
the ViT-native RepQ-ViT W/A post-training method, directly testing whether the
QV gain is complementary to post-LayerNorm scale reparameterization and
post-Softmax log quantization.

Alpha zero is evaluated once per receiver on the self-pair and is exactly the
RepQ-ViT(FP) baseline. One seeded training-split calibration batch is
materialized per receiver and reused for every donor, alpha, and receiver-head
variant, so compared cells differ only in the QV patch. A resume guard is
required because each transfer cell performs two complete RepQ-ViT
calibrations and evaluations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv

load_dotenv()

import gc
import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm

from src.repqvit import apply_repqvit_
from src.task_vectors import TaskVector
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    make_seeded_loader,
    maybe_dictionarize,
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)


log = logging.getLogger(__name__)
IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
HEAD_PREFIX = "model.head."


def _is_head_key(key: str) -> bool:
    return key.startswith(HEAD_PREFIX)


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int) -> str:
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={seed}",
    )


def _qat_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int) -> str:
    skip_tag = (
        "-".join(sorted(cfg.qat.skip_modules))
        if len(cfg.qat.skip_modules) > 0
        else "none"
    )
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "qat",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
    )


def _eval_dir(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    alpha: float,
    eval_split: str,
) -> str:
    qat_skip_tag = (
        "-".join(sorted(cfg.qat.skip_modules))
        if len(cfg.qat.skip_modules) > 0
        else "none"
    )
    repqvit_skip_tag = (
        "-".join(sorted(cfg.repqvit.skip_modules))
        if len(cfg.repqvit.skip_modules) > 0
        else "none"
    )
    repqvit_frag = (
        f"repqvit=wbits={cfg.repqvit.w_bits}_abits={cfg.repqvit.a_bits}"
        f"_skip={repqvit_skip_tag}_cbs={cfg.repqvit.calib_batch_size}"
    )
    return os.path.join(
        os.environ["EVALUATION_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "006_qat_transfer_repqvit",
        "vision",
        "qv_transfer_repqvit",
        sanitize_timm_model_name(cfg.model_name),
        f"src={source_dataset_name}_seed={cfg.source.seed}",
        f"tgt={target_dataset_name}_seed={cfg.target.seed}",
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        repqvit_frag,
        f"qv=alpha={alpha}",
        f"split={eval_split}",
    )


def _materialize_calibration_batch(
    dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> torch.Tensor:
    # Also reset the main process so num_workers=0 yields the same augmented
    # receiver images as the standalone fp_repqvit baseline.
    set_seed(seed)
    loader = make_seeded_loader(
        dataset=dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Receiver training split is empty; cannot calibrate RepQ-ViT") from exc
    return maybe_dictionarize(batch)["images"].cpu()


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    limit_num_batches: int | None = None,
) -> float:
    if split == "test":
        loader = dataset.test_loader
    elif split == "val":
        loader = dataset.val_loader
    else:
        raise ValueError(f"Unsupported split {split!r}; expected 'val' or 'test'")

    num_batches = len(loader)
    effective_num_batches = (
        min(limit_num_batches, num_batches)
        if limit_num_batches is not None
        else num_batches
    )
    model.to(device=device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc=f"Evaluating ({split})",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        for i, batch in batch_bar:
            if i >= effective_num_batches:
                break
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device=device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = model(inputs)
            top1, = accuracy(logits, labels, topk=(1,))
            correct += top1
            total += labels.size(0)
            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / total:.2f}%",
            )
    if total == 0:
        raise RuntimeError(f"No samples were evaluated on split={split!r}")
    return float(correct / total)


def _evaluate_head_variant(
    cfg: DictConfig,
    state_dict: dict[str, torch.Tensor],
    dataset,
    calib_data: torch.Tensor,
    num_classes: int,
    device: torch.device,
    eval_split: str,
    head_label: str,
) -> tuple[float, float, list[str]]:
    classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier.load_state_dict(state_dict)
    classifier.to(device=device, dtype=torch.float32)

    accuracy_before = evaluate(
        dataset=dataset,
        model=classifier,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    quantized_names = apply_repqvit_(
        model=classifier,
        calib_data=calib_data,
        w_bits=cfg.repqvit.w_bits,
        a_bits=cfg.repqvit.a_bits,
        skip_modules=frozenset(cfg.repqvit.skip_modules),
    )
    accuracy_after = evaluate(
        dataset=dataset,
        model=classifier,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    if IS_SLURM:
        log.info(
            "%s head: pre-RepQ=%s, post-RepQ=%s, quantized_modules=%d",
            head_label,
            accuracy_before,
            accuracy_after,
            len(quantized_names),
        )
    else:
        print(
            f"{head_label} head: pre-RepQ={accuracy_before}, "
            f"post-RepQ={accuracy_after}, quantized_modules={len(quantized_names)}"
        )

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accuracy_before, accuracy_after, quantized_names


def _run_pair_alpha(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict[str, torch.Tensor],
    qat_tgt_sd: dict[str, torch.Tensor],
    task_vector: TaskVector,
    alpha: float,
    dataset,
    calib_data: torch.Tensor,
    num_classes: int,
    device: torch.device,
    src_epochs: int,
    tgt_epochs: int,
    eval_split: str,
) -> None:
    patched_backbone = {}
    with torch.no_grad():
        for key, target_value in fp_tgt_sd.items():
            if _is_head_key(key):
                continue
            if key in task_vector.vector:
                vector_value = task_vector.vector[key]
                if vector_value.shape != target_value.shape:
                    raise ValueError(
                        f"Shape mismatch on {key}: QV={tuple(vector_value.shape)} "
                        f"target={tuple(target_value.shape)}"
                    )
                patched_backbone[key] = target_value + alpha * vector_value
            else:
                patched_backbone[key] = target_value

    fp_head = {key: value for key, value in fp_tgt_sd.items() if _is_head_key(key)}
    qat_head = {key: value for key, value in qat_tgt_sd.items() if _is_head_key(key)}
    fp_state = {**patched_backbone, **fp_head}
    qat_state = {**patched_backbone, **qat_head}
    del patched_backbone

    fp_before, fp_after, quantized_names_fp = _evaluate_head_variant(
        cfg=cfg,
        state_dict=fp_state,
        dataset=dataset,
        calib_data=calib_data,
        num_classes=num_classes,
        device=device,
        eval_split=eval_split,
        head_label="FP",
    )
    del fp_state
    qat_before, qat_after, quantized_names_qat = _evaluate_head_variant(
        cfg=cfg,
        state_dict=qat_state,
        dataset=dataset,
        calib_data=calib_data,
        num_classes=num_classes,
        device=device,
        eval_split=eval_split,
        head_label="QAT",
    )
    del qat_state
    if quantized_names_fp != quantized_names_qat:
        raise RuntimeError("FP- and QAT-head variants quantized different module sets")

    fp_source_path = os.path.join(
        _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed),
        f"classifier_epoch_{src_epochs}.pt",
    )
    qat_source_path = os.path.join(
        _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed),
        f"classifier_epoch_{src_epochs}.pt",
    )
    fp_target_path = os.path.join(
        _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed),
        f"classifier_epoch_{tgt_epochs}.pt",
    )
    qat_target_path = os.path.join(
        _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed),
        f"classifier_epoch_{tgt_epochs}.pt",
    )

    num_classes_actual = len(dataset.class_names)
    results = {
        "experiment": "qv_transfer_repqvit",
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "eval_split": eval_split,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "limit_num_batches": cfg.limit_num_batches,
        "device": str(device),
        "source": {
            "dataset_name": source_dataset_name,
            "seed": cfg.source.seed,
            "limit_num_epochs": cfg.source.limit_num_epochs,
            "epochs": src_epochs,
            "fp_classifier_path": fp_source_path,
            "qat_classifier_path": qat_source_path,
        },
        "target": {
            "dataset_name": target_dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_classifier_path": fp_target_path,
            "qat_classifier_path": qat_target_path,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "qv": {"alpha": alpha, "num_keys_in_vector": len(task_vector.vector)},
        "repqvit": {
            "w_bits": cfg.repqvit.w_bits,
            "a_bits": cfg.repqvit.a_bits,
            "skip_modules": list(cfg.repqvit.skip_modules),
            "calib_batch_size": cfg.repqvit.calib_batch_size,
            "actual_calibration_samples": int(calib_data.shape[0]),
            "calibration_split": "train",
        },
        "repqvit_quantized_modules": quantized_names_fp,
        f"{eval_split}_accuracy_fp_head": fp_before,
        f"{eval_split}_accuracy_fp_head_repqvit": fp_after,
        f"{eval_split}_accuracy_qat_head": qat_before,
        f"{eval_split}_accuracy_qat_head_repqvit": qat_after,
        "num_classes": num_classes_actual,
        "random_chance": 1.0 / num_classes_actual,
        "comparison_baseline_note": (
            "Compare alpha>0 cells with this receiver's self-pair alpha=0.0 "
            "cell, which is RepQ-ViT(FP_target) under identical calibration; "
            "cross-check against 000_baselines/fp_repqvit."
        ),
    }
    eval_dir = _eval_dir(
        cfg,
        source_dataset_name,
        target_dataset_name,
        alpha,
        eval_split,
    )
    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    if IS_SLURM:
        log.info("Results saved to: %s", eval_results_path)
    else:
        print(f"Results saved to: {eval_results_path}")


def _run_pair(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict[str, torch.Tensor],
    qat_tgt_sd: dict[str, torch.Tensor],
    dataset,
    calib_data: torch.Tensor,
    num_classes: int,
    device: torch.device,
    tgt_epochs: int,
    eval_split: str,
) -> None:
    alphas = [float(value) for value in OmegaConf.to_container(cfg.qv.alphas, resolve=True)]
    if source_dataset_name != target_dataset_name:
        alphas = [alpha for alpha in alphas if alpha != 0.0]
    if cfg.skip_existing:
        alphas = [
            alpha
            for alpha in alphas
            if not os.path.exists(
                os.path.join(
                    _eval_dir(
                        cfg,
                        source_dataset_name,
                        target_dataset_name,
                        alpha,
                        eval_split,
                    ),
                    "eval_results.json",
                )
            )
        ]
    if not alphas:
        log.info(
            "Skipping source=%s target=%s: no alphas remain",
            source_dataset_name,
            target_dataset_name,
        )
        return

    src_epochs = (
        DATASET_NAME_TO_EPOCHS[source_dataset_name]
        if cfg.source.limit_num_epochs is None
        else cfg.source.limit_num_epochs
    )
    fp_source_path = os.path.join(
        _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed),
        f"classifier_epoch_{src_epochs}.pt",
    )
    qat_source_path = os.path.join(
        _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed),
        f"classifier_epoch_{src_epochs}.pt",
    )
    for path in (fp_source_path, qat_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    qat_src_sd = torch.load(qat_source_path, map_location="cpu")
    src_keys = {key for key in fp_src_sd if not _is_head_key(key)}
    qat_keys = {key for key in qat_src_sd if not _is_head_key(key)}
    tgt_keys = {key for key in fp_tgt_sd if not _is_head_key(key)}
    if src_keys != qat_keys or src_keys != tgt_keys:
        raise ValueError(
            "Backbone key sets differ across FP source, QAT source, and FP target; "
            "refusing to construct a partial QV"
        )

    vector = {}
    with torch.no_grad():
        for key in src_keys:
            value = fp_src_sd[key]
            if value.dtype in (torch.int64, torch.uint8):
                continue
            vector[key] = qat_src_sd[key] - value
    task_vector = TaskVector(vector=vector)
    del fp_src_sd, qat_src_sd

    for alpha in alphas:
        _run_pair_alpha(
            cfg=cfg,
            source_dataset_name=source_dataset_name,
            target_dataset_name=target_dataset_name,
            fp_tgt_sd=fp_tgt_sd,
            qat_tgt_sd=qat_tgt_sd,
            task_vector=task_vector,
            alpha=alpha,
            dataset=dataset,
            calib_data=calib_data,
            num_classes=num_classes,
            device=device,
            src_epochs=src_epochs,
            tgt_epochs=tgt_epochs,
            eval_split=eval_split,
        )
    del task_vector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/006_qat_transfer_repqvit",
    config_name="qv_transfer_repqvit",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)
    target_dataset_names = OmegaConf.to_container(cfg.target.dataset_names, resolve=True)
    if cfg.eval_split not in ("val", "test"):
        raise ValueError(f"Unsupported eval_split {cfg.eval_split!r}; expected 'val' or 'test'")

    set_seed(cfg.target.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])
    total_pairs = len(source_dataset_names) * len(target_dataset_names)
    pair_idx = 0

    for target_idx, target_dataset_name in enumerate(target_dataset_names):
        if IS_SLURM:
            log.info(
                "=== Target %d/%d: %s ===",
                target_idx + 1,
                len(target_dataset_names),
                target_dataset_name,
            )
        tgt_epochs = (
            DATASET_NAME_TO_EPOCHS[target_dataset_name]
            if cfg.target.limit_num_epochs is None
            else cfg.target.limit_num_epochs
        )
        num_classes = DATASET_NAME_TO_NUM_CLASSES[target_dataset_name]
        fp_target_path = os.path.join(
            _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed),
            f"classifier_epoch_{tgt_epochs}.pt",
        )
        qat_target_path = os.path.join(
            _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed),
            f"classifier_epoch_{tgt_epochs}.pt",
        )
        missing = [path for path in (fp_target_path, qat_target_path) if not os.path.exists(path)]
        if missing:
            for path in missing:
                log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, path)
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")
        qat_tgt_sd = torch.load(qat_target_path, map_location="cpu")
        temporary_classifier = ImageClassifier(
            model_name=cfg.model_name,
            num_classes=num_classes,
        )
        dataset = get_dataset(
            dataset_name=target_dataset_name,
            preprocess_train=temporary_classifier.train_preprocess,
            preprocess_inference=temporary_classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            seed=cfg.target.seed,
        )
        del temporary_classifier
        calib_data = _materialize_calibration_batch(
            dataset=dataset,
            batch_size=cfg.repqvit.calib_batch_size,
            num_workers=num_workers,
            seed=cfg.target.seed,
        )
        if IS_SLURM:
            log.info(
                "Materialized %d calibration samples from %s train split",
                calib_data.shape[0],
                target_dataset_name,
            )

        for source_dataset_name in source_dataset_names:
            pair_idx += 1
            if IS_SLURM:
                log.info(
                    "--- Pair %d/%d: source=%s target=%s ---",
                    pair_idx,
                    total_pairs,
                    source_dataset_name,
                    target_dataset_name,
                )
            _run_pair(
                cfg=cfg,
                source_dataset_name=source_dataset_name,
                target_dataset_name=target_dataset_name,
                fp_tgt_sd=fp_tgt_sd,
                qat_tgt_sd=qat_tgt_sd,
                dataset=dataset,
                calib_data=calib_data,
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=cfg.eval_split,
            )

        del dataset, calib_data, fp_tgt_sd, qat_tgt_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
