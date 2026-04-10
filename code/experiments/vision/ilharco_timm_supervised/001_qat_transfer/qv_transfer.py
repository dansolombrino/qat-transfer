# ==============================================================================
# Quantization Vector (QV, qv)
# ==============================================================================
# Let PT be a pre-trained model
#
# Let FP_{S1}^{D1} be a fine-tuning in full precision of PT on dataset D1
# with seed S1
#
# Let QAT_{S1,Q}^{D1} be a quantization-aware fine-tuning of PT on dataset D1
# with seed S1 and quantization configuration Q (i.e. the low precision dtype
# and the granularity of the quantization)
#
# Let QV be the displacement between QAT_{S1,Q}^{D1} and FP_{S1}^{D1}, i.e.
# QV = QAT_{S1,Q}^{D1} - FP_{S1}^{D1}
# ==============================================================================

# ==============================================================================
# QV Transfer
# ==============================================================================
# Can we transfer the benefit of QAT, supposedly captured by a QV, to a model
# that has been trained on a dataset D2 with seed S2?
#
# What does transfer mean?
#
# acc(ptq(QAT_{S2,Q}^{D2}) \approx ptq(FP_{S2}^{D2} + \alpha QV))
#
# Everything related to the dataset, seed and epoch to which we apply the qv
# is referred to as "patched"
#
# Please note that S1 can be either the same or different than S2.
# Please note that D1 can be either the same or different than D2.
# ==============================================================================

import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.quantization import apply_ptq_
from src.task_vectors import TaskVector

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn

# Keys whose prefix matches this belong to the classification head and are
# excluded from the QV (backbone-only transfer).
HEAD_PREFIX = "model.head."


def _is_head_key(k: str) -> bool:
    return k.startswith(HEAD_PREFIX)


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={seed}",
    )


def _qat_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int) -> str:
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "qat",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
    )


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    limit_num_batches: int = None,
):

    loader = dataset.test_loader

    num_batches = len(loader)
    effective_num_batches = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    model.to(device=device)
    model.eval()

    correct = 0
    total = 0

    batch_color = random_tqdm_color()

    with torch.no_grad():

        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc="Evaluating (test)",
            colour=batch_color,
            leave=False,
            **TQDM_KW,
        )

        for i, batch in batch_bar:

            if i >= effective_num_batches:
                break

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(device=device)
            labels = batch['labels'].to(device=device, dtype=torch.long)

            logits = model(inputs)

            top1, = accuracy(logits, labels, topk=(1,))

            correct += top1
            total += labels.size(0)

            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / total:.2f}%"
            )

    top1_acc = correct / total

    return top1_acc


def _run_source(
    cfg: DictConfig,
    source_dataset_name: str,
    fp_tgt_sd: dict,
    qat_tgt_sd: dict,
    dataset,
    num_classes: int,
    device: str,
    tgt_epochs: int,
):
    """Run QV transfer for a single source dataset against the (already loaded) target."""

    src_epochs = DATASET_NAME_TO_EPOCHS[
        source_dataset_name
    ] if cfg.source.limit_num_epochs is None else cfg.source.limit_num_epochs

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed)

    fp_source_path = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    qat_source_path = os.path.join(qat_src_dir, f"classifier_epoch_{src_epochs}.pt")

    fp_target_path = os.path.join(
        _fp_ckpt_dir(cfg, cfg.target.dataset_name, cfg.target.seed),
        f"classifier_epoch_{tgt_epochs}.pt",
    )
    qat_target_path = os.path.join(
        _qat_ckpt_dir(cfg, cfg.target.dataset_name, cfg.target.seed),
        f"classifier_epoch_{tgt_epochs}.pt",
    )

    if IS_SLURM:
        log.info("--- source=%s target=%s ---", source_dataset_name, cfg.target.dataset_name)
        log.info("FP source  classifier: %s", fp_source_path)
        log.info("QAT source classifier: %s", qat_source_path)
    else:
        print(f"\n--- source={source_dataset_name} target={cfg.target.dataset_name} ---")
        print(f"FP source  classifier: {fp_source_path}")
        print(f"QAT source classifier: {qat_source_path}")

    for path in (fp_source_path, qat_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    ############################################################################
    # END checkpoint paths
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV construction (backbone only)
    ############################################################################

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    qat_src_sd = torch.load(qat_source_path, map_location="cpu")

    src_backbone_keys = {k for k in fp_src_sd if not _is_head_key(k)}
    qat_backbone_keys = {k for k in qat_src_sd if not _is_head_key(k)}
    tgt_backbone_keys = {k for k in fp_tgt_sd if not _is_head_key(k)}
    if src_backbone_keys != qat_backbone_keys:
        log.warning(
            "fp_source and qat_source backbone key sets differ "
            f"(only-in-fp={sorted(src_backbone_keys - qat_backbone_keys)[:5]}..., "
            f"only-in-qat={sorted(qat_backbone_keys - src_backbone_keys)[:5]}...)"
        )
    if tgt_backbone_keys != src_backbone_keys:
        log.warning(
            "fp_target and fp_source backbone key sets differ "
            f"(only-in-tgt={sorted(tgt_backbone_keys - src_backbone_keys)[:5]}..., "
            f"only-in-src={sorted(src_backbone_keys - tgt_backbone_keys)[:5]}...)"
        )

    vector = {}
    num_dtype_filtered = 0
    num_head_filtered = 0
    with torch.no_grad():
        for k, v_src in fp_src_sd.items():
            if _is_head_key(k):
                num_head_filtered += 1
                continue
            if v_src.dtype in (torch.int64, torch.uint8):
                num_dtype_filtered += 1
                continue
            if k not in qat_src_sd:
                if IS_SLURM:
                    log.warning("key %s present in fp_source but missing in qat_source — skipping", k)
                else:
                    print(f"Warning: key {k} present in fp_source but missing in qat_source — skipping")
                continue
            vector[k] = qat_src_sd[k] - v_src

    tv = TaskVector(vector=vector)
    if IS_SLURM:
        log.info(
            "QV built (backbone only): %d keys in vector, %d head keys excluded, "
            "%d keys dtype-filtered (int64/uint8)",
            len(tv.vector), num_head_filtered, num_dtype_filtered,
        )
    else:
        print(
            f"\nQV built (backbone only): {len(tv.vector)} keys in vector, "
            f"{num_head_filtered} head keys excluded, "
            f"{num_dtype_filtered} keys dtype-filtered (int64/uint8)\n"
        )

    del fp_src_sd, qat_src_sd

    ############################################################################
    # END QV construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN patched-backbone assembly
    ############################################################################

    alpha = float(cfg.qv.alpha)
    patched_backbone = {}
    with torch.no_grad():
        for k, v_tgt in fp_tgt_sd.items():
            if _is_head_key(k):
                continue
            if k in tv.vector:
                if tv.vector[k].shape != v_tgt.shape:
                    raise ValueError(
                        f"Shape mismatch on key {k}: tv.vector={tuple(tv.vector[k].shape)} vs "
                        f"fp_target={tuple(v_tgt.shape)}"
                    )
                patched_backbone[k] = v_tgt + alpha * tv.vector[k]
            else:
                patched_backbone[k] = v_tgt

        for k in tv.vector:
            if k not in fp_tgt_sd:
                if IS_SLURM:
                    log.warning("key %s present in QV but missing in fp_target — skipping", k)
                else:
                    print(f"Warning: key {k} present in QV but missing in fp_target — skipping")

    del tv

    ############################################################################
    # END patched-backbone assembly
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN build patched state_dicts (backbone + head variants)
    ############################################################################

    fp_tgt_head = {k: v for k, v in fp_tgt_sd.items() if _is_head_key(k)}
    qat_tgt_head = {k: v for k, v in qat_tgt_sd.items() if _is_head_key(k)}

    patched_with_fp_head = {**patched_backbone, **fp_tgt_head}
    patched_with_qat_head = {**patched_backbone, **qat_tgt_head}

    del patched_backbone

    ############################################################################
    # END build patched state_dicts
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval A: patched backbone + FP target head
    ############################################################################

    classifier_fp_head = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier_fp_head.load_state_dict(patched_with_fp_head)
    classifier_fp_head.to(device)
    if IS_SLURM:
        log.info("classifier (patched backbone + FP head): %s", classifier_fp_head)
    else:
        print(f"\n\nclassifier (patched backbone + FP head):")
        pprint(classifier_fp_head, expand_all=True)
        print(f"\n\n")

    test_accuracy_fp_head = evaluate(
        dataset=dataset,
        model=classifier_fp_head,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval test_accuracy (patched + FP head, before PTQ): %s", test_accuracy_fp_head)
    else:
        print(f"\n    eval test_accuracy (patched + FP head, before PTQ): {test_accuracy_fp_head}\n")

    skip_modules = frozenset(cfg.ptq.skip_modules)

    all_linear_names_fp = [
        name for name, module in classifier_fp_head.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names_fp = apply_ptq_(
        model=classifier_fp_head,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    skipped_names_fp = sorted(set(all_linear_names_fp) - set(quantized_names_fp))

    if IS_SLURM:
        log.info(
            "PTQ config: bits=%s, granularity=%s, skip_modules=%s",
            cfg.ptq.bits, cfg.ptq.granularity, list(cfg.ptq.skip_modules),
        )
        log.info(f"Quantized layers ({len(quantized_names_fp)}): {quantized_names_fp}")
        log.info(f"Skipped layers ({len(skipped_names_fp)}): {skipped_names_fp}")
    else:
        print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

        print(f"\nQuantized layers ({len(quantized_names_fp)}):")
        for name in quantized_names_fp:
            print(f"  - {name}")

        print(f"\nSkipped layers ({len(skipped_names_fp)}):")
        for name in skipped_names_fp:
            print(f"  - {name}")
        print()

    test_accuracy_fp_head_ptq = evaluate(
        dataset=dataset,
        model=classifier_fp_head,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval test_accuracy (patched + FP head + PTQ): %s", test_accuracy_fp_head_ptq)
    else:
        print(f"\n    eval test_accuracy (patched + FP head + PTQ): {test_accuracy_fp_head_ptq}\n")

    del classifier_fp_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END Eval A
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval B: patched backbone + QAT target head
    ############################################################################

    classifier_qat_head = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier_qat_head.load_state_dict(patched_with_qat_head)
    classifier_qat_head.to(device)
    if IS_SLURM:
        log.info("classifier (patched backbone + QAT head): %s", classifier_qat_head)
    else:
        print(f"\n\nclassifier (patched backbone + QAT head):")
        pprint(classifier_qat_head, expand_all=True)
        print(f"\n\n")

    test_accuracy_qat_head = evaluate(
        dataset=dataset,
        model=classifier_qat_head,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval test_accuracy (patched + QAT head, before PTQ): %s", test_accuracy_qat_head)
    else:
        print(f"\n    eval test_accuracy (patched + QAT head, before PTQ): {test_accuracy_qat_head}\n")

    all_linear_names_qat = [
        name for name, module in classifier_qat_head.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names_qat = apply_ptq_(
        model=classifier_qat_head,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    skipped_names_qat = sorted(set(all_linear_names_qat) - set(quantized_names_qat))

    test_accuracy_qat_head_ptq = evaluate(
        dataset=dataset,
        model=classifier_qat_head,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval test_accuracy (patched + QAT head + PTQ): %s", test_accuracy_qat_head_ptq)
    else:
        print(f"\n    eval test_accuracy (patched + QAT head + PTQ): {test_accuracy_qat_head_ptq}\n")

    del classifier_qat_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END Eval B
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN summary
    ############################################################################

    num_classes_actual = len(dataset.class_names)
    random_chance = 1.0 / num_classes_actual

    if IS_SLURM:
        log.info("random chance baseline: %s  (1 / %d classes)", random_chance, num_classes_actual)
    else:
        print(f"\n    random chance baseline : {random_chance}  (1 / {num_classes_actual} classes)\n")

    ############################################################################
    # END summary
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save results
    ############################################################################

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "001_qat_transfer",
        "vision",
        "qv_transfer",
        sanitize_timm_model_name(cfg.model_name),
        f"src={source_dataset_name}_seed={cfg.source.seed}",
        f"tgt={cfg.target.dataset_name}_seed={cfg.target.seed}",
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"qv=alpha={alpha}",
    )

    results = {
        "experiment": "qv_transfer",
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
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
            "dataset_name": cfg.target.dataset_name,
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
        "qv": {
            "alpha": alpha,
            "num_keys_in_vector": len(vector),
            "num_head_keys_excluded": num_head_filtered,
            "num_keys_dtype_filtered": num_dtype_filtered,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        "ptq_quantized_modules": quantized_names_fp,
        "ptq_skipped_modules": skipped_names_fp,
        "test_accuracy_fp_head": test_accuracy_fp_head,
        "test_accuracy_fp_head_ptq": test_accuracy_fp_head_ptq,
        "test_accuracy_qat_head": test_accuracy_qat_head,
        "test_accuracy_qat_head_ptq": test_accuracy_qat_head_ptq,
        "num_classes": num_classes_actual,
        "random_chance": random_chance,
        "comparison_baseline_note": (
            "Compare to PTQ(QAT_{S2,Q,D2}); NOT computed here. "
            "Run config/experiments/vision/000_baselines/evaluate_fp_ptq on the QAT checkpoint to obtain it."
        ),
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    if IS_SLURM:
        log.info("Results saved to: %s", eval_results_path)
    else:
        print(f"\nResults saved to: {eval_results_path}")

    ############################################################################
    # END save results
    ############################################################################


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/001_qat_transfer",
    config_name="qv_transfer",
    version_base=None,
)
def main(cfg: DictConfig):

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)

    set_seed(cfg.target.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tgt_epochs = DATASET_NAME_TO_EPOCHS[
        cfg.target.dataset_name
    ] if cfg.target.limit_num_epochs is None else cfg.target.limit_num_epochs

    num_classes = DATASET_NAME_TO_NUM_CLASSES[cfg.target.dataset_name]

    ############################################################################
    # Load target checkpoints once (independent of source)
    ############################################################################

    fp_tgt_dir = _fp_ckpt_dir(cfg, cfg.target.dataset_name, cfg.target.seed)
    qat_tgt_dir = _qat_ckpt_dir(cfg, cfg.target.dataset_name, cfg.target.seed)

    fp_target_path = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")
    qat_target_path = os.path.join(qat_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

    if IS_SLURM:
        log.info("FP target  classifier: %s", fp_target_path)
        log.info("QAT target classifier: %s", qat_target_path)
    else:
        print(f"\nFP target  classifier: {fp_target_path}")
        print(f"QAT target classifier: {qat_target_path}\n")

    for path in (fp_target_path, qat_target_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required target checkpoint missing: {path}")

    fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")
    qat_tgt_sd = torch.load(qat_target_path, map_location="cpu")

    ############################################################################
    # Create target dataset once (reused across all source datasets)
    ############################################################################

    _tmp_classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)

    dataset = get_dataset(
        dataset_name=cfg.target.dataset_name,
        preprocess_train=_tmp_classifier.train_preprocess,
        preprocess_inference=_tmp_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.target.seed,
    )

    del _tmp_classifier

    ############################################################################
    # Iterate over source datasets
    ############################################################################

    for i, source_dataset_name in enumerate(source_dataset_names):
        if IS_SLURM:
            log.info("=== Source %d/%d: %s ===", i + 1, len(source_dataset_names), source_dataset_name)
        else:
            print(f"\n{'='*60}")
            print(f"  Source {i + 1}/{len(source_dataset_names)}: {source_dataset_name}")
            print(f"{'='*60}")

        _run_source(
            cfg=cfg,
            source_dataset_name=source_dataset_name,
            fp_tgt_sd=fp_tgt_sd,
            qat_tgt_sd=qat_tgt_sd,
            dataset=dataset,
            num_classes=num_classes,
            device=device,
            tgt_epochs=tgt_epochs,
        )


if __name__ == "__main__":
    main()
