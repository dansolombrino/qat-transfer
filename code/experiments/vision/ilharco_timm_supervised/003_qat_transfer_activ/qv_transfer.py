# ==============================================================================
# Activation-space QAT transfer (003_qat_transfer_activ)
# ==============================================================================
# Same transfer hypothesis as 001_qat_transfer, run in ACTIVATION space instead
# of weight space.
#
# 001 (weight space): patch the target backbone weights,
#     acc(ptq(FP_tgt + alpha * QV)) ?= acc(ptq(QAT_tgt))
# where QV = QAT_src - FP_src.
#
# 003 (activation space): leave the target backbone weights UNCHANGED and inject
# a precomputed steering vector S into the target's activations via forward
# hooks while it runs PTQ'd,
#     acc(ptq(FP_tgt) with +alpha * S injected) ?= acc(ptq(QAT_tgt))
# where S[tap] = mean_val(act_qat_src[tap]) - mean_val(act_fp_src[tap]) is read
# from the cache written by precompute_steering.py (same directory).
#
# As in 001 we evaluate the (here: steered) model with BOTH the target's FP head
# and its QAT head, each before and after PTQ -> 4 accuracy numbers per pair.
#
# Notes:
#   * The backbone state dict loaded is the plain FP_tgt backbone; the "patch"
#     happens entirely through ActivationInjector hooks at eval time.
#   * apply_ptq_ mutates Linear weights in place without replacing modules, so
#     injector hooks registered on the tap modules survive PTQ and stay active
#     for the after-PTQ eval (quantized weights + activation steering).
#   * steering_strategy here MUST match the one used at precompute (it is part
#     of the cache path); a mismatch shows up as a missing-cache skip.
# ==============================================================================

import gc
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
from src.vision.steering import select_tap_modules, ActivationInjector
from src.quantization import apply_ptq_

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn

# Keys whose prefix matches this belong to the classification head and are kept
# separate from the (unmodified) backbone so we can swap the FP / QAT head.
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


def _steering_dir(cfg: DictConfig, source_dataset_name: str) -> str:
    """Cache location written by precompute_steering.py (must match it exactly)."""
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "steering",
        sanitize_timm_model_name(cfg.model_name),
        source_dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"strategy={cfg.steering_strategy}",
        f"seed={cfg.source.seed}",
    )


def _resolve_alpha(cfg, source_dataset_name, target_dataset_name):
    """Return the numeric alpha to use.

    If cfg.steer.alpha is ``"best"``, reads best_alpha_*.json from disk.
    Otherwise returns ``float(cfg.steer.alpha)``.  Returns ``None`` when
    the best_alpha file is missing (caller should skip the pair).
    """
    raw = str(cfg.steer.alpha)
    if raw != "best":
        return float(raw)

    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    sanitized_model = sanitize_timm_model_name(cfg.model_name)
    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    metric_key = cfg.steer.best_metric
    if metric_key is None:
        raise ValueError("steer.best_metric must be set when steer.alpha='best'")
    # Derive file label from metric key: "val_accuracy_fp_head_ptq" -> "fp_head_ptq"
    label = metric_key.replace("val_accuracy_", "").replace("test_accuracy_", "")

    best_alpha_path = os.path.join(
        evaluation_base_path,
        "vision", "ilharco_timm_supervised", "003_qat_transfer_activ", "vision", "qv_transfer",
        sanitized_model,
        f"src={source_dataset_name}_seed={cfg.source.seed}",
        f"tgt={target_dataset_name}_seed={cfg.target.seed}",
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"strategy={cfg.steering_strategy}",
        f"reduce={cfg.token_reduce}",
        f"best_alpha_{label}.json",
    )

    if not os.path.exists(best_alpha_path):
        return None

    with open(best_alpha_path) as f:
        data = json.load(f)

    if len(data) == 1:
        return float(next(iter(data.values()))["alpha"])

    if metric_key in data:
        return float(data[metric_key]["alpha"])

    raise ValueError(
        f"best_alpha file has keys {list(data.keys())} but metric_key={metric_key!r} not found"
    )


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    limit_num_batches: int = None,
):

    if split == "test":
        loader = dataset.test_loader
    elif split == "val":
        loader = dataset.val_loader
    else:
        raise ValueError(f"Unsupported eval_split: {split!r}. Must be 'val' or 'test'.")

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
            desc=f"Evaluating ({split})",
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


def _evaluate_steered(
    cfg,
    state_dict,
    num_classes,
    taps_strategy,
    steering,
    alpha,
    dataset,
    device,
    eval_split,
):
    """Build a classifier from `state_dict`, inject `alpha * steering`, and
    evaluate before and after PTQ with the hooks active throughout.

    Returns (acc_before_ptq, acc_after_ptq, quantized_names, skipped_names).
    """

    classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier.load_state_dict(state_dict)
    classifier.to(device)

    taps = select_tap_modules(classifier, taps_strategy)
    skip_modules = frozenset(cfg.ptq.skip_modules)

    with ActivationInjector(taps, steering, alpha, cfg.token_reduce, device=device):

        acc_before = evaluate(
            dataset=dataset,
            model=classifier,
            device=device,
            split=eval_split,
            limit_num_batches=cfg.limit_num_batches,
        )

        all_linear_names = [
            name for name, module in classifier.named_modules()
            if isinstance(module, nn.Linear)
        ]
        quantized_names = apply_ptq_(
            model=classifier,
            bits=cfg.ptq.bits,
            granularity=cfg.ptq.granularity,
            skip_modules=skip_modules,
        )
        skipped_names = sorted(set(all_linear_names) - set(quantized_names))

        acc_after = evaluate(
            dataset=dataset,
            model=classifier,
            device=device,
            split=eval_split,
            limit_num_batches=cfg.limit_num_batches,
        )

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc_before, acc_after, quantized_names, skipped_names


def _run_pair(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict,
    qat_tgt_sd: dict,
    dataset,
    num_classes: int,
    device: str,
    tgt_epochs: int,
    eval_split: str,
):
    """Run activation-space QV transfer for a single (source, target) pair."""

    src_epochs = DATASET_NAME_TO_EPOCHS[
        source_dataset_name
    ] if cfg.source.limit_num_epochs is None else cfg.source.limit_num_epochs

    ############################################################################
    # BEGIN load steering vectors (precomputed on the source/donor)
    ############################################################################

    steer_dir = _steering_dir(cfg, source_dataset_name)
    steering_path = os.path.join(steer_dir, "steering_vectors.pt")

    if IS_SLURM:
        log.info("--- source=%s target=%s ---", source_dataset_name, target_dataset_name)
        log.info("steering vectors: %s", steering_path)
    else:
        print(f"\n--- source={source_dataset_name} target={target_dataset_name} ---")
        print(f"steering vectors: {steering_path}")

    if not os.path.exists(steering_path):
        log.warning(
            "Skipping source=%s target=%s: steering cache missing: %s "
            "(run precompute_steering.py with matching steering_strategy first)",
            source_dataset_name, target_dataset_name, steering_path,
        )
        return

    steering = torch.load(steering_path, map_location="cpu")
    tap_names = list(steering.keys())

    if IS_SLURM:
        log.info("loaded steering: %d taps, token_reduce=%s", len(tap_names), cfg.token_reduce)
    else:
        print(f"loaded steering: {len(tap_names)} taps, token_reduce={cfg.token_reduce}\n")

    ############################################################################
    # END load steering vectors
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN resolve alpha
    ############################################################################

    alpha = _resolve_alpha(cfg, source_dataset_name, target_dataset_name)
    if alpha is None:
        log.warning("Skipping source=%s target=%s: best_alpha file missing", source_dataset_name, target_dataset_name)
        return

    ############################################################################
    # END resolve alpha
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN build state_dicts (unmodified FP_tgt backbone + head variants)
    #
    # Unlike 001, the backbone weights are NOT patched here; the QAT direction
    # is injected into activations via hooks at eval time. We only choose which
    # classification head to pair with the (plain) FP target backbone.
    ############################################################################

    fp_backbone = {k: v for k, v in fp_tgt_sd.items() if not _is_head_key(k)}
    fp_tgt_head = {k: v for k, v in fp_tgt_sd.items() if _is_head_key(k)}
    qat_tgt_head = {k: v for k, v in qat_tgt_sd.items() if _is_head_key(k)}

    state_with_fp_head = {**fp_backbone, **fp_tgt_head}
    state_with_qat_head = {**fp_backbone, **qat_tgt_head}

    ############################################################################
    # END build state_dicts
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval A: FP target backbone + steering + FP target head
    ############################################################################

    accuracy_fp_head, accuracy_fp_head_ptq, quantized_names_fp, skipped_names_fp = _evaluate_steered(
        cfg=cfg,
        state_dict=state_with_fp_head,
        num_classes=num_classes,
        taps_strategy=cfg.steering_strategy,
        steering=steering,
        alpha=alpha,
        dataset=dataset,
        device=device,
        eval_split=eval_split,
    )

    if IS_SLURM:
        log.info("eval %s_accuracy (steered + FP head, before PTQ): %s", eval_split, accuracy_fp_head)
        log.info("eval %s_accuracy (steered + FP head + PTQ): %s", eval_split, accuracy_fp_head_ptq)
        log.info(
            "PTQ config: bits=%s, granularity=%s, skip_modules=%s",
            cfg.ptq.bits, cfg.ptq.granularity, list(cfg.ptq.skip_modules),
        )
        log.info(f"Quantized layers ({len(quantized_names_fp)}): {quantized_names_fp}")
        log.info(f"Skipped layers ({len(skipped_names_fp)}): {skipped_names_fp}")
    else:
        print(f"\n    eval {eval_split}_accuracy (steered + FP head, before PTQ): {accuracy_fp_head}")
        print(f"    eval {eval_split}_accuracy (steered + FP head + PTQ): {accuracy_fp_head_ptq}\n")
        print(f"PTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")
        print(f"Quantized layers ({len(quantized_names_fp)}), Skipped layers ({len(skipped_names_fp)})\n")

    ############################################################################
    # END Eval A
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval B: FP target backbone + steering + QAT target head
    ############################################################################

    accuracy_qat_head, accuracy_qat_head_ptq, _, _ = _evaluate_steered(
        cfg=cfg,
        state_dict=state_with_qat_head,
        num_classes=num_classes,
        taps_strategy=cfg.steering_strategy,
        steering=steering,
        alpha=alpha,
        dataset=dataset,
        device=device,
        eval_split=eval_split,
    )

    if IS_SLURM:
        log.info("eval %s_accuracy (steered + QAT head, before PTQ): %s", eval_split, accuracy_qat_head)
        log.info("eval %s_accuracy (steered + QAT head + PTQ): %s", eval_split, accuracy_qat_head_ptq)
    else:
        print(f"\n    eval {eval_split}_accuracy (steered + QAT head, before PTQ): {accuracy_qat_head}")
        print(f"    eval {eval_split}_accuracy (steered + QAT head + PTQ): {accuracy_qat_head_ptq}\n")

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

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed)
    fp_source_path_for_results = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    qat_source_path_for_results = os.path.join(qat_src_dir, f"classifier_epoch_{src_epochs}.pt")

    fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed)
    qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed)
    fp_target_path_for_results = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")
    qat_target_path_for_results = os.path.join(qat_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "003_qat_transfer_activ",
        "vision",
        "qv_transfer",
        sanitize_timm_model_name(cfg.model_name),
        f"src={source_dataset_name}_seed={cfg.source.seed}",
        f"tgt={target_dataset_name}_seed={cfg.target.seed}",
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"strategy={cfg.steering_strategy}",
        f"reduce={cfg.token_reduce}",
        f"steer=alpha={alpha}",
        f"split={eval_split}",
    )

    accuracy_key_fp_head = f"{eval_split}_accuracy_fp_head"
    accuracy_key_fp_head_ptq = f"{eval_split}_accuracy_fp_head_ptq"
    accuracy_key_qat_head = f"{eval_split}_accuracy_qat_head"
    accuracy_key_qat_head_ptq = f"{eval_split}_accuracy_qat_head_ptq"

    results = {
        "experiment": "qv_transfer_activ",
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
            "fp_classifier_path": fp_source_path_for_results,
            "qat_classifier_path": qat_source_path_for_results,
        },
        "target": {
            "dataset_name": target_dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_classifier_path": fp_target_path_for_results,
            "qat_classifier_path": qat_target_path_for_results,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "steer": {
            "alpha": alpha,
            "strategy": cfg.steering_strategy,
            "token_reduce": cfg.token_reduce,
            "steering_vectors_path": steering_path,
            "num_taps": len(tap_names),
            "tap_names": tap_names,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        "ptq_quantized_modules": quantized_names_fp,
        "ptq_skipped_modules": skipped_names_fp,
        accuracy_key_fp_head: accuracy_fp_head,
        accuracy_key_fp_head_ptq: accuracy_fp_head_ptq,
        accuracy_key_qat_head: accuracy_qat_head,
        accuracy_key_qat_head_ptq: accuracy_qat_head_ptq,
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
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/003_qat_transfer_activ",
    config_name="qv_transfer",
    version_base=None,
)
def main(cfg: DictConfig):

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)
    target_dataset_names = OmegaConf.to_container(cfg.target.dataset_names, resolve=True)

    set_seed(cfg.target.seed)

    eval_split = cfg.eval_split
    if eval_split not in ("val", "test"):
        raise ValueError(f"Unsupported eval_split: {eval_split!r}. Must be 'val' or 'test'.")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    total_pairs = len(source_dataset_names) * len(target_dataset_names)
    pair_idx = 0

    for ti, target_dataset_name in enumerate(target_dataset_names):

        if IS_SLURM:
            log.info("=== Target %d/%d: %s ===", ti + 1, len(target_dataset_names), target_dataset_name)
        else:
            print(f"\n{'='*60}")
            print(f"  Target {ti + 1}/{len(target_dataset_names)}: {target_dataset_name}")
            print(f"{'='*60}")

        tgt_epochs = DATASET_NAME_TO_EPOCHS[
            target_dataset_name
        ] if cfg.target.limit_num_epochs is None else cfg.target.limit_num_epochs

        num_classes = DATASET_NAME_TO_NUM_CLASSES[target_dataset_name]

        ####################################################################
        # Load target checkpoints
        ####################################################################

        fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed)
        qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed)

        fp_target_path = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")
        qat_target_path = os.path.join(qat_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

        if IS_SLURM:
            log.info("FP target  classifier: %s", fp_target_path)
            log.info("QAT target classifier: %s", qat_target_path)
        else:
            print(f"\nFP target  classifier: {fp_target_path}")
            print(f"QAT target classifier: {qat_target_path}\n")

        target_missing = False
        for path in (fp_target_path, qat_target_path):
            if not os.path.exists(path):
                log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, path)
                target_missing = True
                break
        if target_missing:
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")
        qat_tgt_sd = torch.load(qat_target_path, map_location="cpu")

        ####################################################################
        # Create dataset (target)
        ####################################################################

        _tmp_classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)

        dataset = get_dataset(
            dataset_name=target_dataset_name,
            preprocess_train=_tmp_classifier.train_preprocess,
            preprocess_inference=_tmp_classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ['TORCH_NUM_WORKERS']),
            seed=cfg.target.seed,
        )

        del _tmp_classifier

        ####################################################################
        # Iterate over source datasets
        ####################################################################

        for si, source_dataset_name in enumerate(source_dataset_names):
            pair_idx += 1

            if IS_SLURM:
                log.info("--- Pair %d/%d: source=%s target=%s ---", pair_idx, total_pairs, source_dataset_name, target_dataset_name)
            else:
                print(f"\n--- Pair {pair_idx}/{total_pairs}: source={source_dataset_name} target={target_dataset_name} ---")

            _run_pair(
                cfg=cfg,
                source_dataset_name=source_dataset_name,
                target_dataset_name=target_dataset_name,
                fp_tgt_sd=fp_tgt_sd,
                qat_tgt_sd=qat_tgt_sd,
                dataset=dataset,
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=eval_split,
            )

        ####################################################################
        # Cleanup between target iterations
        ####################################################################

        del dataset, fp_tgt_sd, qat_tgt_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
