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

import importlib.util
import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load `task_vectors.py` directly by file path, instead of putting
# `references/task_vectors/src` on sys.path: that directory contains a
# `datasets/` package which would shadow HuggingFace's `datasets` package
# and break downstream dataset loaders.
_TASK_VECTORS_PATH = (
    Path(__file__).resolve().parents[3]
    / "references" / "task_vectors" / "src" / "task_vectors.py"
)
_spec = importlib.util.spec_from_file_location("task_vectors", _TASK_VECTORS_PATH)
_task_vectors_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task_vectors_mod)
TaskVector = _task_vectors_mod.TaskVector

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from src.vision.modeling import ImageClassifier, ImageEncoder
from src.vision.heads import get_classification_head
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_hf_model_name,
    set_seed,
)
from src.quantization import apply_ptq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn


def _fp_ckpt_path(cfg: DictConfig, dataset_name: str, seed: int, epochs: int) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "fp",
        sanitize_hf_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={seed}",
        f"epoch_{epochs}.pt",
    )


def _qat_ckpt_path(cfg: DictConfig, dataset_name: str, seed: int, epochs: int) -> str:
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "qat",
        sanitize_hf_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
        f"epoch_{epochs}.pt",
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
            leave=False
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


@hydra.main(
    config_path="../../../experiments/001_qat_transfer",
    config_name="qv_transfer",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.target.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    src_epochs = DATASET_NAME_TO_EPOCHS[
        cfg.source.dataset_name
    ] if cfg.source.limit_num_epochs is None else cfg.source.limit_num_epochs

    tgt_epochs = DATASET_NAME_TO_EPOCHS[
        cfg.target.dataset_name
    ] if cfg.target.limit_num_epochs is None else cfg.target.limit_num_epochs

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_source_path = _fp_ckpt_path(cfg, cfg.source.dataset_name, cfg.source.seed, src_epochs)
    qat_source_path = _qat_ckpt_path(cfg, cfg.source.dataset_name, cfg.source.seed, src_epochs)
    fp_target_path = _fp_ckpt_path(cfg, cfg.target.dataset_name, cfg.target.seed, tgt_epochs)

    print(f"\nFP source  checkpoint: {fp_source_path}")
    print(f"QAT source checkpoint: {qat_source_path}")
    print(f"FP target  checkpoint: {fp_target_path}\n")

    for path in (fp_source_path, qat_source_path, fp_target_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required checkpoint missing: {path}")

    ############################################################################
    # END checkpoint paths
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV construction
    ############################################################################
    #
    # We bypass TaskVector's two-checkpoint constructor branch and apply_to(),
    # because both call torch.load(path).state_dict(), which assumes a pickled
    # *model object*. Our checkpoints are saved by ImageEncoder.save() as plain
    # state_dict pickles (torch.save(self.state_dict(), filename)), so they
    # would crash that codepath. Instead we load the three state_dicts directly
    # and build the vector dict ourselves, then hand it to TaskVector via the
    # vector=... constructor branch (preserving arithmetic support for future
    # compositions).

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    qat_src_sd = torch.load(qat_source_path, map_location="cpu")
    fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")

    src_keys = set(fp_src_sd.keys())
    qat_keys = set(qat_src_sd.keys())
    tgt_keys = set(fp_tgt_sd.keys())
    if src_keys != qat_keys:
        log.warning(
            "fp_source and qat_source state_dict key sets differ "
            f"(only-in-fp={sorted(src_keys - qat_keys)[:5]}..., "
            f"only-in-qat={sorted(qat_keys - src_keys)[:5]}...)"
        )
    if tgt_keys != src_keys:
        log.warning(
            "fp_target and fp_source state_dict key sets differ "
            f"(only-in-tgt={sorted(tgt_keys - src_keys)[:5]}..., "
            f"only-in-src={sorted(src_keys - tgt_keys)[:5]}...)"
        )

    vector = {}
    num_dtype_filtered = 0
    with torch.no_grad():
        for k, v_src in fp_src_sd.items():
            # Mirror TaskVector's dtype filter for integer counters/masks.
            if v_src.dtype in (torch.int64, torch.uint8):
                num_dtype_filtered += 1
                continue
            if k not in qat_src_sd:
                print(f"Warning: key {k} present in fp_source but missing in qat_source — skipping")
                continue
            vector[k] = qat_src_sd[k] - v_src

    tv = TaskVector(vector=vector)
    print(
        f"\nQV built: {len(tv.vector)} keys in vector, "
        f"{num_dtype_filtered} keys dtype-filtered (int64/uint8)\n"
    )

    ############################################################################
    # END QV construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN patched-model assembly
    ############################################################################
    #
    # patched = FP_target + alpha * QV (key-by-key)

    alpha = float(cfg.qv.alpha)
    patched = {}
    with torch.no_grad():
        for k, v_tgt in fp_tgt_sd.items():
            if k in tv.vector:
                if tv.vector[k].shape != v_tgt.shape:
                    raise ValueError(
                        f"Shape mismatch on key {k}: tv.vector={tuple(tv.vector[k].shape)} vs "
                        f"fp_target={tuple(v_tgt.shape)}"
                    )
                patched[k] = v_tgt + alpha * tv.vector[k]
            else:
                # int64/uint8 buffer (dtype-filtered) or missing in src/qat — pass through.
                patched[k] = v_tgt

        for k in tv.vector:
            if k not in fp_tgt_sd:
                print(f"Warning: key {k} present in QV but missing in fp_target — skipping")

    ############################################################################
    # END patched-model assembly
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN load patched encoder
    ############################################################################

    image_encoder = ImageEncoder(model_name=cfg.model_name)
    image_encoder.load_state_dict(patched)
    image_encoder.to(device)
    print(f"\n\nimage_encoder (patched):")
    pprint(image_encoder, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_encoder (patched):\n{image_encoder}")

    ############################################################################
    # END load patched encoder
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation (target)
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.target.dataset_name,
        preprocess_train=image_encoder.train_preprocess,
        preprocess_inference=image_encoder.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.target.seed,
    )

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN classification head creation (target)
    ############################################################################

    head_base_path = os.environ['HEAD_BASE_PATH']

    classification_head = get_classification_head(
        model_name=cfg.model_name,
        dataset_name=cfg.target.dataset_name,
        save_dir=head_base_path,
        device=device,
    )

    ############################################################################
    # END classification head creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN image classifier creation
    ############################################################################

    image_classifier = ImageClassifier(
        image_encoder=image_encoder,
        classification_head=classification_head
    )
    image_classifier.to(device)
    print(f"\n\nimage_classifier:")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier:\n{image_classifier}")

    ############################################################################
    # END image classifier creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation (patched QAT, before PTQ)
    ############################################################################
    #
    # Evaluate the patched model in full precision FIRST, before applying PTQ
    # in-place, so we can report both numbers.

    test_accuracy_patched_qat = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (patched QAT, FP_target + {alpha}*QV): {test_accuracy_patched_qat}\n")

    ############################################################################
    # END evaluation (patched QAT)
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN PTQ
    ############################################################################

    skip_modules = frozenset(cfg.ptq.skip_modules)

    all_linear_names = [
        name for name, module in image_classifier.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names = apply_ptq_(
        model=image_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    skipped_names = sorted(set(all_linear_names) - set(quantized_names))

    print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

    print(f"\nQuantized layers ({len(quantized_names)}):")
    for name in quantized_names:
        print(f"  - {name}")

    print(f"\nSkipped layers ({len(skipped_names)}):")
    for name in skipped_names:
        print(f"  - {name}")
    print()

    if cfg.log_to_file:
        log.info(f"Quantized layers ({len(quantized_names)}): {quantized_names}")
        log.info(f"Skipped layers ({len(skipped_names)}): {skipped_names}")

    ############################################################################
    # END PTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation (patched QAT + PTQ)
    ############################################################################

    test_accuracy_patched_qat_ptq = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (patched QAT + PTQ, FP_target + {alpha}*QV): {test_accuracy_patched_qat_ptq}\n")

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes
    print(f"    random chance baseline : {random_chance}  (1 / {num_classes} classes)\n")

    ############################################################################
    # END evaluation (patched QAT + PTQ)
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
        "001_qat_transfer",
        "vision",
        "qv_transfer",
        sanitize_hf_model_name(cfg.model_name),
        f"src={cfg.source.dataset_name}_seed={cfg.source.seed}",
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
            "dataset_name": cfg.source.dataset_name,
            "seed": cfg.source.seed,
            "limit_num_epochs": cfg.source.limit_num_epochs,
            "epochs": src_epochs,
            "fp_checkpoint_path": fp_source_path,
            "qat_checkpoint_path": qat_source_path,
        },
        "target": {
            "dataset_name": cfg.target.dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_checkpoint_path": fp_target_path,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "qv": {
            "alpha": alpha,
            "num_keys_in_vector": len(tv.vector),
            "num_keys_dtype_filtered": num_dtype_filtered,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        "ptq_quantized_modules": quantized_names,
        "ptq_skipped_modules": skipped_names,
        "test_accuracy_patched_qat": test_accuracy_patched_qat,
        "test_accuracy_patched_qat_ptq": test_accuracy_patched_qat_ptq,
        "num_classes": num_classes,
        "random_chance": random_chance,
        "comparison_baseline_note": (
            "Compare to PTQ(QAT_{S2,Q,D2}); NOT computed here. "
            "Run experiments/000_baselines/evaluate_fp_ptq on the QAT checkpoint to obtain it."
        ),
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {eval_results_path}")

    ############################################################################
    # END save results
    ############################################################################


if __name__ == "__main__":
    main()