"""Fit residual-stream steering vectors for ViT post-training quantization.

For each block of a timm ViT wrapped in ImageClassifier, this script:
  1. Runs the FP model over the val split and records FP predictions.
  2. Applies weight-only PTQ in place (existing apply_ptq_).
  3. Runs the quantized model over the same val split with forward hooks that
     capture the CLS-token output of every block, and records Q predictions.
  4. Splits inputs into:
        good = FP-correct & Q-correct
        bad  = FP-correct & Q-wrong
  5. Per block, computes two steering vectors over the CLS activations:
        v_mean[i] = mean(good) - mean(bad)
        v_csvd[i] = top eigenvector of (Cov(good) - Cov(bad)), sign-aligned with v_mean.
  6. Saves both as a single .pt file at the steering-vectors path.
"""

import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import (
    DATASET_NAME_TO_NUM_CLASSES,
    DATASET_NAME_TO_EPOCHS,
    maybe_dictionarize,
)
from src.vision.utils import (
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.quantization import apply_ptq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


def _collect_preds(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    limit_num_batches: int = None,
    desc: str = "Forward",
    cls_per_block: list = None,
):
    """Run model over loader, returning (preds, labels). If cls_per_block is a
    list of length L (one per timm block), the CLS activation at each block is
    appended to that list in order — caller is responsible for registering the
    hooks before calling this."""

    model.eval()

    num_batches = len(loader)
    effective = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    all_preds, all_labels = [], []
    with torch.no_grad():
        bar = tqdm(
            enumerate(loader),
            total=effective,
            desc=desc,
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        for i, batch in bar:
            if i >= effective:
                break
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(device=device)
            labels = batch['labels'].to(device=device, dtype=torch.long)
            logits = model(inputs)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_preds), torch.cat(all_labels)


def _compute_steering_vectors(cls_good: torch.Tensor, cls_bad: torch.Tensor):
    """Return (v_mean, v_csvd) from (n_good, D) and (n_bad, D) CLS activations."""

    mean_g = cls_good.mean(dim=0)
    mean_b = cls_bad.mean(dim=0)
    v_mean = mean_g - mean_b

    g_centered = cls_good - mean_g
    b_centered = cls_bad - mean_b
    n_g, n_b = cls_good.size(0), cls_bad.size(0)
    cov_g = (g_centered.T @ g_centered) / n_g
    cov_b = (b_centered.T @ b_centered) / n_b
    contrastive = (cov_g - cov_b).float()
    # eigh wants a symmetric matrix; symmetrize to wash out fp noise.
    contrastive = (contrastive + contrastive.T) * 0.5
    _, eigvecs = torch.linalg.eigh(contrastive)
    v_csvd = eigvecs[:, -1]  # eigh returns ascending; take the largest.
    # Sign-align with the mean-diff so v_csvd consistently points good-ward.
    if torch.dot(v_csvd, v_mean) < 0:
        v_csvd = -v_csvd
    return v_mean, v_csvd


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/002_quant_steering",
    config_name="fit_steering_vector",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    epochs = DATASET_NAME_TO_EPOCHS[
        cfg.dataset_name
    ] if cfg.limit_num_epochs is None else cfg.limit_num_epochs

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    ############################################################################
    # BEGIN checkpoint loading
    ############################################################################

    checkpoint_base_path = os.environ['CHECKPOINT_BASE_PATH']

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    checkpoint_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "fp_dryrun" if is_dryrun else "fp",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        checkpoint_dir_parts.append(f"lnb={lnb}_lne={lne}")
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")

    print(f"\nLoading encoder from: {classifier_path}")
    image_classifier = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        filename=classifier_path,
    )
    image_classifier.to(device)

    # Scope of this phase: timm ViT only. model.model.blocks must be present.
    assert hasattr(image_classifier.model, 'blocks'), (
        "Expected timm ViT with .model.blocks; got "
        f"{type(image_classifier.model).__name__}"
    )
    blocks = list(image_classifier.model.blocks)
    num_blocks = len(blocks)
    print(f"  ViT has {num_blocks} blocks")

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_classifier.train_preprocess,
        preprocess_inference=image_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN FP-prediction pass (no hooks)
    ############################################################################

    print("\nForward (FP) over val_loader ...")
    fp_preds, labels = _collect_preds(
        image_classifier,
        dataset.val_loader,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
        desc="FP (val)",
    )

    ############################################################################
    # END FP-prediction pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN PTQ (in-place fake-quantize)
    ############################################################################

    skip_modules = frozenset(cfg.ptq.skip_modules)

    quantized_names = apply_ptq_(
        model=image_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, "
          f"skip_modules={list(cfg.ptq.skip_modules)}")
    print(f"  Quantized layers: {len(quantized_names)}")

    ############################################################################
    # END PTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Q-prediction pass with CLS-activation hooks
    ############################################################################

    cls_buffers = [[] for _ in range(num_blocks)]

    def _make_hook(idx):
        def _hook(_mod, _inp, out):
            # out is (B, N, D); CLS is index 0.
            cls_buffers[idx].append(out[:, 0, :].detach().to('cpu'))
        return _hook

    handles = [b.register_forward_hook(_make_hook(i)) for i, b in enumerate(blocks)]

    try:
        print("\nForward (PTQ) over val_loader, capturing CLS per block ...")
        q_preds, labels_q = _collect_preds(
            image_classifier,
            dataset.val_loader,
            device=device,
            limit_num_batches=cfg.limit_num_batches,
            desc="Q (val)",
        )
    finally:
        for h in handles:
            h.remove()

    assert torch.equal(labels, labels_q), \
        "Label order changed between FP and Q passes — non-deterministic val_loader"

    cls_per_block = [torch.cat(buf, dim=0) for buf in cls_buffers]  # (N_val, D) each

    ############################################################################
    # END Q-prediction pass
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN good / bad split + steering-vector fitting
    ############################################################################

    fp_correct = (fp_preds == labels)
    q_correct = (q_preds == labels)
    good_mask = fp_correct & q_correct
    bad_mask = fp_correct & ~q_correct

    num_total = int(labels.numel())
    num_fp_correct = int(fp_correct.sum())
    num_q_correct = int(q_correct.sum())
    num_good = int(good_mask.sum())
    num_bad = int(bad_mask.sum())

    fp_accuracy = num_fp_correct / num_total if num_total > 0 else float('nan')
    q_accuracy = num_q_correct / num_total if num_total > 0 else float('nan')

    print()
    print(f"  total val inputs:      {num_total}")
    print(f"  FP-correct:            {num_fp_correct}  ({100 * fp_accuracy:.2f}%)")
    print(f"  Q-correct:             {num_q_correct}  ({100 * q_accuracy:.2f}%)")
    print(f"  good (FP-c & Q-c):     {num_good}")
    print(f"  bad  (FP-c & Q-w):     {num_bad}")

    embed_dim = cls_per_block[0].size(1)

    if num_good == 0 or num_bad == 0:
        log.warning(
            f"Degenerate good/bad split (num_good={num_good}, num_bad={num_bad}); "
            "steering vectors will be saved as zeros."
        )
        v_mean_per_block = torch.zeros(num_blocks, embed_dim)
        v_csvd_per_block = torch.zeros(num_blocks, embed_dim)
    else:
        v_mean_list, v_csvd_list = [], []
        for i in range(num_blocks):
            cls = cls_per_block[i]
            v_mean, v_csvd = _compute_steering_vectors(cls[good_mask], cls[bad_mask])
            v_mean_list.append(v_mean)
            v_csvd_list.append(v_csvd)
        v_mean_per_block = torch.stack(v_mean_list, dim=0)  # (L, D)
        v_csvd_per_block = torch.stack(v_csvd_list, dim=0)  # (L, D)

    ############################################################################
    # END good / bad split + steering-vector fitting
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save steering vectors
    ############################################################################

    skip_modules_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    save_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "steering_vectors",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_modules_tag}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        save_dir_parts.append(f"lnb={lnb}_lne={lne}")
    save_dir = os.path.join(*save_dir_parts)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "steering_vectors.pt")
    torch.save(
        {
            'mean_diff': v_mean_per_block,
            'contrastive_svd': v_csvd_per_block,
            'num_blocks': num_blocks,
            'embed_dim': embed_dim,
            'num_total': num_total,
            'num_fp_correct': num_fp_correct,
            'num_q_correct': num_q_correct,
            'num_good': num_good,
            'num_bad': num_bad,
        },
        save_path,
    )
    print(f"\nSteering vectors saved to: {save_path}")

    metadata = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "limit_num_epochs": cfg.limit_num_epochs,
        "limit_num_batches": cfg.limit_num_batches,
        "epochs": epochs,
        "device": str(device),
        "encoder_path": classifier_path,
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "quantized_layers": quantized_names,
        "num_blocks": num_blocks,
        "embed_dim": embed_dim,
        "num_total": num_total,
        "num_fp_correct": num_fp_correct,
        "num_q_correct": num_q_correct,
        "num_good": num_good,
        "num_bad": num_bad,
        "fp_accuracy": fp_accuracy,
        "q_accuracy": q_accuracy,
        "vectors_path": save_path,
    }
    metadata_path = os.path.join(save_dir, "fit_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    ############################################################################
    # END save steering vectors
    ############################################################################


if __name__ == "__main__":
    main()
