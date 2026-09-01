"""Gap-driven bit allocation on CLASSIFICATION.

The retrieval experiment (text/sentence_transformers/004_input_fragility/gap_allocation.py)
showed that ranking bit-width by sensitivity of the top-1/top-2 gap beats both a random split at
the same budget and the reconstruction-error criterion existing methods use. The semiorder
argument is task-agnostic, so the same criterion should transfer to classification -- but
classification sits far closer to the certificate boundary, so there is much less damage available
to recover. This measures whether it transfers, and by how much.

Metrics mirror the retrieval script:
  flip          prediction changes vs the FP model      (lower better)
  acc           top-1 accuracy                          (higher better)
  correct_lost  of samples FP got right, the fraction the quantized model loses (lower better)
"""
import copy
import json
import logging
import os
import sys
from pathlib import Path
from pprint import pprint as _pp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

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
from src.vision.utils import sanitize_timm_model_name, set_seed
from src.allocation import (gap_sensitivity, mse_sensitivity, hawq_sensitivity_clsf, allocate_bits,
                            apply_bit_allocation_, _linears)
from src.gptq import apply_gptq_
from src.awq import apply_awq_

import hydra
from omegaconf import DictConfig
import numpy as np
import torch


@torch.no_grad()
def _logits(model, loader, device, limit_batches=None):
    model.eval()
    Z, L = [], []
    for bi, batch in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:
            break
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device=device, non_blocking=True)
        Z.append(model(images).float().cpu())
        L.append(batch["labels"].to(dtype=torch.long).cpu())
    return torch.cat(Z).numpy(), torch.cat(L).numpy()


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/004_input_fragility",
    config_name="gap_allocation_clsf",
    version_base=None,
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    epochs = (DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
              if cfg.limit_num_epochs is None else cfg.limit_num_epochs)
    _pp(dict(cfg))

    cbp = os.environ["CHECKPOINT_BASE_PATH"]
    ckpt_dir = os.path.join(
        cbp, "vision", "ilharco_timm_supervised", "fp",
        sanitize_timm_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.ckpt_seed}",
    )
    classifier_path = os.path.join(ckpt_dir, f"classifier_epoch_{epochs}.pt")
    num_classes = DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]
    print(f"\nLoading: {classifier_path}")
    model = ImageClassifier.load(model_name=cfg.model_name, num_classes=num_classes,
                                 filename=classifier_path).to(device)
    fp_state = copy.deepcopy(model.state_dict())

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=model.train_preprocess,
        preprocess_inference=model.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
        seed=cfg.seed,
    )

    print("\nFP pass (test):")
    z_fp, labels = _logits(model, dataset.test_loader, device, cfg.limit_num_batches)
    fp_top1 = z_fp.argmax(1)
    fp_correct = fp_top1 == labels
    print(f"  N={len(labels)}  FP acc={fp_correct.mean():.4f}  classes={num_classes}")

    # Sensitivity is measured on TRAIN batches so the test split never fits the allocation.
    def score_fn():
        z, _ = _logits(model, dataset.train_loader, device, cfg.calib_batches)
        return z

    skip = frozenset(cfg.ptq.skip_modules)
    numel = {n: m.weight.numel() for n, m in _linears(model, skip)}
    gran = cfg.ptq.granularity
    print(f"\nmeasuring gap sensitivity over {len(numel)} layers "
          f"({cfg.calib_batches} calibration batches from train):")
    g_sens, g_tail = gap_sensitivity(model, skip, cfg.ptq.bits, gran, score_fn,
                                     tail_frac=float(cfg.get("tail_frac", 0.25)))
    m_sens = mse_sensitivity(model, skip, cfg.ptq.bits, gran)
    h_sens = None
    if bool(cfg.get("add_hawq", False)):
        hb = []
        for bi, batch in enumerate(dataset.train_loader):
            if bi >= cfg.calib_batches:
                break
            batch = maybe_dictionarize(batch)
            hb.append((batch["images"], batch["labels"]))
        def _fwd_ce(b):
            imgs, labels = b
            return model(imgs.to(device)), labels.to(dtype=torch.long, device=device)
        print("measuring HAWQ (Fisher-trace of the task loss) sensitivity:")
        h_sens = hawq_sensitivity_clsf(model, skip, cfg.ptq.bits, gran, hb, _fwd_ce)
    from scipy import stats
    common = [k for k in g_sens if k in m_sens]
    rho, pv = stats.spearmanr([g_sens[k] for k in common], [m_sens[k] for k in common])
    print(f"  Spearman(gap, mse) over layers = {rho:+.3f} (p={pv:.2g})")

    choices = list(cfg.bit_choices)
    target = float(cfg.avg_bits)
    rnd = {}
    for t in range(cfg.n_random):
        r = np.random.default_rng(1000 + t)
        rnd[f"random{t}"] = allocate_bits({k: float(r.random()) for k in numel},
                                          numel, choices, target)
    allocs = {"floor": {k: min(choices) for k in numel},
              "ceil": {k: max(choices) for k in numel},
              **rnd,
              "mse": allocate_bits(m_sens, numel, choices, target),
              "gap": allocate_bits(g_sens, numel, choices, target),
              "gap_tail": allocate_bits(g_tail, numel, choices, target)}
    if h_sens is not None:
        allocs["hawq"] = allocate_bits(h_sens, numel, choices, target)

    method = str(getattr(cfg.ptq, "method", "rtn")).lower()
    calib = []
    if method in ("gptq", "awq"):
        # calibration images from the TRAIN split, like the sensitivity measurement
        for bi, batch in enumerate(dataset.train_loader):
            if bi >= cfg.calib_batches:
                break
            batch = maybe_dictionarize(batch)
            calib.append(batch["images"])

    res = {}
    for name, alloc in allocs.items():
        model.load_state_dict(fp_state)
        if method == "rtn":
            n = apply_bit_allocation_(model, alloc, gran)
        else:
            fn = apply_gptq_ if method == "gptq" else apply_awq_
            n = 0
            for b in sorted(set(alloc.values())):
                only = frozenset(k for k, v in alloc.items() if v == b)
                n += len(fn(model=model, bits=b, granularity=gran, skip_modules=skip,
                            only=only, calib_batches=calib,
                            forward_fn=lambda m, x: m(x.to(device)), verbose=False))
        z_q, lab2 = _logits(model, dataset.test_loader, device, cfg.limit_num_batches)
        assert (lab2 == labels).all(), "row order mismatch between FP and PTQ pass"
        q_top1 = z_q.argmax(1)
        q_correct = q_top1 == labels
        flip = float((q_top1 != fp_top1).mean())
        acc = float(q_correct.mean())
        lost = float((fp_correct & ~q_correct).sum() / max(fp_correct.sum(), 1))
        eps = float(np.median(np.abs(z_q - z_fp).max(1)))
        bits = sum(numel[k] * alloc[k] for k in numel) / sum(numel.values())
        res[name] = dict(flip=flip, acc=acc, correct_lost=lost, median_eps=eps,
                         avg_bits=bits, n_layers=n, acc_fp=float(fp_correct.mean()))
        print(f"  {name:<8} flip {flip:6.2%}   acc {acc:6.2%}   correct lost {lost:6.2%}   "
              f"avg bits {bits:.4f}")
    model.load_state_dict(fp_state)

    out = Path(cbp) / "vision" / "ilharco_timm_supervised" / "gap_allocation_clsf" / \
        sanitize_timm_model_name(cfg.model_name) / cfg.dataset_name / \
        f"bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"method={method}" / \
        f"avg={float(target):g}" / f"seed={cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_allocation_clsf.json").write_text(json.dumps(
        dict(model_name=cfg.model_name, dataset_name=cfg.dataset_name, task="classification",
             bits=cfg.ptq.bits, granularity=cfg.ptq.granularity, bit_choices=choices,
             method=method, seed=int(cfg.seed), avg_bits_target=target,
             num_classes=num_classes, n_test=int(len(labels)),
             layer_rho=float(rho), results=res,
             alloc_gap={k: int(v) for k, v in allocs["gap"].items()}), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
