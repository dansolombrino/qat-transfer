"""Gap-driven bit allocation on TEXT classification.

Companion to the vision classification script and the retrieval one. Same criterion, same
baselines, same pooled capture metric -- only the task changes, which is the point: if the
allocation is a property of rankings rather than of retrieval, it should transfer.

Metrics mirror the other two:
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

import hydra
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.text.data.registry import get_dataset
from src.vision.utils import sanitize_hf_model_name, set_seed
from src.allocation import (gap_sensitivity, mse_sensitivity, hawq_sensitivity_clsf, allocate_bits,
                            apply_bit_allocation_, _linears)
from src.gptq import apply_gptq_
from src.awq import apply_awq_

_HEAD_MODULE = {
    "google-bert/bert-base-uncased": "classifier",
    "google-bert/bert-large-uncased": "classifier",
    "google/embeddinggemma-300m": "score",
    "Qwen/Qwen3-Embedding-0.6B": "score",
}


def _load_split_checkpoint_(model, backbone_path, head_path, device):
    model.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=False),
                          strict=False)
    model.load_state_dict(torch.load(head_path, map_location=device, weights_only=False),
                          strict=False)


@torch.no_grad()
def _logits(model, tokenizer, loader, device, max_length, limit_batches=None):
    model.eval()
    Z, L = [], []
    for i, batch in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        texts, labels = batch
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        out = model(input_ids=enc["input_ids"].to(device, non_blocking=True),
                    attention_mask=enc["attention_mask"].to(device, non_blocking=True)).logits
        Z.append(out.float().cpu())
        L.append(labels.to(dtype=torch.long).cpu())
    return torch.cat(Z).numpy(), torch.cat(L).numpy()


@hydra.main(
    config_path="../../../../../config/experiments/text/ilharco_automodelforsequenceclassification/004_input_fragility",
    config_name="gap_allocation_clsf",
    version_base=None,
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    base_epochs = DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
    epochs = (min(base_epochs, cfg.limit_num_epochs)
              if cfg.limit_num_epochs is not None else base_epochs)
    _pp(dict(cfg))

    dataset = get_dataset(dataset_name=cfg.dataset_name, batch_size=cfg.batch_size,
                          num_workers=int(os.environ["TORCH_NUM_WORKERS"]), seed=cfg.seed)
    num_classes = len(dataset.class_names)

    if cfg.model_name not in _HEAD_MODULE:
        raise ValueError(f"Unsupported model_name={cfg.model_name!r}; add it to _HEAD_MODULE "
                         f"after checking pooling assumptions.")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name,
                                                               num_labels=num_classes)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=torch.float32)

    cbp = os.environ["CHECKPOINT_BASE_PATH"]
    ckpt_dir = os.path.join(
        cbp, "text", "ilharco_automodelforsequenceclassification", "fp",
        sanitize_hf_model_name(cfg.model_name), cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        f"seed={cfg.ckpt_seed}",
    )
    print(f"\nLoading: {ckpt_dir}")
    _load_split_checkpoint_(model,
                            os.path.join(ckpt_dir, f"backbone_epoch_{epochs}.pt"),
                            os.path.join(ckpt_dir, f"head_epoch_{epochs}.pt"), device)
    fp_state = copy.deepcopy(model.state_dict())

    print("\nFP pass (test):")
    z_fp, labels = _logits(model, tokenizer, dataset.test_loader, device, cfg.max_length,
                           cfg.limit_num_batches)
    fp_top1 = z_fp.argmax(1)
    fp_correct = fp_top1 == labels
    print(f"  N={len(labels)}  FP acc={fp_correct.mean():.4f}  classes={num_classes}")

    # Sensitivity is measured on TRAIN batches; the test split never fits the allocation.
    def score_fn():
        z, _ = _logits(model, tokenizer, dataset.train_loader, device, cfg.max_length,
                       cfg.calib_batches)
        return z

    skip = frozenset(cfg.ptq.skip_modules)
    numel = {n: m.weight.numel() for n, m in _linears(model, skip)}
    gran = cfg.ptq.granularity
    print(f"\nmeasuring gap sensitivity over {len(numel)} layers "
          f"({cfg.calib_batches} calibration batches from train):")
    g_sens = gap_sensitivity(model, skip, cfg.ptq.bits, gran, score_fn)
    m_sens = mse_sensitivity(model, skip, cfg.ptq.bits, gran)
    h_sens = None
    if bool(cfg.get("add_hawq", False)):
        hb = []
        for bi, batch in enumerate(dataset.train_loader):
            if bi >= cfg.calib_batches:
                break
            hb.append(batch)
        def _fwd_ce(b):
            texts, labels = b
            enc = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length,
                            return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            return logits, labels.to(dtype=torch.long, device=device)
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
              "gap": allocate_bits(g_sens, numel, choices, target)}
    if h_sens is not None:
        allocs["hawq"] = allocate_bits(h_sens, numel, choices, target)

    method = str(getattr(cfg.ptq, "method", "rtn")).lower()
    calib = []
    if method in ("gptq", "awq"):
        for bi, batch in enumerate(dataset.train_loader):
            if bi >= cfg.calib_batches:
                break
            calib.append(batch[0])          # list of raw texts

    def _fwd(m, texts):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length,
                        return_tensors="pt")
        return m(input_ids=enc["input_ids"].to(device),
                 attention_mask=enc["attention_mask"].to(device)).logits

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
                            only=only, calib_batches=calib, forward_fn=_fwd, verbose=False))
        z_q, lab2 = _logits(model, tokenizer, dataset.test_loader, device, cfg.max_length,
                            cfg.limit_num_batches)
        assert (lab2 == labels).all(), "row order mismatch between FP and PTQ pass"
        q_top1 = z_q.argmax(1)
        q_correct = q_top1 == labels
        flip = float((q_top1 != fp_top1).mean())
        lost = float((fp_correct & ~q_correct).sum() / max(fp_correct.sum(), 1))
        bits = sum(numel[k] * alloc[k] for k in numel) / sum(numel.values())
        res[name] = dict(flip=flip, acc=float(q_correct.mean()), correct_lost=lost,
                         median_eps=float(np.median(np.abs(z_q - z_fp).max(1))),
                         avg_bits=bits, n_layers=n, acc_fp=float(fp_correct.mean()))
        print(f"  {name:<8} flip {flip:6.2%}   acc {q_correct.mean():6.2%}   "
              f"correct lost {lost:6.2%}   avg bits {bits:.4f}")
    model.load_state_dict(fp_state)

    out = Path(cbp) / "text" / "ilharco_automodelforsequenceclassification" / \
        "gap_allocation_clsf" / sanitize_hf_model_name(cfg.model_name) / cfg.dataset_name / \
        f"bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"method={method}" / \
        f"avg={float(target):g}" / f"seed={cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_allocation_clsf.json").write_text(json.dumps(
        dict(model_name=cfg.model_name, dataset_name=cfg.dataset_name, task="classification",
             modality="text", bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
             bit_choices=choices, method=method, seed=int(cfg.seed), avg_bits_target=target,
             num_classes=num_classes, n_test=int(len(labels)),
             layer_rho=float(rho), results=res,
             alloc_gap={k: int(v) for k, v in allocs["gap"].items()}), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
