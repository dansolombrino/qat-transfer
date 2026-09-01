"""The missing controlled row: one checkpoint, one dataset, two readouts.

The ViT rows of the headline table hold the checkpoint fixed and change only how the output is
read; the Qwen row did not (classification used finetuned classifiers, retrieval the pretrained
embedder). This closes that gap on the text side. ONE encoder (Qwen3-Embedding, pretrained), ONE
labeled dataset: embeddings of the test split are read out (a) as classification, through a
linear probe trained on FP embeddings of the train split, and (b) as self-retrieval over the
same test split (nearest neighbour by cosine, self-match masked). The encoder is quantized once;
both readouts consume the same perturbed embeddings, so any asymmetry is attributable to the
readout alone.
"""
import copy, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from dotenv import load_dotenv
load_dotenv()

import hydra, numpy as np, torch
from omegaconf import DictConfig
from rich import print as rprint
from src.quantization import apply_ptq_
from src.text.data.registry import get_dataset


def _texts_labels(loader, limit=None):
    T, Y = [], []
    for i, (texts, labels) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        T += list(texts)
        Y += [int(y) for y in labels]
    return T, np.array(Y)


@hydra.main(config_path="../../../../../config/experiments/text/sentence_transformers/004_input_fragility",
            config_name="same_checkpoint_contrast", version_base=None)
def main(cfg: DictConfig):
    rprint(dict(cfg))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    ds = get_dataset(dataset_name=cfg.dataset_name, batch_size=cfg.batch_size,
                     num_workers=int(os.environ["TORCH_NUM_WORKERS"]), seed=cfg.seed)
    tr_texts, tr_y = _texts_labels(ds.train_loader, cfg.limit_train_batches)
    te_texts, te_y = _texts_labels(ds.test_loader, cfg.limit_test_batches)
    rprint(f"{cfg.dataset_name}: {len(tr_texts)} train, {len(te_texts)} test, "
           f"{len(set(te_y.tolist()))} classes")

    model = SentenceTransformer(cfg.model_name, device=str(device))
    model.max_seq_length = cfg.max_length
    inner = model[0].auto_model
    fp_state = copy.deepcopy(inner.state_dict())

    def emb(texts):
        return np.asarray(model.encode(texts, batch_size=cfg.batch_size,
                                       show_progress_bar=False, normalize_embeddings=True))

    rprint("FP embeddings:")
    E_tr = emb(tr_texts)
    E_fp = emb(te_texts)
    # probe trained ONCE on FP train embeddings and frozen; only the encoder is quantized
    probe = LogisticRegression(max_iter=2000, C=10.0).fit(E_tr, tr_y)
    W, b = probe.coef_, probe.intercept_

    def profile(scores, mask_diag=False):
        s = scores.copy()
        if mask_diag:
            np.fill_diagonal(s, -np.inf)
        order = np.argsort(-s, axis=1)
        top1 = order[:, 0]
        g = np.take_along_axis(s, order, axis=1)
        gap2 = g[:, 0] - g[:, 1]
        return top1, gap2

    results = {}
    for tag, bits in (("fp", None), ("ptq", cfg.ptq.bits)):
        inner.load_state_dict(fp_state)
        if bits is not None:
            n = apply_ptq_(model=inner, bits=bits, granularity=cfg.ptq.granularity,
                           skip_modules=frozenset(cfg.ptq.skip_modules))
            rprint(f"PTQ bits={bits}: {len(n)} layers")
        E = emb(te_texts)
        logits = E @ W.T + b
        sims = E @ E.T                              # self-retrieval within the SAME embeddings
        c_top1, c_gap = profile(logits)
        r_top1, r_gap = profile(sims, mask_diag=True)
        results[tag] = dict(E=E, logits=logits, c_top1=c_top1, c_gap=c_gap,
                            r_top1=r_top1, r_gap=r_gap)
    inner.load_state_dict(fp_state)

    fp, q = results["fp"], results["ptq"]
    eps_c = np.abs(q["logits"] - fp["logits"]).max(1)
    S_fp = fp["E"] @ fp["E"].T; np.fill_diagonal(S_fp, np.nan)
    S_q = q["E"] @ q["E"].T;   np.fill_diagonal(S_q, np.nan)
    eps_r = np.nanmax(np.abs(S_q - S_fp), axis=1)
    out = dict(
        model_name=cfg.model_name, dataset_name=cfg.dataset_name,
        bits=cfg.ptq.bits, granularity=cfg.ptq.granularity, seed=int(cfg.seed),
        n_test=int(len(te_texts)), n_classes=int(len(set(te_y.tolist()))),
        probe_acc_fp=float((fp["c_top1"] == te_y).mean()),
        probe_acc_ptq=float((q["c_top1"] == te_y).mean()),
        clsf_flip=float((q["c_top1"] != fp["c_top1"]).mean()),
        retr_flip=float((q["r_top1"] != fp["r_top1"]).mean()),
        clsf_sep=float(np.median(fp["c_gap"] / np.maximum(2 * eps_c, 1e-12))),
        retr_sep=float(np.median(fp["r_gap"] / np.maximum(2 * eps_r, 1e-12))),
    )
    rprint(f"\n  classification: flip {out['clsf_flip']:.2%}  sep {out['clsf_sep']:.3f}  "
           f"(probe acc {out['probe_acc_fp']:.1%} -> {out['probe_acc_ptq']:.1%})")
    rprint(f"  self-retrieval: flip {out['retr_flip']:.2%}  sep {out['retr_sep']:.3f}")
    rprint(f"  ratio: {out['retr_flip']/max(out['clsf_flip'],1e-9):.1f}x")

    d = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text" / "sentence_transformers" / \
        "same_checkpoint_contrast" / cfg.model_name.replace("/", "_") / cfg.dataset_name / \
        f"bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"seed={cfg.seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "contrast.json").write_text(json.dumps(out, indent=2))
    rprint(f"Saved: {d}")


if __name__ == "__main__":
    main()
