"""Ranking fragility on a deployed cross-modal retrieval system: CLIP text-to-image.

The other retrieval settings here are self-retrieval over a classification test set (a controlled
construction) and a bi-encoder over BEIR. Neither is a shipped cross-modal system. CLIP
text->image is: real captions as queries, real images as the corpus, ground-truth pairing as
relevance, a different architecture family, and no finetuning.
"""
import json, os, sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
from dotenv import load_dotenv
load_dotenv(_R / ".env")

import hydra, numpy as np, pandas as pd, torch
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from src.quantization import apply_ptq_

K_GAPS = 200


def _profile(sims, k):
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(part, order, axis=1)
    return vals[:, :1] - vals, idx


@hydra.main(config_path="../../../../../config/experiments/vision/ilharco_hf_clip/004_input_fragility",
            config_name="crossmodal_gap_profile", version_base=None)
def main(cfg: DictConfig):
    rprint(OmegaConf.to_container(cfg, resolve=True))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model.eval()

    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_path, split=cfg.split,
                      cache_dir=os.environ.get("HF_DATASETS_CACHE"),
                      trust_remote_code=True, verification_mode="no_checks")
    if cfg.max_images and len(ds) > cfg.max_images:
        ds = ds.select(range(cfg.max_images))
    caps, img_of_cap = [], []
    for i in range(len(ds)):
        c = ds[i][cfg.caption_field]
        c = c[0] if isinstance(c, list) and c else (c if isinstance(c, str) else None)
        if c:
            caps.append(c)
            img_of_cap.append(i)
    if cfg.max_queries and len(caps) > cfg.max_queries:
        sel = np.random.default_rng(cfg.seed).choice(len(caps), cfg.max_queries, replace=False)
        caps = [caps[i] for i in sel]
        img_of_cap = [img_of_cap[i] for i in sel]
    rprint(f"\n{cfg.dataset_path}: {len(ds)} images, {len(caps)} caption queries")

    @torch.no_grad()
    def encode():
        ims = []
        for a in range(0, len(ds), cfg.batch_size):
            b = torch.stack([preprocess(ds[j]["image"].convert("RGB"))
                             for j in range(a, min(a + cfg.batch_size, len(ds)))]).to(device)
            ims.append(torch.nn.functional.normalize(model.encode_image(b).float(), dim=-1).cpu())
        txt = []
        for a in range(0, len(caps), cfg.batch_size):
            t = tokenizer(caps[a:a + cfg.batch_size]).to(device)
            txt.append(torch.nn.functional.normalize(model.encode_text(t).float(), dim=-1).cpu())
        return torch.cat(ims).numpy(), torch.cat(txt).numpy()

    rprint("\nFP encoding:")
    I_fp, T_fp = encode()
    n = apply_ptq_(model=model, bits=cfg.ptq.bits, granularity=cfg.ptq.granularity,
                   skip_modules=frozenset(cfg.ptq.skip_modules))
    rprint(f"\nPTQ: bits={cfg.ptq.bits}, gran={cfg.ptq.granularity}; quantised {len(n)} layers")
    rprint("\nPTQ encoding:")
    I_q, T_q = encode()

    s_fp, s_q = T_fp @ I_fp.T, T_q @ I_q.T
    k = min(K_GAPS, s_fp.shape[1] - 1)
    gf, idf = _profile(s_fp, k)
    gq, idq = _profile(s_q, k)
    eps = np.abs(s_q - s_fp).max(axis=1)

    gold = np.asarray(img_of_cap)
    flip = (idq[:, 0] != idf[:, 0])
    r1_fp = float((idf[:, 0] == gold).mean())
    r1_q = float((idq[:, 0] == gold).mean())
    had = (idf[:, 0] == gold)
    lost = float((had & (idq[:, 0] != gold)).sum() / max(had.sum(), 1))
    rprint(f"\n  top-1 retrieval flip rate  : {flip.mean():.2%}")
    rprint(f"  Recall@1  FP {r1_fp:.2%} -> PTQ {r1_q:.2%}  ({(r1_q-r1_fp)/max(r1_fp,1e-9)*100:+.1f}% rel)")
    rprint(f"  gold image lost from top-1 : {lost:.2%} of queries that had it")

    df = pd.DataFrame({"query_idx": np.arange(len(caps)), "eps_linf": eps.astype(np.float32),
                       "condition": "both_quantized", "corpus_size": len(ds)})
    for j in range(k):
        df[f"fp_gap_{j+1}"] = gf[:, j].astype(np.float32)
        df[f"q_gap_{j+1}"] = gq[:, j].astype(np.float32)
        df[f"fp_cls_{j+1}"] = idf[:, j].astype(np.int64)
        df[f"q_cls_{j+1}"] = idq[:, j].astype(np.int64)
    out = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision" / "clip_crossmodal" / \
        cfg.model_name.replace("/", "_") / cfg.dataset_path.replace("/", "_") / \
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"seed={cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "crossmodal_gap_profile.parquet", index=False)
    (out / "metadata.json").write_text(json.dumps(
        {"model": cfg.model_name, "pretrained": cfg.pretrained, "dataset": cfg.dataset_path,
         "n_images": len(ds), "n_queries": len(caps), "top1_flip": float(flip.mean()),
         "recall1_fp": r1_fp, "recall1_ptq": r1_q, "gold_lost_top1": lost,
         "ptq": OmegaConf.to_container(cfg.ptq, resolve=True)}, indent=2))
    rprint(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
