"""Gap-driven bit allocation on CLIP text-to-image retrieval.

The intervention was established on bi-encoder text retrieval. CLIP text->image is the paper's
other retrieval setting -- a shipped cross-modal system, a different architecture family, no
finetuning -- so if the allocation is a property of rankings rather than of one encoder family,
it should carry over here. Same criterion, same baselines, same pooled capture metric.
"""
import copy, json, os, sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
from dotenv import load_dotenv
load_dotenv(_R / ".env")

import hydra, numpy as np, torch
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from src.allocation import (gap_sensitivity, mse_sensitivity, allocate_bits,
                            apply_bit_allocation_, _linears)
from src.gptq import apply_gptq_
from src.awq import apply_awq_


@hydra.main(config_path="../../../../../config/experiments/vision/ilharco_hf_clip/004_input_fragility",
            config_name="gap_allocation_clip", version_base=None)
def main(cfg: DictConfig):
    rprint(OmegaConf.to_container(cfg, resolve=True))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model.eval()
    fp_state = copy.deepcopy(model.state_dict())

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
            caps.append(c); img_of_cap.append(i)
    if cfg.max_queries and len(caps) > cfg.max_queries:
        sel = np.random.default_rng(cfg.seed).choice(len(caps), cfg.max_queries, replace=False)
        caps = [caps[i] for i in sel]
        img_of_cap = [img_of_cap[i] for i in sel]
    gold = np.asarray(img_of_cap)
    rprint(f"\n{cfg.dataset_path}: {len(ds)} images, {len(caps)} caption queries")

    # Preprocess ONCE. PIL decode+resize is single-threaded CPU work, and gap sensitivity
    # re-encodes the calibration slice once per layer -- without this cache the GPU sits idle
    # behind ~70 layers x 512 redundant image preprocesses per run.
    rprint("preprocessing images once:")
    IMG = torch.stack([preprocess(ds[j]["image"].convert("RGB")) for j in range(len(ds))])
    TOK = tokenizer(caps)

    @torch.no_grad()
    def _encode(images=True, texts=True, n_img=None, n_txt=None):
        ims = txt = None
        if images:
            lim = n_img if n_img is not None else len(ds)
            out = []
            for a in range(0, lim, cfg.batch_size):
                b = IMG[a:min(a + cfg.batch_size, lim)].to(device, non_blocking=True)
                out.append(torch.nn.functional.normalize(model.encode_image(b).float(), dim=-1).cpu())
            ims = torch.cat(out).numpy()
        if texts:
            lim = n_txt if n_txt is not None else len(caps)
            out = []
            for a in range(0, lim, cfg.batch_size):
                t = TOK[a:min(a + cfg.batch_size, lim)].to(device, non_blocking=True)
                out.append(torch.nn.functional.normalize(model.encode_text(t).float(), dim=-1).cpu())
            txt = torch.cat(out).numpy()
        return ims, txt

    rprint("\nFP encoding:")
    I_fp, T_fp = _encode()
    s_fp = T_fp @ I_fp.T
    fp_top1 = s_fp.argmax(1)
    r1_fp = float((fp_top1 == gold).mean())
    rprint(f"  FP Recall@1 = {r1_fp:.2%}")

    # Calibration slice for the sensitivity measurement: a subset of captions against a subset
    # of images. Both towers are quantized, so both must be re-encoded when a layer changes.
    n_ci, n_ct = min(cfg.calib_images, len(ds)), min(cfg.calib_queries, len(caps))

    def score_fn():
        i_c, t_c = _encode(n_img=n_ci, n_txt=n_ct)
        return t_c @ i_c.T

    skip = frozenset(cfg.ptq.skip_modules)
    numel = {n: m.weight.numel() for n, m in _linears(model, skip)}
    gran = cfg.ptq.granularity
    rprint(f"\nmeasuring gap sensitivity over {len(numel)} layers "
           f"({n_ct} calibration queries, {n_ci} images):")
    g_sens = gap_sensitivity(model, skip, cfg.ptq.bits, gran, score_fn)
    m_sens = mse_sensitivity(model, skip, cfg.ptq.bits, gran)
    from scipy import stats
    common = [k for k in g_sens if k in m_sens]
    rho, pv = stats.spearmanr([g_sens[k] for k in common], [m_sens[k] for k in common])
    rprint(f"  Spearman(gap, mse) over layers = {rho:+.3f} (p={pv:.2g})")

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

    method = str(getattr(cfg.ptq, "method", "rtn")).lower()
    res = {}
    for name, alloc in allocs.items():
        model.load_state_dict(fp_state)
        if method == "rtn":
            n = apply_bit_allocation_(model, alloc, gran)
        else:
            fn = apply_gptq_ if method == "gptq" else apply_awq_
            cal = [IMG[a:min(a + cfg.batch_size, n_ci)]
                   for a in range(0, n_ci, cfg.batch_size)]
            n = 0
            for b in sorted(set(alloc.values())):
                only = frozenset(k for k, v in alloc.items() if v == b)
                n += len(fn(model=model, bits=b, granularity=gran, skip_modules=skip,
                            only=only, calib_batches=cal,
                            forward_fn=lambda m, x: m.encode_image(x.to(device)),
                            verbose=False))
        I_q, T_q = _encode()
        s_q = T_q @ I_q.T
        q_top1 = s_q.argmax(1)
        flip = float((q_top1 != fp_top1).mean())
        r1_q = float((q_top1 == gold).mean())
        had = fp_top1 == gold
        lost = float((had & (q_top1 != gold)).sum() / max(had.sum(), 1))
        bits = sum(numel[k] * alloc[k] for k in numel) / sum(numel.values())
        res[name] = dict(flip=flip, recall1=r1_q, recall1_fp=r1_fp, gold_lost=lost,
                         median_eps=float(np.median(np.abs(s_q - s_fp).max(1))),
                         avg_bits=bits, n_layers=n)
        rprint(f"  {name:<8} flip {flip:6.2%}   R@1 {r1_q:6.2%}   gold lost {lost:6.2%}   "
               f"avg bits {bits:.4f}")
    model.load_state_dict(fp_state)

    out = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision" / "ilharco_hf_clip" / \
        "gap_allocation_clip" / cfg.model_name.replace("/", "_") / \
        f"bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}" / f"method={method}" / \
        f"avg={float(target):g}" / f"seed={cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_allocation_clip.json").write_text(json.dumps(
        dict(model_name=cfg.model_name, dataset_name=cfg.dataset_path, task="crossmodal_retrieval",
             bits=cfg.ptq.bits, granularity=cfg.ptq.granularity, bit_choices=choices,
             method=method, seed=int(cfg.seed), avg_bits_target=target,
             n_images=int(len(ds)), n_queries=int(len(caps)),
             layer_rho=float(rho), results=res,
             alloc_gap={k: int(v) for k, v in allocs["gap"].items()}), indent=2))
    rprint(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
