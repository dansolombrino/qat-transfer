"""Aggregate the layer-wise CKA sweeps and compare PTQ configurations.

CKA near 1 means quantization left the representation essentially intact (the
premise of the margin bound holds); a collapsing curve means the quantized network
has stopped being a perturbation of the FP one.
"""
import argparse, json, os, sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_R / "code")); os.chdir(_R)
from dotenv import load_dotenv; load_dotenv(_R / ".env")
import numpy as np
from src.vision.utils import sanitize_timm_model_name

def load(model, bs, bits, gran, lr="1e-05", wd="0.1", ls="0.0", wl="500", mgn="1.0", seed="2038"):
    base = (Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision" / "ilharco_timm_supervised"
            / "layerwise_cka" / sanitize_timm_model_name(model))
    if not base.exists(): return {}
    out = {}
    for ds in sorted(base.iterdir()):
        p = (ds / f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={mgn}_bs={bs}"
             / f"ptq=bits={bits}_gran={gran}_skip=head" / f"seed={seed}" / "cka_results.json")
        if p.exists(): out[ds.name] = json.loads(p.read_text())
    return out

CONFIGS = [("vit_base_patch16_224.orig_in21k","128",4,"channel",  "ViT-B W4-channel"),
           ("vit_base_patch16_224.orig_in21k","128",3,"channel",  "ViT-B W3-channel"),
           ("vit_base_patch16_224.orig_in21k","128",3,"group_128","ViT-B W3-group128"),
           ("vit_base_patch16_224.orig_in21k","128",4,"group_128","ViT-B W4-group128"),
           ("vit_large_patch16_224.orig_in21k","64",4,"channel",  "ViT-L W4-channel"),
           ("vit_large_patch16_224.orig_in21k","64",3,"channel",  "ViT-L W3-channel")]

rows = []
for m, bs, bits, gran, label in CONFIGS:
    d = load(m, bs, bits, gran)
    if not d: continue
    cka = np.array([v["cka_per_block"] for v in d.values()], dtype=float)
    agr = np.array([v["top1_agree_per_block"] for v in d.values()], dtype=float)
    rows.append((label, len(d), cka.mean(0), agr.mean(0)))

print(f"{'config':<20}{'n':>4}  {'CKA b0':>8}{'CKA mid':>9}{'CKA last':>10}   {'agree last':>11}")
for label, n, cka, agr in rows:
    print(f"  {label:<18}{n:>4}  {cka[0]:>8.3f}{cka[len(cka)//2]:>9.3f}{cka[-1]:>10.3f}   {agr[-1]*100:>10.1f}%")

for label, n, cka, agr in rows:
    print(f"\n  {label} ({n} tasks) CKA by block:")
    print("    " + " ".join(f"{c:.3f}" for c in cka))
