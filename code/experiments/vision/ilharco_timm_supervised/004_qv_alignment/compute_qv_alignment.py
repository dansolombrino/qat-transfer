"""004 — Donor/receiver quantization-vector similarity (timm supervised)

Reviewer 3HFP's central methodological objection is that Proposition 1's
cos^2-alignment law is the paper's main theoretical claim about *when* transfer
works, and the paper never measures the quantity it is about:

    "the paper never directly measures the cosine similarity between donor and
    receiver QVs and tests whether it predicts observed transfer gain
    Delta(D,R).  [...]  Does it correlate with observed Delta(D,R) across the
    22x22 vision matrix?"

This script computes that similarity for every donor-receiver pair, at three
granularities and under any of seven metrics, and writes it as JSON.  It does not
touch accuracy: the join against Delta(D,R) -- which
998_rebuttal/001_zero_shot_reframing already computed and wrote -- happens in
aggregate_qv_alignment.py, so no accuracy quantity is ever recomputed here.

What is compared
----------------
For each dataset D, the quantization vector is the displacement QAT_D - FP_D,
the same object 001_qat_transfer/qv_transfer.py builds and patches with.  The
similarity is measured over exactly the weights `apply_ptq_` fake-quantizes:
the classification head is excluded (as it is from the patch), and so are
LayerNorm gains and the patch-embedding kernel, which are never quantized and
therefore cannot bear on quantization robustness.  See qv_alignment_common.py
for the argument, and for why the three granularities are three different
questions rather than three estimates of one.

Every metric and every aggregation operator is selected from the CLI.  Nothing is
hardcoded, because which aggregation is defensible is exactly what is at issue --
a mean over neurons, a norm-weighted mean over layers and a single flattened
cosine are three different claims about the same pair of vectors.

The full 22 x 22 matrix costs one pass
--------------------------------------
Every metric except CKA is a function of the pairwise inner products and norms,
and those are additive over any partition of the coordinates.  So one pass over
the quantized weights, computing an (N, N) Gram per output row, serves all N^2
pairs and all three granularities simultaneously: rows sum exactly to layers and
layers sum exactly to the model.  The cost is N^2 * (parameter count) multiply-
accumulates -- about 41 GFLOP for ViT-B/16 across the whole matrix, seconds on a
CPU.  What actually constrains this script is memory, not arithmetic: it holds
all N quantization vectors at once, which is 3.7 GB for ViT-B/16, ~13 GB for
ViT-L and ~28 GB for ViT-H at float32.  --cache-dtype float16 halves that for the
largest backbone, with all inner products still accumulated in float32; the
diagonal deviations written to the output are what makes the resulting precision
loss visible rather than assumed.

Deviations from the repo conventions, both deliberate
-----------------------------------------------------
This is an analysis script that reads CHECKPOINT_BASE_PATH, which CLAUDE.md says
analysis scripts never do.  It still builds no model, runs no forward pass, needs
no dataset and no GPU, so it keeps the property that rule exists to protect --
but it is consequently *not* runnable in a code-only clone.
998_rebuttal/004_quantization_mechanism set this precedent.

The output path uses an `alignment/` leaf where 001's grammar has
`split={val,test}`.  A weight-space quantity does not depend on an evaluation
split, and inventing one would claim a provenance these numbers do not have.

Just above that leaf the path carries an `agg=` fragment naming the requested
metrics, granularities, neuron modes and operators.  The checkpoint fragments
say which vectors were compared; this one says how they were reduced to a
number, and since the aggregations demonstrably disagree with each other, a file
that does not record its own is not interpretable.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

# The quantized-key predicate is imported from 998_rebuttal/004 rather than
# copied.  A copy would drift, and drift in this predicate silently changes which
# tensors every number in this file describes.  That sub-phase verified it
# against the real `apply_ptq_` module list via its --verify-modules flag.
_QUANT_MECHANISM_DIR = _CODE_DIR / "experiments" / "998_rebuttal" / "004_quantization_mechanism"
if str(_QUANT_MECHANISM_DIR) not in sys.path:
    sys.path.insert(0, str(_QUANT_MECHANISM_DIR))

os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import torch

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name

from quant_mechanism_common import select_quantized_keys

from qv_alignment_common import (
    DIAGONAL_TOL,
    GRANULARITIES,
    GRANULARITY_LAYER,
    GRANULARITY_MODEL,
    GRANULARITY_NEURON,
    IDENTITY_VALUE,
    METRICS,
    METRICS_LAYER_ONLY,
    METRICS_NEEDING_CENTERED,
    METRICS_NEEDING_SIGN,
    METRIC_CKA,
    NEURON_MODES,
    NEURON_MODE_POOLED,
    NEURON_MODE_TWO_STAGE,
    OPERATORS,
    aggregate,
    cka_layer,
    metric_values,
    norm_pair_weights,
    reduce_to_unit,
    row_accumulators,
    stack_operator_values,
    to_float,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_timm_supervised"
MODALITY = "vision"

EVAL_ROOT_OUT = "evaluations/vision/ilharco_timm_supervised/004_qv_alignment/vision/qv_alignment"

OUT_FILE = "qv_alignment.json"

# Keys of the classification head, excluded from the QV by 001's qv_transfer.py.
# The head is additionally excluded by skip_modules, so this is belt and braces.
HEAD_PREFIX = "model.head."

CACHE_DTYPES = {"float32": torch.float32, "float16": torch.float16}

MODEL_DISPLAY_NAMES = {
    "vit_base_patch16_224.orig_in21k": "ViT-B/16",
    "vit_large_patch16_224.orig_in21k": "ViT-L/16",
    "vit_huge_patch14_224.orig_in21k": "ViT-H/14",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16 (AugReg2)",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k": "ViT-B/16 (CLIP)",
    "vit_large_patch14_clip_224.openai_ft_in12k_in1k": "ViT-L/14 (CLIP)",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B/16",
    "deit3_base_patch16_224.fb_in22k_ft_in1k": "DeiT3-B/16 (IN22k)",
    "deit3_large_patch16_224.fb_in1k": "DeiT3-L/16",
    "deit3_large_patch16_224.fb_in22k_ft_in1k": "DeiT3-L/16 (IN22k)",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k": "Swin-L",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",    required=True, nargs="+",
                        help="timm model names, e.g. vit_base_patch16_224.orig_in21k")
    parser.add_argument("--seed",           required=True, type=int)

    parser.add_argument("--optim",          required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",             required=True, type=float)
    parser.add_argument("--wd",             required=True, type=float)
    parser.add_argument("--ls",             required=True, type=float)
    parser.add_argument("--wl",             required=True, type=int)
    parser.add_argument("--max-grad-norm",  required=True, type=float)
    parser.add_argument("--batch-sizes",    required=True, type=int, nargs="+",
                        help="Batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",       required=True, type=int,
                        help="Bit width of the QAT run the QV is differenced from. "
                             "Selects the checkpoint; quantizes nothing here.")
    parser.add_argument("--ptq-bits",       required=True, type=int,
                        help="Quantizes nothing here either. Carried so the output "
                             "path grammar matches 001 and 003, which is what lets "
                             "aggregate_qv_alignment.py join on the directory alone.")
    parser.add_argument("--granularity",    required=True, choices=["tensor", "channel"],
                        help="The QUANTIZER's granularity, part of the checkpoint path. "
                             "Not to be confused with --granularities below.")
    parser.add_argument("--skip-modules",   required=True, nargs="+",
                        help="Module names skipped during quantization "
                             "(no default: must be specified explicitly). Defines "
                             "which weights the similarity is measured over.")

    parser.add_argument("--metrics",        required=True, nargs="+", choices=list(METRICS),
                        help="Similarity metrics to emit. See qv_alignment_common.py "
                             "for what each one can see that the others cannot.")
    parser.add_argument("--granularities",  required=True, nargs="+", choices=list(GRANULARITIES),
                        help="AGGREGATION levels: neuron (per output row), layer "
                             "(per quantized weight tensor), model (everything "
                             "flattened). Not the quantizer's --granularity.")
    parser.add_argument("--neuron-modes",   default=None, nargs="+", choices=list(NEURON_MODES),
                        help="Required when 'neuron' is in --granularities. two_stage "
                             "aggregates neurons within a layer then layers; pooled "
                             "aggregates every neuron in the model at once.")
    parser.add_argument("--operators",      required=True, nargs="+", choices=list(OPERATORS),
                        help="How per-unit similarities are reduced to one number. "
                             "Applied at every aggregation step.")

    parser.add_argument("--donors",         default=None, nargs="+",
                        help="Donor datasets. Default: every dataset with checkpoints present.")
    parser.add_argument("--receivers",      default=None, nargs="+",
                        help="Receiver datasets. Default: every dataset with checkpoints present.")

    parser.add_argument("--per-layer",      action="store_true",
                        help="Also emit per-layer values. Multiplies output size by "
                             "the layer count for every pair and metric.")
    parser.add_argument("--out-name",       default=OUT_FILE,
                        help="Output filename inside the alignment/ leaf. Rarely "
                             "needed: the path now carries an agg= fragment naming "
                             "the requested metrics, granularities, neuron modes and "
                             "operators, so two runs differing in what they asked for "
                             "already land in different directories. Reach for this "
                             "only to keep two otherwise-identical runs apart.")
    parser.add_argument("--cache-dtype",    default="float32", choices=list(CACHE_DTYPES),
                        help="Storage dtype for the in-memory QV cache. Inner products "
                             "are accumulated in float32 regardless. float16 halves "
                             "the ~28 GB ViT-H footprint at a cost visible in the "
                             "reported diagonal deviations.")

    args = parser.parse_args()

    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")

    # Checked before --neuron-modes so that `--metrics cka --granularities neuron`
    # reports the reason it is actually wrong, not a missing companion argument.
    layer_only = METRICS_LAYER_ONLY & set(args.metrics)
    bad = [g for g in args.granularities if g != GRANULARITY_LAYER]
    if layer_only and len(args.granularities) > 1:
        parser.error(
            f"{sorted(layer_only)} are defined at layer granularity only "
            f"(a single neuron is a vector, and a model-wise version would have to "
            f"concatenate layers of differing row counts), but --granularities also "
            f"requests {bad}. Run them as separate invocations."
        )
    if layer_only and GRANULARITY_LAYER not in args.granularities:
        parser.error(f"{sorted(layer_only)} require --granularities layer")

    if GRANULARITY_NEURON in args.granularities and not args.neuron_modes:
        parser.error("--neuron-modes is required when 'neuron' is in --granularities")

    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _agg_frag(args):
    """The aggregation request set, as a path fragment.

    The checkpoint fragments above identify *which vectors* were compared; this
    one identifies *how they were reduced to a number*, and the two are
    independent.  Without it a run asking for one metric at one granularity
    lands on top of a run asking for all six at all three, and the surviving
    file's name says nothing about which one it is.  That is not hypothetical:
    `cka` is usually run separately, since it is the one metric restricted to
    layer granularity.

    One file holds every variant it was asked for, so the fragment describes the
    whole requested *set* rather than a single (metric, granularity, mode,
    operator) cell -- the per-figure form of that lives in the visualization
    scripts.  Each component is sorted so the fragment is a function of the set
    and not of CLI argument order.

    `modes=` is omitted entirely when `neuron` was not requested, because
    --neuron-modes is then meaningless rather than empty.
    """
    parts = [
        f"metrics={'-'.join(sorted(args.metrics))}",
        f"grans={'-'.join(sorted(args.granularities))}",
    ]
    if GRANULARITY_NEURON in args.granularities:
        parts.append(f"modes={'-'.join(sorted(args.neuron_modes))}")
    parts.append(f"ops={'-'.join(sorted(args.operators))}")
    return "agg=" + "_".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
def _fp_ckpt_path(args, model_dir, optim_frag, dataset, epochs):
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision", FAMILY, "fp", model_dir, dataset,
        optim_frag, f"seed={args.seed}",
        f"classifier_epoch_{epochs}.pt",
    )


def _qat_ckpt_path(args, model_dir, optim_frag, dataset, epochs):
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision", FAMILY, "qat", model_dir, dataset,
        optim_frag, _qat_frag(args), f"seed={args.seed}",
        f"classifier_epoch_{epochs}.pt",
    )


def _load(path):
    """Load a checkpoint, or None if it is not on disk or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
def _out_dir(args, model_dir, optim_frag):
    """Mirrors 001's qv_transfer grammar, with `alignment` in place of split=.

    The doubled modality segment in EVAL_ROOT_OUT is redundant but load-bearing
    for every existing path in the project, so it is emitted here too.

    The `agg=` level sits between the quantization fragments and the `alignment`
    leaf, so the leaf keeps the slot the sibling phases give to `split=`.
    """
    return os.path.join(
        EVAL_ROOT_OUT, model_dir,
        f"seed={args.seed}",
        optim_frag,
        _qat_frag(args),
        _ptq_frag(args),
        _agg_frag(args),
        "alignment",
    )


# ---------------------------------------------------------------------------
# Quantization-vector cache
# ---------------------------------------------------------------------------
def build_qv_cache(args, model_dir, optim_frag, datasets, cache_dtype, missing):
    """QAT_D - FP_D over the quantized weights, for every dataset, held in RAM.

    Every pair needs both of its vectors, and the single-pass Gram over all N^2
    pairs needs all N of them simultaneously, so there is no streaming variant
    that would not reload each checkpoint O(N) times.  The estimated footprint is
    printed before the first allocation so an impossible run fails visibly rather
    than after twenty minutes of loading.

    Datasets whose FP or QAT checkpoint is absent are dropped and recorded in
    `missing`; a partial matrix is more useful than no matrix, provided the
    absence is stated.
    """
    cache = {}
    keys = None

    for dataset in datasets:
        epochs = DATASET_NAME_TO_EPOCHS[dataset]
        fp_path  = _fp_ckpt_path(args, model_dir, optim_frag, dataset, epochs)
        qat_path = _qat_ckpt_path(args, model_dir, optim_frag, dataset, epochs)

        fp_sd = _load(fp_path)
        if fp_sd is None:
            missing.append(fp_path)
            continue

        qat_sd = _load(qat_path)
        if qat_sd is None:
            missing.append(qat_path)
            del fp_sd
            continue

        if keys is None:
            keys = select_quantized_keys(fp_sd, frozenset(args.skip_modules))
            keys = [k for k in keys if not k.startswith(HEAD_PREFIX)]

            n_params = sum(fp_sd[k].numel() for k in keys)
            gib = n_params * len(datasets) * cache_dtype.itemsize / 1024 ** 3
            print(f"  {len(keys)} quantized weights, {n_params/1e6:.1f}M params each; "
                  f"cache ~{gib:.1f} GiB at {cache_dtype}")

        absent = [k for k in keys if k not in qat_sd]
        if absent:
            print(f"  [SKIP] {dataset}: {len(absent)} keys absent from the QAT "
                  f"checkpoint, first is {absent[0]}", file=sys.stderr)
            missing.append(qat_path)
            del fp_sd, qat_sd
            continue

        cache[dataset] = {
            k: (qat_sd[k].detach().to(torch.float32) - fp_sd[k].detach().to(torch.float32))
                .to(cache_dtype)
            for k in keys
        }

        del fp_sd, qat_sd

    return cache, keys


# ---------------------------------------------------------------------------
# The single pass
# ---------------------------------------------------------------------------
def compute_model(args, cache, keys, names):
    """Every requested metric, granularity and operator, in one pass over `keys`.

    Returns (results, diagonal), where `results` is keyed
    [metric][granularity][...] with (N, N) tensors at the leaves and `diagonal`
    records, per metric and granularity, how far the donor == receiver cells sat
    from their algebraic identity value.
    """
    metrics    = list(args.metrics)
    operators  = list(args.operators)
    grans      = set(args.granularities)
    modes      = set(args.neuron_modes or [])

    need_centered = bool(METRICS_NEEDING_CENTERED & set(metrics))
    need_sign     = bool(METRICS_NEEDING_SIGN & set(metrics))
    vector_metrics = [m for m in metrics if m not in METRICS_LAYER_ONLY]

    n = len(names)

    # Layer-level per-key accumulations, reused by the layer granularity, by the
    # second stage of the neuron two-stage aggregation, and by the model total.
    layer_dot, layer_sq, layer_total, layer_sdot, layer_numel = [], [], [], [], []

    # Row-level accumulators, retained only if the pooled neuron mode is on.
    pooled_dot, pooled_sq, pooled_total, pooled_sdot, pooled_numel = [], [], [], [], []

    # Stage-1 results of the two-stage neuron aggregation: one dict per key.
    two_stage = {m: [] for m in vector_metrics}
    per_layer_neuron = {m: {} for m in vector_metrics}

    cka_values, cka_weights = [], []

    diagonal = {m: {} for m in metrics}

    def _track_diagonal(metric, granularity, values):
        """Largest deviation of the donor == receiver cells from their identity.

        See qv_alignment_common.py: this is a numerical self-check on the
        accumulators and on --cache-dtype, not evidence about checkpoint
        provenance -- the diagonal is 1 by algebra, not by measurement.
        """
        target = IDENTITY_VALUE[metric]
        if target is None:
            return
        diag = values[..., torch.arange(n), torch.arange(n)]
        dev = float((diag - target).abs().max())
        prev = diagonal[metric].get(granularity)
        diagonal[metric][granularity] = dev if prev is None else max(prev, dev)

    for key in keys:
        stacked = torch.stack([cache[d][key] for d in names])
        rows, cols = stacked.shape[1], stacked.shape[2]

        dot, sq, total, sdot = row_accumulators(stacked, need_centered, need_sign)
        l_dot, l_sq, l_total, l_sdot = reduce_to_unit(dot, sq, total, sdot)

        layer_dot.append(l_dot)
        layer_sq.append(l_sq)
        layer_total.append(l_total)
        layer_sdot.append(l_sdot)
        layer_numel.append(rows * cols)

        if GRANULARITY_NEURON in grans:
            # sq/total arrive as (N, rows); the metric helpers want the unit axis
            # first, so every per-row quantity is transposed to (rows, N).
            sq_r    = sq.transpose(0, 1)
            total_r = total.transpose(0, 1) if total is not None else None

            if NEURON_MODE_POOLED in modes:
                pooled_dot.append(dot)
                pooled_sq.append(sq_r)
                pooled_total.append(total_r)
                pooled_sdot.append(sdot)
                pooled_numel.append(torch.full((rows,), float(cols)))

            if NEURON_MODE_TWO_STAGE in modes:
                w_params = torch.full((rows,), float(cols))
                w_norm   = norm_pair_weights(sq_r)

                for m in vector_metrics:
                    values = metric_values(m, dot, sq_r, total_r, sdot, cols)
                    _track_diagonal(m, GRANULARITY_NEURON, values)
                    agg = aggregate(values, w_params, w_norm, operators)
                    two_stage[m].append(agg)
                    if args.per_layer:
                        per_layer_neuron[m][key] = agg
            else:
                # pooled-only: the diagonal still has to be checked at row level.
                for m in vector_metrics:
                    _track_diagonal(
                        m, GRANULARITY_NEURON,
                        metric_values(m, dot, sq_r, total_r, sdot, cols),
                    )

        if METRIC_CKA in metrics:
            values, weights = cka_layer(stacked)
            _track_diagonal(METRIC_CKA, GRANULARITY_LAYER, values)
            cka_values.append(values)
            cka_weights.append(weights)

        del stacked, dot, sq, total, sdot

    results = {m: {} for m in metrics}

    # ---- layer granularity, and the model total ---------------------------
    k_dot   = torch.stack(layer_dot)                                        # (K, N, N)
    k_sq    = torch.stack(layer_sq)                                         # (K, N)
    k_total = torch.stack(layer_total) if need_centered else None
    k_sdot  = torch.stack(layer_sdot) if need_sign else None
    k_numel = torch.tensor([float(x) for x in layer_numel])                 # (K,)
    k_norm_w = norm_pair_weights(k_sq)                                      # (K, N, N)

    m_dot   = k_dot.sum(dim=0)
    m_sq    = k_sq.sum(dim=0)
    m_total = k_total.sum(dim=0) if need_centered else None
    m_sdot  = k_sdot.sum(dim=0) if need_sign else None
    m_numel = float(k_numel.sum())

    for m in vector_metrics:
        if GRANULARITY_LAYER in grans:
            values = metric_values(m, k_dot, k_sq, k_total, k_sdot, k_numel)
            _track_diagonal(m, GRANULARITY_LAYER, values)
            block = {"agg": aggregate(values, k_numel, k_norm_w, operators)}
            if args.per_layer:
                block["per_layer"] = {keys[i]: values[i] for i in range(len(keys))}
            results[m][GRANULARITY_LAYER] = block

        if GRANULARITY_MODEL in grans:
            values = metric_values(m, m_dot, m_sq, m_total, m_sdot, m_numel)
            _track_diagonal(m, GRANULARITY_MODEL, values)
            results[m][GRANULARITY_MODEL] = values

    # ---- neuron granularity ------------------------------------------------
    if GRANULARITY_NEURON in grans:
        for m in vector_metrics:
            block = {}

            if NEURON_MODE_TWO_STAGE in modes:
                stage1 = stack_operator_values(two_stage[m], operators)
                block[NEURON_MODE_TWO_STAGE] = {
                    op: aggregate(stage1[op], k_numel, k_norm_w, [op])[op]
                    for op in operators
                }
                if args.per_layer:
                    block["per_layer"] = per_layer_neuron[m]

            if NEURON_MODE_POOLED in modes:
                p_dot   = torch.cat(pooled_dot)
                p_sq    = torch.cat(pooled_sq)
                p_total = torch.cat(pooled_total) if need_centered else None
                p_sdot  = torch.cat(pooled_sdot) if need_sign else None
                p_numel = torch.cat(pooled_numel)

                values = metric_values(m, p_dot, p_sq, p_total, p_sdot, p_numel)
                block[NEURON_MODE_POOLED] = aggregate(
                    values, p_numel, norm_pair_weights(p_sq), operators
                )

            results[m][GRANULARITY_NEURON] = block

    # ---- CKA ---------------------------------------------------------------
    if METRIC_CKA in metrics:
        values = torch.stack(cka_values)
        block = {"agg": aggregate(values, k_numel, torch.stack(cka_weights), operators)}
        if args.per_layer:
            block["per_layer"] = {keys[i]: values[i] for i in range(len(keys))}
        results[METRIC_CKA][GRANULARITY_LAYER] = block

    return results, diagonal


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def _pair_entry(results, names, i, j, norms):
    """One donor-receiver cell, pulled out of the (N, N) result matrices."""
    donor, receiver = names[i], names[j]

    entry = {
        "donor":     donor,
        "receiver":  receiver,
        "same_task": donor == receiver,
        # A similarity with no norm beside it cannot distinguish "right
        # direction, wrong magnitude" from "right direction and comparable
        # magnitude", which is the overshoot-versus-anti-alignment question 001
        # raises and cannot answer.
        "norm_donor":    norms[donor],
        "norm_receiver": norms[receiver],
        "norm_ratio":    norms[donor] / norms[receiver] if norms[receiver] else None,
        "metrics":       {},
    }

    for metric, by_gran in results.items():
        block = {}

        if GRANULARITY_MODEL in by_gran:
            block["model_wise"] = to_float(by_gran[GRANULARITY_MODEL][i, j])

        if GRANULARITY_LAYER in by_gran:
            layer = by_gran[GRANULARITY_LAYER]
            out = {"agg": {op: to_float(v[i, j]) for op, v in layer["agg"].items()}}
            if "per_layer" in layer:
                out["per_layer"] = {k: to_float(v[i, j]) for k, v in layer["per_layer"].items()}
            block["layer_wise"] = out

        if GRANULARITY_NEURON in by_gran:
            neuron = by_gran[GRANULARITY_NEURON]
            out = {}
            for mode in (NEURON_MODE_TWO_STAGE, NEURON_MODE_POOLED):
                if mode in neuron:
                    out[mode] = {op: to_float(v[i, j]) for op, v in neuron[mode].items()}
            if "per_layer" in neuron:
                out["per_layer"] = {
                    k: {op: to_float(v[i, j]) for op, v in aggs.items()}
                    for k, aggs in neuron["per_layer"].items()
                }
            block["neuron_wise"] = out

        entry["metrics"][metric] = block

    return entry


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    all_datasets = sorted(DATASET_NAME_TO_EPOCHS.keys())
    donors    = args.donors    if args.donors    is not None else all_datasets
    receivers = args.receivers if args.receivers is not None else all_datasets

    unknown = sorted((set(donors) | set(receivers)) - set(all_datasets))
    if unknown:
        sys.exit(f"unknown datasets: {unknown}")

    # The Gram is over the union: a cell needs both of its vectors, and a vector
    # loaded for a donor row is the same object the receiver column needs.
    names_requested = sorted(set(donors) | set(receivers))
    cache_dtype = CACHE_DTYPES[args.cache_dtype]

    for model_name, batch_size in zip(args.model_names, args.batch_sizes):
        model_dir  = sanitize_timm_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)
        missing = []

        print(f"\n{model_name}  ({model_dir})")

        cache, keys = build_qv_cache(
            args, model_dir, optim_frag, names_requested, cache_dtype, missing
        )
        if not cache or keys is None:
            print(f"  [SKIP] no checkpoints on disk", file=sys.stderr)
            continue

        names = [d for d in names_requested if d in cache]
        if len(names) < len(names_requested):
            dropped = sorted(set(names_requested) - set(names))
            print(f"  [WARN] {len(dropped)} datasets dropped: {dropped}", file=sys.stderr)

        qv_norms = {
            d: {
                "global":  to_float(torch.stack(
                    [cache[d][k].to(torch.float32).pow(2).sum() for k in keys]
                ).sum().sqrt()),
                "per_key": {
                    k: to_float(cache[d][k].to(torch.float32).norm()) for k in keys
                },
            }
            for d in names
        }

        print(f"  {len(names)} datasets -> {len(names)**2} cells, "
              f"metrics={args.metrics}, granularities={args.granularities}")

        results, diagonal = compute_model(args, cache, keys, names)

        # A diagonal off its algebraic value means the accumulators are wrong or
        # the cache dtype lost too much; either way every off-diagonal cell is
        # suspect, so this is fatal rather than a warning.
        worst = [
            (m, g, dev, DIAGONAL_TOL[m])
            for m, by_gran in diagonal.items()
            for g, dev in by_gran.items()
            if DIAGONAL_TOL[m] is not None and dev > DIAGONAL_TOL[m]
        ]
        if worst:
            for m, g, dev, tol in worst:
                print(f"  [FATAL] {m}/{g}: diagonal deviates by {dev:.3e} "
                      f"(tolerance {tol:.0e})", file=sys.stderr)
            sys.exit(1)

        global_norms = {d: qv_norms[d]["global"] for d in names}
        pairs = [
            _pair_entry(results, names, names.index(d), names.index(r), global_norms)
            for d in donors if d in cache
            for r in receivers if r in cache
        ]

        payload = {
            "model_name":   model_name,
            "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "model_dir":    model_dir,
            "family":       FAMILY,
            "modality":     MODALITY,
            "config": {
                "seed":           args.seed,
                "optim":          args.optim,
                "lr":             args.lr,
                "wd":             args.wd,
                "ls":             args.ls,
                "wl":             args.wl,
                "max_grad_norm":  args.max_grad_norm,
                "batch_size":     batch_size,
                "qat_bits":       args.qat_bits,
                "ptq_bits":       args.ptq_bits,
                "granularity":    args.granularity,
                "skip_modules":   sorted(args.skip_modules),
                "metrics":        list(args.metrics),
                "granularities":  list(args.granularities),
                "neuron_modes":   list(args.neuron_modes or []),
                "operators":      list(args.operators),
                "donors":         [d for d in donors if d in cache],
                "receivers":      [r for r in receivers if r in cache],
                "per_layer":      args.per_layer,
                "cache_dtype":    args.cache_dtype,
                "support":        "quantized_weights_only",
                "diagonal_meaning": (
                    "algebraic identity, so a numerical self-check on the "
                    "accumulators and the cache dtype -- not evidence of "
                    "checkpoint provenance"
                ),
            },
            "datasets":       names,
            "quantized_keys": keys,
            "n_keys":         len(keys),
            "qv_norms":       qv_norms,
            "diagonal":       diagonal,
            "pairs":          pairs,
            "missing":        missing,
        }

        out_dir = _out_dir(args, model_dir, optim_frag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.out_name)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
