"""004 — Shared weight-space similarity math for the QV alignment analysis.

Family-independent helpers imported by compute_qv_alignment.py.  Only path logic
differs between families and it stays in the per-family script, mirroring the
split used by 998_rebuttal's 001, 003 and 004 sub-phases.

The question
------------
Reviewer 3HFP:

    "while Proposition 1's cos^2-alignment prediction is the central theoretical
    claim about when and why transfer works, the paper never directly measures
    the cosine similarity between donor and receiver QVs and tests whether it
    predicts observed transfer gain Delta(D,R).  [...]  Even the Euclidean (H=I)
    cosine similarity between matched donor and receiver QVs would be
    informative.  Does it correlate with observed Delta(D,R) across the 22x22
    vision matrix?"

That is the whole brief, and it is answerable from state dicts alone.  This
module supplies the similarity math; compute_qv_alignment.py supplies the
checkpoints; aggregate_qv_alignment.py performs the correlation against the
Delta values 001_zero_shot_reframing already wrote.

What this can and cannot establish
----------------------------------
Proposition 1's quantity is cos_H, the cosine in the metric induced by the loss
Hessian at the receiver's optimum.  Everything here is Euclidean, H = I.  That is
not an approximation to cos_H, it is a different number: a QV component lying in
a flat direction of the loss contributes to the Euclidean cosine and contributes
nothing to cos_H.  So a strong Euclidean correlation supports the theory, and a
weak one does *not* refute it -- it is equally consistent with the Hessian being
far from isotropic, which is the expected case.  The reviewer asked for the
Euclidean number explicitly, knowing this; the honest framing of a null result is
"the weight-space shadow of the law is not visible without the metric", not "the
law fails".  Computing cos_H needs data, forward passes and a GPU, which would
make this a different script under a different rule.

Support
-------
The similarity is measured over exactly the weights `apply_ptq_` fake-quantizes.
The QV as 001's qv_transfer.py applies it is broader -- every non-head,
non-integer key, including LayerNorm gains and the patch-embedding kernel -- but
those tensors are never quantized, so their contribution to an alignment score
cannot bear on quantization robustness.  Including them would let a large,
perfectly-aligned LayerNorm displacement inflate a cosine that is supposed to be
about the quantized subspace.  The restriction is therefore a claim about what
the number means, not a convenience.

Three granularities, which differ in *when* you flatten
-------------------------------------------------------
    neuron   one output row of a layer against the same row of the other QV.
             Similarities are aggregated over the layer's neurons, then over
             layers.
    layer    the layer flattened into a single long vector.  One similarity per
             layer, aggregated over layers.
    model    every layer flattened and concatenated.  One similarity, no
             aggregation.

These are not three estimates of one quantity.  Flattening is averaging in
disguise: a model-wise cosine is dominated by whichever layers carry the most
QV energy, while a neuron-wise mean gives a 768-wide projection row the same vote
as a 3072-wide FFN row.  When they disagree, the disagreement is the finding --
it localizes alignment rather than summarizing it.

Neuron-wise is emitted twice.  `two_stage` is neurons -> layer -> model, so every
layer counts equally at the final step.  `pooled` puts every neuron in the model
into one set and aggregates once, so a 3072-neuron layer counts four times a
768-neuron one.  They coincide only when layer widths are uniform, which in these
backbones they are not.

The operators, and why a mean is not enough
-------------------------------------------
An operator reduces a set of per-unit similarities to one number, and the choice
is load-bearing rather than cosmetic:

    mean           plain average.  Opposite-signed units cancel, so a model whose
                   neurons are half aligned and half anti-aligned reads the same
                   as one that is uniformly unaligned.
    wmean_params   weighted by unit numel.  Makes a neuron-wise aggregate agree
                   with how much of the model each unit actually is.
    wmean_norm     weighted by ||a_u|| * ||b_u||.  The aggregate is then
                   sum_u <a_u, b_u> / sum_u ||a_u|| ||b_u||, which equals the
                   flattened model-wise cosine exactly when the per-unit norm
                   ratios are constant.  The gap between wmean_norm and the
                   model-wise value is therefore a direct readout of how unevenly
                   the QV's energy is spread, which no single cosine shows.
    median         says whether a mean is being dragged by a tail.
    std, min, max  uniform alignment versus alignment concentrated in a few
                   layers -- itself a likely finding.
    frac_positive  separates "weakly aligned everywhere" from "half aligned, half
                   anti-aligned", the two cases `mean` conflates.

In the two-stage aggregations the *same* operator is applied at both steps, so
`median` means the median over layers of the per-layer medians.  Mixing operators
across steps would multiply the output by 64 for no interpretable gain.

The metrics
-----------
    cosine          <a,b> / ||a|| ||b||.  The reviewer's quantity.
    correlation     cosine after centering each unit.  Removes a shared mean
                    offset, separating "the two QVs point the same way" from
                    "both shifted their unit by a similar constant".
    dot             <a,b>, unnormalized.  Cosine deliberately discards magnitude,
                    and magnitude is what decides whether lambda = 1 overshoots;
                    this keeps it visible.
    normalized_l2   ||a/||a|| - b/||b||||, i.e. sqrt(2 - 2 cos).  Pointwise a
                    monotone restatement of cosine, but it does *not* aggregate to
                    the same place: the mean of a concave function of cosine is
                    not that function of the mean.  Reporting both is how you see
                    whether a conclusion survives the reparameterization.
    sign_agreement  fraction of coordinates where a and b share a sign.  Scale
                    free and immune to a handful of large coordinates, which the
                    cosine is not: in a QV whose energy concentrates in a few
                    weights, cosine can be high while most coordinates disagree.
    angular         arccos(cos)/pi in [0,1].  A true metric, so aggregating it is
                    better behaved than aggregating raw cosines.
    cka             linear CKA between the two layers viewed as matrices,
                    ||b^T a||_F^2 / (||a^T a||_F ||b^T b||_F).  Every other metric
                    treats a layer as a bag of numbers; this one respects the
                    (out, in) structure and is invariant to an orthogonal
                    rotation of the input space, so it can see a shared subspace
                    that a coordinatewise cosine misses.

`cka` is defined at layer granularity only.  A single neuron is a vector, so
neuron-wise CKA is not a thing, and a model-wise CKA would require concatenating
layers with different row counts.  compute_qv_alignment.py rejects those
combinations at the CLI rather than dropping them silently.

What the diagonal check does and does not prove
------------------------------------------------
Every similarity is 1 (or 0, for the distances) when donor == receiver, and it is
so *algebraically*: the same vector is being compared with itself.  The diagonal
is therefore not evidence that the checkpoints differenced here are the ones the
accuracy numbers came from -- unlike 998_rebuttal/004's identity check, which
compares FP + (QAT - FP) against an independently loaded QAT and can genuinely
fail.  Here it is a numerical self-check: it catches an accumulator bug, and it
catches precision loss from --cache-dtype float16.  It is reported honestly as
such.  Provenance is established instead by the resolved checkpoint paths and the
per-QV norms written into the output, which are comparable against 004's.
"""

import math

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRIC_COSINE         = "cosine"
METRIC_CORRELATION    = "correlation"
METRIC_DOT            = "dot"
METRIC_NORMALIZED_L2  = "normalized_l2"
METRIC_SIGN_AGREEMENT = "sign_agreement"
METRIC_ANGULAR        = "angular"
METRIC_CKA            = "cka"

METRICS = (
    METRIC_COSINE,
    METRIC_CORRELATION,
    METRIC_DOT,
    METRIC_NORMALIZED_L2,
    METRIC_SIGN_AGREEMENT,
    METRIC_ANGULAR,
    METRIC_CKA,
)

# Metrics needing the centered accumulators / the sign accumulator.  Kept as sets
# so the per-key pass can skip an einsum nobody asked for.
METRICS_NEEDING_CENTERED = frozenset({METRIC_CORRELATION})
METRICS_NEEDING_SIGN     = frozenset({METRIC_SIGN_AGREEMENT})

# Metrics defined only when a unit has a matrix structure, i.e. at layer level.
METRICS_LAYER_ONLY = frozenset({METRIC_CKA})

GRANULARITY_NEURON = "neuron"
GRANULARITY_LAYER  = "layer"
GRANULARITY_MODEL  = "model"
GRANULARITIES = (GRANULARITY_NEURON, GRANULARITY_LAYER, GRANULARITY_MODEL)

NEURON_MODE_TWO_STAGE = "two_stage"
NEURON_MODE_POOLED    = "pooled"
NEURON_MODES = (NEURON_MODE_TWO_STAGE, NEURON_MODE_POOLED)

OP_MEAN          = "mean"
OP_WMEAN_PARAMS  = "wmean_params"
OP_WMEAN_NORM    = "wmean_norm"
OP_MEDIAN        = "median"
OP_STD           = "std"
OP_MIN           = "min"
OP_MAX           = "max"
OP_FRAC_POSITIVE = "frac_positive"

OPERATORS = (
    OP_MEAN,
    OP_WMEAN_PARAMS,
    OP_WMEAN_NORM,
    OP_MEDIAN,
    OP_STD,
    OP_MIN,
    OP_MAX,
    OP_FRAC_POSITIVE,
)

# Value each metric takes when donor == receiver.  `dot` has none: its diagonal is
# ||rho||^2, which varies per donor, so it is excluded from the diagonal check
# rather than given a fake target.
IDENTITY_VALUE = {
    METRIC_COSINE:         1.0,
    METRIC_CORRELATION:    1.0,
    METRIC_DOT:            None,
    METRIC_NORMALIZED_L2:  0.0,
    METRIC_SIGN_AGREEMENT: 1.0,
    METRIC_ANGULAR:        0.0,
    METRIC_CKA:            1.0,
}

# Per-metric tolerance on the diagonal deviation.  `None` means the diagonal is
# reported but not asserted.
#
# The tolerances are not uniform because the metrics do not respond uniformly to
# the same underlying error.  A float32 inner product over ~10^8 coordinates
# lands its cosine within ~1e-6 of 1, but normalized_l2 = sqrt(2 - 2 cos) maps
# that to sqrt(2e-6) ~ 1.4e-3 and angular = arccos(cos)/pi to ~4.5e-4: the square
# root has an infinite derivative at cos = 1, so both legitimately show three
# decades more deviation than the quantity they are computed from.  Tightening
# them to cosine's tolerance would fail every correct run.  They are still
# asserted, generously, so a gross error trips.
#
# `sign_agreement` is reported and not asserted, because its diagonal is not 1:
# sign(0) = 0, so a coordinate where the QV is exactly zero contributes 0 rather
# than 1 and the self-similarity comes out at 1 - z/2n for a zero fraction z.
# The reported diagonal is therefore a readout of how many quantized weights QAT
# left bit-identical to FP, which is worth having and is not an error.
#
# `dot`'s diagonal is ||rho||^2, which varies per donor, so there is no identity
# value to compare against; the norms are reported separately instead.
DIAGONAL_TOL = {
    METRIC_COSINE:         1e-4,
    METRIC_CORRELATION:    1e-4,
    METRIC_CKA:            1e-4,
    METRIC_NORMALIZED_L2:  2e-2,
    METRIC_ANGULAR:        1e-2,
    METRIC_SIGN_AGREEMENT: None,
    METRIC_DOT:            None,
}

EPS = 1e-12


# ---------------------------------------------------------------------------
# Per-unit accumulators
# ---------------------------------------------------------------------------
def row_accumulators(stacked, need_centered, need_sign):
    """Pairwise accumulators for every output row of one stacked layer.

    `stacked` is (N, rows, cols): the same layer of all N quantization vectors.
    Every metric except CKA is a function of the four quantities returned here,
    which is what lets one pass over the layer serve all N^2 donor-receiver pairs
    and all three granularities at once.

        dot   (rows, N, N)  <rho_i[r], rho_j[r]>
        sq    (N, rows)     ||rho_i[r]||^2
        sum   (N, rows)     sum of the row's entries, None unless `need_centered`
        sdot  (rows, N, N)  <sign(rho_i[r]), sign(rho_j[r])>, None unless `need_sign`

    Row-level accumulators sum exactly to layer level, and layer level sums
    exactly to model level, so nothing is recomputed and the three granularities
    cannot silently disagree about the same underlying inner products.

    The hierarchy is also what makes the model-wise number *correct*, not merely
    cheap.  A flat float32 reduction over the concatenated QV -- the obvious
    implementation, and what `F.cosine_similarity` on 85M coordinates does -- is
    wrong by about 1%: the running sum reaches ~10^2 while each addend is ~10^-8,
    which is below the ULP, so most of the tail is silently discarded.  Measured
    on ViT-B/16, that route gives ||rho|| = 11.6996 where the float64 answer is
    11.8242, and a model-wise cosine of 0.04556 against a true 0.04461.
    Accumulating per row, then per layer, then over layers keeps every partial
    sum within range of its addends and reproduces float64 to seven digits.
    """
    a = stacked.to(torch.float32)

    dot = torch.einsum("irc,jrc->rij", a, a)
    sq  = a.pow(2).sum(dim=2)

    total = a.sum(dim=2) if need_centered else None
    sdot  = torch.einsum("irc,jrc->rij", torch.sign(a), torch.sign(a)) if need_sign else None

    return dot, sq, total, sdot


def reduce_to_unit(dot, sq, total, sdot):
    """Sum row-level accumulators into the accumulators of the enclosing unit.

    Used twice: rows -> layer, and layer -> model.  Every accumulator is additive
    over a partition of the coordinates, which is the reason the three
    granularities share one pass.
    """
    return (
        dot.sum(dim=0),
        sq.sum(dim=1),
        total.sum(dim=1) if total is not None else None,
        sdot.sum(dim=0) if sdot is not None else None,
    )


def metric_values(name, dot, sq, total, sdot, numel):
    """Evaluate one metric for every pair, from accumulators of any granularity.

    Shapes: `dot`/`sdot` are (..., N, N), `sq`/`total` are (..., N), `numel` is
    the coordinate count of a single unit (a scalar, or a (...,) tensor when
    units differ in size).  Returns (..., N, N).
    """
    norms = sq.clamp_min(0.0).sqrt()
    npair = norms.unsqueeze(-1) * norms.unsqueeze(-2)

    if name == METRIC_DOT:
        return dot

    if name == METRIC_SIGN_AGREEMENT:
        n = numel if torch.is_tensor(numel) else torch.tensor(float(numel))
        n = n.reshape(sdot.shape[:-2] + (1, 1)) if torch.is_tensor(numel) else n
        return (sdot / n + 1.0) / 2.0

    if name == METRIC_CORRELATION:
        n = numel if torch.is_tensor(numel) else torch.tensor(float(numel))
        n_dot = n.reshape(dot.shape[:-2] + (1, 1)) if torch.is_tensor(numel) else n
        n_sq  = n.reshape(sq.shape[:-1] + (1,))    if torch.is_tensor(numel) else n

        cdot  = dot - total.unsqueeze(-1) * total.unsqueeze(-2) / n_dot
        csq   = (sq - total.pow(2) / n_sq).clamp_min(0.0)
        cnorm = csq.sqrt()
        return cdot / (cnorm.unsqueeze(-1) * cnorm.unsqueeze(-2)).clamp_min(EPS)

    cosine = dot / npair.clamp_min(EPS)

    if name == METRIC_COSINE:
        return cosine
    if name == METRIC_NORMALIZED_L2:
        return (2.0 - 2.0 * cosine).clamp_min(0.0).sqrt()
    if name == METRIC_ANGULAR:
        return torch.arccos(cosine.clamp(-1.0, 1.0)) / math.pi

    raise ValueError(f"metric not computable from vector accumulators: {name}")


def norm_pair_weights(sq):
    """||rho_i|| * ||rho_j|| for every pair, the weight `wmean_norm` uses.

    `sq` is (..., N) and the result is (..., N, N).  Defined for every metric,
    including sign_agreement: the weight describes how much QV energy the unit
    carries, which is a property of the vectors and not of the metric applied to
    them.
    """
    norms = sq.clamp_min(0.0).sqrt()
    return norms.unsqueeze(-1) * norms.unsqueeze(-2)


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------
def cka_layer(stacked):
    """Linear CKA between every pair of QVs for one layer, plus the norm weights.

    For X, Y of shape (rows, cols),

        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)

    and ||Y^T X||_F^2 = tr(X^T Y Y^T X) = <X X^T, Y Y^T>_F, so the whole N x N
    matrix follows from the N row-space Grams G_i = X_i X_i^T.  Going through the
    rows x rows Gram rather than cols x cols is what keeps this affordable: it
    costs one rows^2 * cols matmul per QV instead of an N^2 pass over cols^2.

    Returns (cka, weights) with weights[i, j] = ||G_i||_F ||G_j||_F, the natural
    `wmean_norm` weight at this granularity -- CKA's own denominator, so a
    norm-weighted mean of CKAs is the ratio of pooled numerators to pooled
    denominators, exactly as it is for cosine.
    """
    a = stacked.to(torch.float32)
    n = a.shape[0]

    grams = torch.stack([a[i] @ a[i].T for i in range(n)])       # (N, rows, rows)
    flat  = grams.reshape(n, -1)

    num   = flat @ flat.T                                        # <G_i, G_j>_F
    fnorm = flat.norm(dim=1)
    denom = fnorm.unsqueeze(1) * fnorm.unsqueeze(0)

    return num / denom.clamp_min(EPS), denom


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(values, w_params, w_norm, operators):
    """Reduce per-unit similarities to one value per pair, under each operator.

    `values` is (U, N, N) over U units; `w_params` is (U,) and `w_norm` is
    (U, N, N), since the norm weight depends on which pair is being weighted.
    Returns {operator: (N, N) tensor}.
    """
    out = {}
    u = values.shape[0]

    for op in operators:
        if op == OP_MEAN:
            out[op] = values.mean(dim=0)
        elif op == OP_WMEAN_PARAMS:
            w = w_params.reshape(-1, 1, 1).to(values.dtype)
            out[op] = (values * w).sum(dim=0) / w.sum(dim=0).clamp_min(EPS)
        elif op == OP_WMEAN_NORM:
            out[op] = (values * w_norm).sum(dim=0) / w_norm.sum(dim=0).clamp_min(EPS)
        elif op == OP_MEDIAN:
            out[op] = values.median(dim=0).values
        elif op == OP_STD:
            # Population std: with a single unit the spread is 0, not undefined.
            out[op] = values.std(dim=0, unbiased=False) if u > 1 else torch.zeros_like(values[0])
        elif op == OP_MIN:
            out[op] = values.min(dim=0).values
        elif op == OP_MAX:
            out[op] = values.max(dim=0).values
        elif op == OP_FRAC_POSITIVE:
            out[op] = (values > 0).to(values.dtype).mean(dim=0)
        else:
            raise ValueError(f"unknown operator: {op}")

    return out


def stack_operator_values(per_unit, operators):
    """Stack a list of {operator: (N, N)} into {operator: (U, N, N)}.

    The bridge between the two steps of a two-stage aggregation: step 1 produces
    one dict per layer, step 2 needs one tensor per operator across layers.
    """
    return {op: torch.stack([d[op] for d in per_unit]) for op in operators}


# ---------------------------------------------------------------------------
# Correlation, used by aggregate_qv_alignment.py
# ---------------------------------------------------------------------------
def pearson(xs, ys):
    """Pearson r with n and a Fisher-z 95% interval.

    scipy is not a declared dependency of this project and a correlation
    coefficient does not justify adding one.  The Fisher z-transform gives the
    interval in closed form, with no special functions:

        z = atanh(r),  se = 1/sqrt(n - 3),  CI = tanh(z +- 1.96 se)

    Pairs where either value is None are dropped, and the surviving n is
    reported so a correlation over four points cannot be mistaken for one over
    four hundred.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return {"r": None, "n": n, "ci_lo": None, "ci_hi": None}

    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n

    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)

    if sxx <= EPS or syy <= EPS:
        return {"r": None, "n": n, "ci_lo": None, "ci_hi": None}

    r = sxy / math.sqrt(sxx * syy)

    if n < 4 or abs(r) >= 1.0:
        return {"r": r, "n": n, "ci_lo": None, "ci_hi": None}

    z  = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    return {
        "r":     r,
        "n":     n,
        "ci_lo": math.tanh(z - 1.96 * se),
        "ci_hi": math.tanh(z + 1.96 * se),
    }


def _midranks(values):
    """Ranks with ties averaged, as Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1

    return ranks


def spearman(xs, ys):
    """Spearman rho, i.e. Pearson over midrank-corrected ranks.

    Monotone rather than linear association.  It is the more defensible statistic
    here: Proposition 1 predicts a specific functional form only under a local
    quadratic model, so an ordering claim survives assumptions a linear fit does
    not.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"r": None, "n": len(pairs), "ci_lo": None, "ci_hi": None}

    rx = _midranks([p[0] for p in pairs])
    ry = _midranks([p[1] for p in pairs])
    return pearson(rx, ry)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def to_float(value):
    """Tensor scalar to a JSON-serializable float, mapping non-finite to None."""
    v = float(value)
    return v if math.isfinite(v) else None


def matrix_to_json(matrix, names):
    """(N, N) tensor to {donor: {receiver: float}} keyed by dataset name."""
    return {
        names[i]: {names[j]: to_float(matrix[i, j]) for j in range(len(names))}
        for i in range(len(names))
    }
