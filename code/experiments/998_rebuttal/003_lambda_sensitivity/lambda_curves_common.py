"""998 — Shared grid and curve math for the lambda sensitivity analysis.

Family-independent helpers imported by the three compute_lambda_curves_*.py
scripts.  Only the path logic differs between families, and that stays in the
per-family scripts, mirroring 001_zero_shot_reframing.

The reviewers asked how sensitive QV patching is to the scaling lambda.  The
answer is the shape of Delta(lambda) around its optimum, summarised here by four
per-cell quantities:

    safe_interval   the widest contiguous lambda range around the optimum with
                    Delta > 0: how wrong one may be about lambda before the
                    patch does harm.
    plateau_width   the width of the region retaining >= 90% of the best gain:
                    how flat the optimum is.
    unit_retention  Delta(1) / Delta(lambda*): what the data-free default costs.
    unimodal        whether the curve rises then falls, as the local quadratic
                    picture behind Proposition 1 predicts.

Two properties of the grid drive the implementation:

    Non-uniform spacing.  The shared grid steps by 0.15 except between 1.0 and
    1.05, which are 0.05 apart.  Counting grid points with Delta > 0 would
    therefore be a biased width estimator, so every width is measured in lambda
    units with linearly interpolated crossings.

    Right censoring.  The grid stops at 1.5, so a cell still winning there has
    an unbounded interval.  These are flagged rather than silently truncated at
    1.5.  The left side needs no such treatment: Delta(0) = 0 exactly, since
    patching with a zero vector returns the fine-tuned checkpoint, so lambda = 0
    is prepended as a known anchor and bounds every interval from below.
"""

import statistics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The lambda values shared by every family.  Some models were additionally swept
# on a 0.05..2.0 grid, but they are restricted to these so that all curves are
# directly comparable.
GRID = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.05, 1.2, 1.35, 1.5]

# Delta(0) = 0 by construction, not by measurement.  Used as the left anchor.
ANCHOR_ALPHA = 0.0
ANCHOR_DELTA = 0.0

# The data-free default scaling.
UNIT_ALPHA = 1.0

# A cell is "on its plateau" while it retains this fraction of its best gain.
PLATEAU_FRAC = 0.9


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------
def curve_key(alpha):
    """Stable string key for a grid lambda, for use in JSON objects."""
    return str(alpha)


def _anchored_points(curve):
    """Grid points as a sorted (alpha, delta) list with the lambda=0 anchor."""
    pts = [(a, curve[curve_key(a)]) for a in GRID if curve_key(a) in curve]
    return [(ANCHOR_ALPHA, ANCHOR_DELTA)] + pts


def _cross(a_lo, d_lo, a_hi, d_hi, level):
    """Lambda at which the segment crosses `level`, by linear interpolation."""
    if d_hi == d_lo:
        return a_lo
    return a_lo + (level - d_lo) * (a_hi - a_lo) / (d_hi - d_lo)


def _interval_above(points, star_idx, level):
    """Widest contiguous interval around points[star_idx] with delta > level.

    Returns (lo, hi, left_censored, right_censored).  Crossings are linearly
    interpolated; an endpoint of the grid that is still above `level` is
    reported at the grid edge and flagged as censored.
    """
    # Walk left until the curve drops to or below the level, then interpolate.
    lo, left_censored = points[0][0], True
    for i in range(star_idx, 0, -1):
        a_prev, d_prev = points[i - 1]
        if d_prev <= level:
            lo = _cross(a_prev, d_prev, points[i][0], points[i][1], level)
            left_censored = False
            break

    # Same to the right.
    hi, right_censored = points[-1][0], True
    for i in range(star_idx, len(points) - 1):
        a_next, d_next = points[i + 1]
        if d_next <= level:
            hi = _cross(points[i][0], points[i][1], a_next, d_next, level)
            right_censored = False
            break

    return lo, hi, left_censored, right_censored


def _sign_changes(deltas):
    """Sign changes in the successive differences, ignoring exact ties."""
    signs = []
    for prev, cur in zip(deltas, deltas[1:]):
        diff = cur - prev
        if diff > 0:
            signs.append(1)
        elif diff < 0:
            signs.append(-1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def curve_stats(curve):
    """Derived sensitivity statistics for one donor-receiver cell.

    `curve` maps curve_key(alpha) -> Delta(alpha) over (a subset of) GRID.
    Quantities that presuppose a positive gain are None when the cell never
    wins, since neither a plateau nor a retention fraction means anything there.
    """
    points = _anchored_points(curve)
    measured = points[1:]
    if not measured:
        return None

    # The optimum is taken over the measured grid, never over the anchor: a cell
    # whose curve is negative everywhere should report the lambda that hurt it
    # least, not lambda = 0.
    measured_idx, (lambda_star, delta_max) = max(
        enumerate(measured), key=lambda kv: kv[1][1]
    )
    star_idx = measured_idx + 1  # position of the same point in `points`

    delta_unit = curve.get(curve_key(UNIT_ALPHA))
    deltas = [d for _, d in measured]

    stats = {
        "n_points":      len(measured),
        "lambda_star":   lambda_star,
        "delta_max":     delta_max,
        "delta_unit":    delta_unit,
        "wins_at_unit":  None if delta_unit is None else delta_unit > 0,
        "n_sign_changes": _sign_changes(deltas),
        "unimodal":      _sign_changes(deltas) <= 1,
        "safe_interval": None,
        "plateau":       None,
        "unit_retention": None,
    }

    # A cell that never beats the baseline has no interval to report: the whole
    # grid sits at or below zero, and lambda_star is the anchor itself.
    if delta_max <= 0:
        return stats

    lo, hi, _, right_censored = _interval_above(points, star_idx, 0.0)
    stats["safe_interval"] = {
        "lo":             lo,
        "hi":             hi,
        "width":          hi - lo,
        "right_censored": right_censored,
    }

    p_lo, p_hi, _, p_censored = _interval_above(
        points, star_idx, PLATEAU_FRAC * delta_max
    )
    stats["plateau"] = {
        "lo":             p_lo,
        "hi":             p_hi,
        "width":          p_hi - p_lo,
        "right_censored": p_censored,
        "threshold":      PLATEAU_FRAC * delta_max,
    }

    if delta_unit is not None:
        stats["unit_retention"] = delta_unit / delta_max

    return stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def dist_stats(values):
    """Summary of a distribution, skipping None entries."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "mean": None, "median": None,
                "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n":      len(vals),
        "mean":   statistics.fmean(vals),
        "median": statistics.median(vals),
        "p25":    vals[len(vals) // 4],
        "p75":    vals[(3 * len(vals)) // 4],
        "min":    vals[0],
        "max":    vals[-1],
    }


def pooled_curve(cells):
    """Delta(lambda) pooled across cells, one summary per grid lambda.

    `cells` is a list of curve dicts.  The win rate at each lambda is the
    protocol-level answer to the sensitivity question: it says how much of the
    benefit survives a badly chosen scaling.
    """
    out = {}
    for alpha in GRID:
        key = curve_key(alpha)
        vals = sorted(c[key] for c in cells if key in c)
        if not vals:
            out[key] = {"n": 0, "mean": None, "median": None,
                        "p25": None, "p75": None, "win_rate": None}
            continue
        out[key] = {
            "n":        len(vals),
            "mean":     statistics.fmean(vals),
            "median":   statistics.median(vals),
            "p25":      vals[len(vals) // 4],
            "p75":      vals[(3 * len(vals)) // 4],
            "win_rate": sum(1 for v in vals if v > 0) / len(vals),
        }
    return out


def _frac(flags):
    """Fraction of True among the non-None flags."""
    vals = [f for f in flags if f is not None]
    if not vals:
        return None
    return sum(1 for f in vals if f) / len(vals)


def summarize_cells(pairs):
    """Aggregate the per-cell statistics of one model over a set of pairs."""
    curves = [p["curve"] for p in pairs if p.get("curve")]
    stats  = [p["stats"] for p in pairs if p.get("stats")]
    if not stats:
        return {"n": 0}

    safe    = [s["safe_interval"] for s in stats if s["safe_interval"] is not None]
    plateau = [s["plateau"]       for s in stats if s["plateau"] is not None]

    return {
        "n":             len(stats),
        "curve":         pooled_curve(curves),
        "safe_width":    dist_stats([s["width"] for s in safe]),
        "safe_lo":       dist_stats([s["lo"]    for s in safe]),
        "safe_hi":       dist_stats([s["hi"]    for s in safe]),
        "plateau_width": dist_stats([s["width"] for s in plateau]),
        "unit_retention": dist_stats([s["unit_retention"] for s in stats]),
        "lambda_star":   dist_stats([s["lambda_star"] for s in stats]),
        "frac_unimodal": _frac([s["unimodal"] for s in stats]),
        "frac_right_censored": _frac([s["right_censored"] for s in safe]),
        "n_never_wins":  sum(1 for s in stats if s["safe_interval"] is None),
    }
