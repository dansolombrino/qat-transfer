import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from common.run_id import hydra_override_arg, run_id_flat, run_id_path


ANALYSIS_PATH = ROOT / (
    "code/experiments/998_rebuttal/003_lambda_sensitivity/"
    "001_signed_bert/signed_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("signed_analysis", ANALYSIS_PATH)
SIGNED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIGNED)


def test_run_id_percent_encodes_unsafe_values():
    cfg = {"model": "a/b", "alpha": -0.05}
    assert str(run_id_path(cfg, ["model", "alpha"])) == "model=a%2Fb/alpha=-0.05"
    assert run_id_flat(cfg, ["model", "alpha"]) == "model=a%2Fb,alpha=-0.05"


def test_hydra_override_arg_quotes_string_delimiters():
    value = "evaluations/seed=2038/qat=bits=3/result.json"
    assert hydra_override_arg("outcome_path", value) == (
        'outcome_path="evaluations/seed=2038/qat=bits=3/result.json"'
    )


def curve(negative=0.4, zero=0.5, positive=0.6):
    return {-0.1: negative, -0.05: negative, 0.0: zero, 0.05: positive, 0.1: positive}


def test_positive_only_classification():
    result = SIGNED.classify(curve())
    assert result["category"] == "positive-only"
    assert result["winning_signs"] == ["positive"]


def test_negative_only_uses_closest_to_zero_tie_break():
    result = SIGNED.classify(curve(negative=0.7))
    assert result["category"] == "negative-only"
    assert result["frozen_negative_alpha"] == -0.05


def test_zero_only_classification():
    result = SIGNED.classify(curve(zero=0.8))
    assert result["category"] == "zero-only"


def test_cross_sign_tie_is_ambiguous():
    result = SIGNED.classify(curve(negative=0.7, zero=0.5, positive=0.7))
    assert result["category"] == "sign-tied"
    assert result["winning_signs"] == ["negative", "positive"]


def test_missing_zero_is_rejected():
    points = curve()
    del points[0.0]
    try:
        SIGNED.classify(points)
    except RuntimeError as exc:
        assert "zero" in str(exc)
    else:
        raise AssertionError("missing zero group was accepted")


def test_missing_zero_can_compare_observed_signed_arms():
    points = curve(negative=0.7, positive=0.6)
    del points[0.0]
    result = SIGNED.classify(points, require_zero=False)
    assert result["category"] == "negative-only"
    assert result["winning_signs"] == ["negative"]
