"""Tie-aware signed-lambda analysis and frozen negative test manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(PROJECT_ROOT)

from src.text.data.common import DATASET_NAME_TO_EPOCHS


MODEL_SPECS = {
    "bert-base": {
        "name": "google-bert/bert-base-uncased",
        "dir": "google_bert_bert_base_uncased",
        "expected_failures": 71,
    },
    "bert-large": {
        "name": "google-bert/bert-large-uncased",
        "dir": "google_bert_bert_large_uncased",
        "expected_failures": 93,
    },
}
RAW_ROOT = Path(
    "evaluations/text/ilharco_automodelforsequenceclassification/"
    "001_qat_transfer/text/qv_transfer"
)
WIN_LOSS = Path(
    "evaluations/998_rebuttal/001_zero_shot_reframing/seed=2038/"
    "qat=bits=3_gran=channel/ptq=bits=3_gran=channel/split=test/"
    "win_loss_ilharco_automodelforsequenceclassification.json"
)
OUT_ROOT = Path(
    "evaluations/998_rebuttal/003_lambda_sensitivity/001_signed_bert/analysis"
)
TIE_EPS = 1e-12
REQUIRED_NEGATIVE_GRID = {
    -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40,
    -0.45, -0.50, -0.75, -1.00, -1.25, -1.50, -1.75, -2.00,
}
REQUIRED_POSITIVE_GRID = {round(0.05 * index, 2) for index in range(1, 41)}


def cell_prefix(model_dir: str, donor: str, receiver: str) -> Path:
    return (
        RAW_ROOT
        / model_dir
        / f"src={donor}_seed=2038"
        / f"tgt={receiver}_seed=2038"
        / "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128"
        / "qat=bits=3_gran=channel_skip=classifier"
        / "ptq=bits=3_gran=channel_skip=classifier"
    )


def load_curve(
    model_dir: str, donor: str, receiver: str, split: str
) -> dict[float, float]:
    prefix = cell_prefix(model_dir, donor, receiver)
    curve = {}
    if not prefix.exists():
        return curve
    for alpha_dir in prefix.glob("qv=alpha=*"):
        path = alpha_dir / f"split={split}" / "eval_results.json"
        if not path.exists():
            continue
        try:
            alpha = float(alpha_dir.name.split("=", 2)[2])
            data = json.loads(path.read_text())
            value = data[f"{split}_accuracy_fp_head_ptq"]
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        if isinstance(value, (int, float)):
            curve[alpha] = float(value)
    return curve


def maximizing(points: dict[float, float]) -> tuple[float, list[float]]:
    best = max(points.values())
    return best, sorted(
        a for a, value in points.items()
        if math.isclose(value, best, rel_tol=0.0, abs_tol=TIE_EPS)
    )


def classify(curve: dict[float, float], require_zero: bool = True) -> dict:
    groups = {
        "negative": {a: v for a, v in curve.items() if a < 0},
        "zero": {a: v for a, v in curve.items() if a == 0},
        "positive": {a: v for a, v in curve.items() if a > 0},
    }
    required_groups = ["negative", "positive"] + (["zero"] if require_zero else [])
    if any(not groups[name] for name in required_groups):
        missing = [name for name in required_groups if not groups[name]]
        raise RuntimeError(f"curve missing sign groups: {missing}")
    groups = {name: values for name, values in groups.items() if values}

    maxima = {name: maximizing(values) for name, values in groups.items()}
    global_best = max(value for value, _ in maxima.values())
    winners = sorted(
        name for name, (value, _) in maxima.items()
        if math.isclose(value, global_best, rel_tol=0.0, abs_tol=TIE_EPS)
    )
    category = winners[0] + "-only" if len(winners) == 1 else "sign-tied"
    negative_alphas = maxima["negative"][1]
    frozen_negative = min(negative_alphas, key=lambda a: abs(a))
    negative_sampled_boundary = min(groups["negative"])
    return {
        "category": category,
        "winning_signs": winners,
        "global_best_val_accuracy": global_best,
        "sign_maxima": {
            name: {"accuracy": value, "alphas": alphas}
            for name, (value, alphas) in maxima.items()
        },
        "frozen_negative_alpha": frozen_negative,
        "negative_sampled_boundary": negative_sampled_boundary,
        "negative_boundary_censored": negative_sampled_boundary in negative_alphas,
    }


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["category"] for row in rows)
    total = len(rows)
    return {
        "n": total,
        "category_counts": dict(sorted(counts.items())),
        "category_fractions": {
            category: count / total for category, count in sorted(counts.items())
        } if total else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="bert-large")
    parser.add_argument(
        "--allow-missing-zero",
        action="store_true",
        help="compare observed negative and positive arms when lambda=0 was not run",
    )
    parser.add_argument("--require-test", action="store_true")
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    model_name = spec["name"]
    model_dir = spec["dir"]

    win_loss = json.loads(WIN_LOSS.read_text())
    model = win_loss["models"][model_name]
    all_rows = []
    for pair in model["pairs"]:
        donor, receiver = pair["donor"], pair["receiver"]
        curve = load_curve(model_dir, donor, receiver, "val")
        if args.model == "bert-large":
            sampled = set(curve)
            missing_negative = sorted(REQUIRED_NEGATIVE_GRID - sampled)
            missing_positive = sorted(REQUIRED_POSITIVE_GRID - sampled)
            if missing_negative or missing_positive or 0.0 not in sampled:
                raise RuntimeError(
                    f"incomplete signed curve for {donor}->{receiver}: "
                    f"missing negative={missing_negative}, "
                    f"positive={missing_positive}, zero={0.0 not in sampled}"
                )
        result = classify(curve, require_zero=not args.allow_missing_zero)
        result.update(
            donor=donor,
            receiver=receiver,
            same_task=pair["same_task"],
            unit_test_delta=pair["delta"],
            unit_scale_success=pair["delta"] > 0,
            positive_alpha_best=pair["alpha_best"],
            positive_best_test_delta=pair["delta_best"],
        )

        neg_alpha = result["frozen_negative_alpha"]
        neg_test_path = cell_prefix(model_dir, donor, receiver) / f"qv=alpha={neg_alpha}" / "split=test" / "eval_results.json"
        if neg_test_path.exists():
            neg_test = json.loads(neg_test_path.read_text())["test_accuracy_fp_head_ptq"]
            result["negative_test_accuracy"] = neg_test
            result["negative_test_delta"] = neg_test - pair["baseline_acc"]
        else:
            result["negative_test_accuracy"] = None
            result["negative_test_delta"] = None

        all_rows.append(result)

    cross_task_rows = [row for row in all_rows if not row["same_task"]]
    rows = [row for row in cross_task_rows if not row["unit_scale_success"]]
    winning_rows = [row for row in cross_task_rows if row["unit_scale_success"]]
    same_task_rows = [row for row in all_rows if row["same_task"]]
    if len(rows) != spec["expected_failures"]:
        raise RuntimeError(
            f"expected {spec['expected_failures']} unit-scale failures, "
            f"found {len(rows)}"
        )

    followups = []
    for result in rows:
        if result["category"] == "negative-only" or (
            result["category"] == "sign-tied"
            and "negative" in result["winning_signs"]
        ):
            if result["negative_test_accuracy"] is None:
                followups.append(
                    {
                        "donor": result["donor"],
                        "receiver": result["receiver"],
                        "alpha": result["frozen_negative_alpha"],
                    }
                )

    if args.require_test and followups:
        raise RuntimeError(f"{len(followups)} selected negative test cells are missing")

    report = {
        "model": model_name,
        "n_failing_at_unit": len(rows),
        "zero_arm_required": not args.allow_missing_zero,
        "category_counts": summarize(rows)["category_counts"],
        "scope_summaries": {
            "all_pairs": summarize(all_rows),
            "cross_task": summarize(cross_task_rows),
            "cross_task_unit_failures": summarize(rows),
            "cross_task_unit_successes": summarize(winning_rows),
            "same_task": summarize(same_task_rows),
        },
        "n_negative_boundary_censored": sum(
            row["negative_boundary_censored"] for row in rows
        ),
        "n_selected_test_missing": len(followups),
        "pairs": rows,
        "all_pairs": all_rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("-", "_")
    (OUT_ROOT / f"signed_{model_slug}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    manifest_name = (
        "selected_negative_test_manifest.json"
        if args.model == "bert-large"
        else f"selected_negative_test_manifest_{model_slug}.json"
    )
    (OUT_ROOT / manifest_name).write_text(
        json.dumps({"runs": followups}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["category_counts"], sort_keys=True))
    print(f"selected negative test followups: {len(followups)}")


if __name__ == "__main__":
    main()
