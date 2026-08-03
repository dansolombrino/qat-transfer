"""Validation and output routing for row-wise 005 figure replicas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from _common import (
    PROJECT_ROOT,
    EXPERIMENT,
    _safe_run_path,
    default_plot_root,
    matrix_from_records,
    save_figure,
)


SCHEMA_VERSION = "rowwise_statistics_v1"
ANALYSIS_SPEC = "reviewer_3hfp_rowwise_v1"


def load_statistics(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"statistics JSON root must be an object: {path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"statistics schema={data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
    if data.get("analysis_spec") != ANALYSIS_SPEC:
        raise ValueError(f"unsupported analysis spec: {data.get('analysis_spec')!r}")
    if data.get("alignment_aggregation") != "row_cosine_mean_v1":
        raise ValueError("statistics artifact is not the approved row-wise aggregation")
    if data.get("missing") != []:
        raise ValueError(
            f"statistics artifact reports missing records: {data.get('missing')}"
        )
    if len(data.get("points", [])) != 462:
        raise ValueError("statistics artifact must contain exactly 462 cross-task points")
    if len(data.get("diagonal_audit", [])) != 22:
        raise ValueError("statistics artifact must contain exactly 22 diagonal records")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict) or len(provenance.get("task_order", [])) != 22:
        raise ValueError("statistics artifact has invalid task-order provenance")
    if provenance.get("producer_stage") != "rowwise_alignment":
        raise ValueError("statistics artifact has invalid producer-stage provenance")
    return data


def output_directory(data: Dict[str, Any], script_stem: str, root: Path) -> Path:
    provenance = data["provenance"]
    producer = _safe_run_path(str(provenance["producer_run_id_path"]))
    analyzer = _safe_run_path(str(provenance["analyzer_run_id_path"]))
    return (
        root
        / EXPERIMENT
        / script_stem
        / "rowwise_alignment"
        / producer
        / "analysis"
        / analyzer
    )


__all__ = [
    "PROJECT_ROOT",
    "default_plot_root",
    "load_statistics",
    "matrix_from_records",
    "output_directory",
    "save_figure",
]
