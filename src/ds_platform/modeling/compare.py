"""Model comparison and cross-validation aggregation."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ds_platform.hashing import SHA256_HEX_PATTERN
from ds_platform.modeling.evaluate import EvaluationReport, evaluate
from ds_platform.modeling.spec import MetricSpec
from ds_platform.types import FrozenJSON, freeze_json_value

type ComparisonMethod = Literal["holdout_delta", "paired_bootstrap"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonReport(_FrozenModel):
    """Delta between two evaluations on the same metric names.

    ``deltas`` are ``left - right``. Bootstrap intervals, when present,
    describe the sampling distribution of those deltas.
    """

    method: ComparisonMethod
    left_payload_id: str = Field(pattern=SHA256_HEX_PATTERN)
    right_payload_id: str = Field(pattern=SHA256_HEX_PATTERN)
    deltas: Mapping[str, float]
    interval_low: Mapping[str, float] | None = None
    interval_high: Mapping[str, float] | None = None
    n: int
    seed: int | None = None
    notes: tuple[str, ...] = ()

    @field_validator("deltas", "interval_low", "interval_high")
    @classmethod
    def _freeze_metric_maps(cls, metrics: Mapping[str, float] | None) -> object:
        if metrics is None:
            return None
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        metric_values = {name: float(value) for name, value in metrics.items()}
        frozen = freeze_json_value(metric_values)
        assert isinstance(frozen, FrozenJSON)
        return frozen


def compare_evaluations(
    left: EvaluationReport,
    right: EvaluationReport,
    *,
    left_payload_id: str,
    right_payload_id: str,
) -> ComparisonReport:
    """Return metric deltas between two evaluation reports.

    Both reports must contain the same metric names and the same ``n``.
    """
    if left.n != right.n:
        raise ValueError("evaluation reports must have the same n")
    metric_names = _shared_metric_names(left.metrics, right.metrics)
    deltas = {
        name: float(left.metrics[name]) - float(right.metrics[name])
        for name in metric_names
    }
    return ComparisonReport(
        method="holdout_delta",
        left_payload_id=left_payload_id,
        right_payload_id=right_payload_id,
        deltas=deltas,
        n=left.n,
    )


def paired_bootstrap(
    y_true: Sequence[object],
    y_pred_left: Sequence[object],
    y_pred_right: Sequence[object],
    metrics: Sequence[MetricSpec],
    *,
    left_payload_id: str,
    right_payload_id: str,
    seed: int,
    n_resamples: int = 1000,
    y_proba_left: Sequence[Sequence[float] | None] | None = None,
    y_proba_right: Sequence[Sequence[float] | None] | None = None,
    classes: Sequence[object] | None = None,
) -> ComparisonReport:
    """Estimate metric deltas with paired bootstrap resampling.

    Resamples aligned ``(y_true, y_pred_left, y_pred_right)`` tuples with
    replacement. Point ``deltas`` come from the full sample; default
    intervals use the 2.5 and 97.5 percentiles of bootstrap deltas.
    """
    if len(y_true) != len(y_pred_left) or len(y_true) != len(y_pred_right):
        raise ValueError("y_true, y_pred_left, and y_pred_right must align")
    if y_proba_left is not None and len(y_proba_left) != len(y_true):
        raise ValueError("y_proba_left length must match y_true length")
    if y_proba_right is not None and len(y_proba_right) != len(y_true):
        raise ValueError("y_proba_right length must match y_true length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    left_report = evaluate(
        y_true,
        y_pred_left,
        metrics,
        y_proba=y_proba_left,
        classes=classes,
    )
    right_report = evaluate(
        y_true,
        y_pred_right,
        metrics,
        y_proba=y_proba_right,
        classes=classes,
    )
    comparison = compare_evaluations(
        left_report,
        right_report,
        left_payload_id=left_payload_id,
        right_payload_id=right_payload_id,
    )
    metric_names = tuple(comparison.deltas)
    if not metric_names:
        return comparison.model_copy(
            update={
                "method": "paired_bootstrap",
                "seed": seed,
                "notes": ("paired_bootstrap used aligned predictions",),
            }
        )

    rng = random.Random(seed)
    n_pairs = len(y_true)
    bootstrap_deltas: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(n_resamples):
        sample_indexes = [rng.randrange(n_pairs) for _ in range(n_pairs)]
        sample_true = [y_true[index] for index in sample_indexes]
        sample_left = [y_pred_left[index] for index in sample_indexes]
        sample_right = [y_pred_right[index] for index in sample_indexes]
        sample_proba_left = (
            [y_proba_left[index] for index in sample_indexes]
            if y_proba_left is not None
            else None
        )
        sample_proba_right = (
            [y_proba_right[index] for index in sample_indexes]
            if y_proba_right is not None
            else None
        )
        left_sample = evaluate(
            sample_true,
            sample_left,
            metrics,
            y_proba=sample_proba_left,
            classes=classes,
        )
        right_sample = evaluate(
            sample_true,
            sample_right,
            metrics,
            y_proba=sample_proba_right,
            classes=classes,
        )
        for name in metric_names:
            bootstrap_deltas[name].append(
                float(left_sample.metrics[name]) - float(right_sample.metrics[name])
            )

    interval_low = {
        name: _percentile(bootstrap_deltas[name], 2.5) for name in metric_names
    }
    interval_high = {
        name: _percentile(bootstrap_deltas[name], 97.5) for name in metric_names
    }
    return ComparisonReport(
        method="paired_bootstrap",
        left_payload_id=left_payload_id,
        right_payload_id=right_payload_id,
        deltas=comparison.deltas,
        interval_low=interval_low,
        interval_high=interval_high,
        n=n_pairs,
        seed=seed,
        notes=("paired_bootstrap used aligned predictions",),
    )


def aggregate_fold_reports(reports: Sequence[EvaluationReport]) -> EvaluationReport:
    """Return the unweighted mean of fold metrics."""
    if not reports:
        raise ValueError("reports must contain at least one entry")
    metric_names = _shared_metric_names(*[report.metrics for report in reports])
    aggregated = {
        name: sum(float(report.metrics[name]) for report in reports) / len(reports)
        for name in metric_names
    }
    return EvaluationReport(
        metrics=aggregated,
        n=sum(report.n for report in reports),
        n_missing=sum(report.n_missing for report in reports),
        notes=(f"aggregated from {len(reports)} folds",),
    )


def _shared_metric_names(*metric_maps: Mapping[str, float]) -> tuple[str, ...]:
    if not metric_maps:
        return ()
    names = set(metric_maps[0])
    for metrics in metric_maps[1:]:
        if set(metrics) != names:
            raise ValueError("evaluation reports must contain the same metric names")
    return tuple(sorted(names))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
