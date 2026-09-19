"""Model-free evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from ds_platform.modeling.spec import MetricSpec
from ds_platform.types import FrozenJSON, freeze_json_value

type MetricFn = Callable[[list[object], list[object]], float]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationReport(_FrozenModel):
    metrics: Mapping[str, float]
    n: int
    n_missing: int
    notes: tuple[str, ...] = ()

    @field_validator("metrics")
    @classmethod
    def _freeze_metrics(cls, metrics: Mapping[str, float]) -> FrozenJSON:
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        frozen = freeze_json_value(
            {name: float(value) for name, value in metrics.items()}
        )
        assert isinstance(frozen, FrozenJSON)
        return frozen


def evaluate(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    metrics: Sequence[MetricSpec],
    *,
    y_proba: Sequence[Sequence[float]] | None = None,
) -> EvaluationReport:
    """Compute named metrics over aligned label and prediction vectors.

    Pairs with a missing ``y_true`` or ``y_pred`` (``None``) are excluded
    from metric computation and counted in ``n_missing``. Missing values are
    never imputed.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if y_proba is not None and len(y_proba) != len(y_true):
        raise ValueError("y_proba length must match y_true length")

    valid_true: list[object] = []
    valid_pred: list[object] = []
    n_missing = 0
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        if true_value is None or pred_value is None:
            n_missing += 1
            continue
        valid_true.append(true_value)
        valid_pred.append(pred_value)

    if y_proba is not None:
        raise ValueError("y_proba is not used by built-in metrics")

    computed: dict[str, float] = {}
    for metric in metrics:
        if metric.name in computed:
            raise ValueError(f"duplicate metric name: {metric.name!r}")
        try:
            metric_fn = _METRIC_BY_NAME[metric.name]
        except KeyError as exc:
            raise ValueError(f"unknown metric: {metric.name!r}") from exc
        if metric.params:
            raise ValueError(
                f"metric {metric.name!r} does not accept params: {dict(metric.params)}"
            )
        computed[metric.name] = metric_fn(valid_true, valid_pred)

    return EvaluationReport(
        metrics=computed,
        n=len(y_true),
        n_missing=n_missing,
    )


def accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return the fraction of exactly matching predictions."""
    if not y_true:
        return 0.0
    correct = sum(
        1
        for true_value, pred_value in zip(y_true, y_pred, strict=True)
        if true_value == pred_value
    )
    return correct / len(y_true)


def macro_f1(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return unweighted mean F1 across labels present in ``y_true``."""
    labels = sorted({label for label in y_true}, key=repr)
    if not labels:
        return 0.0
    scores = [_binary_f1(y_true, y_pred, label) for label in labels]
    return sum(scores) / len(scores)


def rmse(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return root mean squared error for numeric targets."""
    if not y_true:
        return 0.0
    squared_errors = [
        (_as_float(true_value) - _as_float(pred_value)) ** 2
        for true_value, pred_value in zip(y_true, y_pred, strict=True)
    ]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def mae(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return mean absolute error for numeric targets."""
    if not y_true:
        return 0.0
    absolute_errors = [
        abs(_as_float(true_value) - _as_float(pred_value))
        for true_value, pred_value in zip(y_true, y_pred, strict=True)
    ]
    return sum(absolute_errors) / len(absolute_errors)


def _binary_f1(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    label: object,
) -> float:
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        if true_value == label:
            actual_positive += 1
            if pred_value == label:
                true_positive += 1
        if pred_value == label:
            predicted_positive += 1
    if actual_positive == 0 or predicted_positive == 0:
        return 0.0
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {type(value)!r}")
    return float(value)


_METRIC_BY_NAME: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "rmse": rmse,
    "mae": mae,
}
