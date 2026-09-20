"""Model-free evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from ds_platform.modeling.spec import MetricSpec
from ds_platform.types import FrozenJSON, freeze_json_value

type MetricFn = Callable[[list[object], list[object]], float]
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)

_LOG_LOSS_EPS = 1e-15
_PROBA_METRICS = frozenset({"log_loss", "brier_score", "roc_auc_binary"})
_ALLOWED_PARAMS: dict[str, frozenset[str]] = {
    "roc_auc_binary": frozenset({"pos_label"}),
}


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
    y_proba: Sequence[Sequence[float] | None] | None = None,
    extra_metrics: Mapping[str, MetricFn] | None = None,
    classes: Sequence[object] | None = None,
) -> EvaluationReport:
    """Compute named metrics over aligned label and prediction vectors.

    Pairs with a missing ``y_true`` or ``y_pred`` (``None``) are excluded
    from metric computation and counted in ``n_missing``. Missing values are
    never imputed. Probability rows are aligned to the same kept pairs.

    Probability metrics (``log_loss``, ``brier_score``, ``roc_auc_binary``)
    require ``y_proba``. Columns follow ``classes`` when provided, otherwise
    the sorted unique labels in the kept ``y_true`` values. Pass ``classes``
    when the probability width includes labels absent from this slice.

    ``EvaluationReport.notes`` records which metrics used probabilities.
    ``MetricSpec.params`` is rejected except for the closed set each metric
    documents (``roc_auc_binary`` accepts ``pos_label``).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if y_proba is not None and len(y_proba) != len(y_true):
        raise ValueError("y_proba length must match y_true length")

    valid_true: list[object] = []
    valid_pred: list[object] = []
    valid_proba: list[Sequence[float] | None] | None = None
    if y_proba is not None:
        valid_proba = []
    n_missing = 0
    for index, (true_value, pred_value) in enumerate(zip(y_true, y_pred, strict=True)):
        if true_value is None or pred_value is None:
            n_missing += 1
            continue
        valid_true.append(true_value)
        valid_pred.append(pred_value)
        if valid_proba is not None and y_proba is not None:
            valid_proba.append(y_proba[index])

    computed: dict[str, float] = {}
    notes: list[str] = []
    for metric in metrics:
        if metric.name in computed:
            raise ValueError(f"duplicate metric name: {metric.name!r}")
        if extra_metrics is not None and metric.name in extra_metrics:
            if metric.params:
                raise ValueError(
                    f"metric {metric.name!r} does not accept params: "
                    f"{dict(metric.params)}"
                )
            computed[metric.name] = extra_metrics[metric.name](valid_true, valid_pred)
            continue
        if metric.name not in _KNOWN_METRICS:
            raise ValueError(f"unknown metric: {metric.name!r}")
        allowed = _ALLOWED_PARAMS.get(metric.name, frozenset())
        unexpected = set(metric.params) - allowed
        if unexpected:
            raise ValueError(
                f"metric {metric.name!r} does not accept params: {dict(metric.params)}"
            )
        if metric.name in _PROBA_METRICS:
            if valid_proba is None:
                raise ValueError(f"metric {metric.name!r} requires y_proba")
            computed[metric.name] = _call_proba_metric(
                metric.name,
                valid_true,
                valid_proba,
                params=metric.params,
                classes=classes,
            )
            notes.append(f"{metric.name} used y_proba")
            continue
        computed[metric.name] = _METRIC_BY_NAME[metric.name](valid_true, valid_pred)

    return EvaluationReport(
        metrics=computed,
        n=len(y_true),
        n_missing=n_missing,
        notes=tuple(notes),
    )


def accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return the fraction of exactly matching predictions."""
    if not y_true:
        return 0.0
    _require_classification_labels(y_true, metric="accuracy")
    _require_classification_labels(y_pred, metric="accuracy")
    correct = sum(
        1
        for true_value, pred_value in zip(y_true, y_pred, strict=True)
        if true_value == pred_value
    )
    return correct / len(y_true)


def macro_f1(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return unweighted mean F1 across labels present in ``y_true``."""
    _require_classification_labels(y_true, metric="macro_f1")
    _require_classification_labels(y_pred, metric="macro_f1")
    labels = sorted({label for label in y_true}, key=repr)
    if not labels:
        return 0.0
    scores = [_binary_f1(y_true, y_pred, label) for label in labels]
    return sum(scores) / len(scores)


def macro_precision(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return unweighted mean precision across labels present in ``y_true``."""
    _require_classification_labels(y_true, metric="macro_precision")
    _require_classification_labels(y_pred, metric="macro_precision")
    labels = sorted({label for label in y_true}, key=repr)
    if not labels:
        return 0.0
    scores = [_binary_precision(y_true, y_pred, label) for label in labels]
    return sum(scores) / len(scores)


def macro_recall(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return unweighted mean recall across labels present in ``y_true``."""
    _require_classification_labels(y_true, metric="macro_recall")
    _require_classification_labels(y_pred, metric="macro_recall")
    labels = sorted({label for label in y_true}, key=repr)
    if not labels:
        return 0.0
    scores = [_binary_recall(y_true, y_pred, label) for label in labels]
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


def r2(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Return the coefficient of determination for numeric targets.

    A constant ``y_true`` returns ``1.0`` when predictions match and ``0.0``
    otherwise, so the score stays finite.
    """
    if not y_true:
        return 0.0
    true_values = [_as_float(value) for value in y_true]
    pred_values = [_as_float(value) for value in y_pred]
    residual = sum(
        (true_value - pred_value) ** 2
        for true_value, pred_value in zip(true_values, pred_values, strict=True)
    )
    mean_true = sum(true_values) / len(true_values)
    total = sum((true_value - mean_true) ** 2 for true_value in true_values)
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / total


def log_loss(
    y_true: Sequence[object],
    y_proba: Sequence[Sequence[float]],
    *,
    classes: Sequence[object] | None = None,
) -> float:
    """Return mean clipped multinomial log loss.

    ``y_proba`` columns follow ``classes``, or the sorted unique labels in
    ``y_true`` when ``classes`` is omitted. Rows are clipped to
    ``[1e-15, 1 - 1e-15]`` and renormalized before taking logs.
    """
    if not y_true:
        return 0.0
    resolved, rows = _prepare_proba_inputs(
        y_true,
        y_proba,
        classes=classes,
        metric="log_loss",
    )
    index_by_class = {label: index for index, label in enumerate(resolved)}
    total = 0.0
    for true_value, row in zip(y_true, rows, strict=True):
        clipped = _clip_and_renormalize(row)
        total += -math.log(clipped[index_by_class[true_value]])
    return total / len(y_true)


def brier_score(
    y_true: Sequence[object],
    y_proba: Sequence[Sequence[float]],
    *,
    classes: Sequence[object] | None = None,
) -> float:
    """Return mean multiclass Brier score.

    For each row this is the sum of squared errors between the probability
    vector and a one-hot encoding of the label. Binary two-column scores are
    therefore twice the one-probability binary Brier score.
    """
    if not y_true:
        return 0.0
    resolved, rows = _prepare_proba_inputs(
        y_true,
        y_proba,
        classes=classes,
        metric="brier_score",
    )
    index_by_class = {label: index for index, label in enumerate(resolved)}
    total = 0.0
    for true_value, row in zip(y_true, rows, strict=True):
        true_index = index_by_class[true_value]
        for class_index, probability in enumerate(row):
            target = 1.0 if class_index == true_index else 0.0
            delta = probability - target
            total += delta * delta
    return total / len(y_true)


def roc_auc_binary(
    y_true: Sequence[object],
    y_proba: Sequence[Sequence[float]],
    *,
    classes: Sequence[object] | None = None,
    pos_label: object | None = None,
) -> float:
    """Return ROC AUC for a binary labeling using the positive-class score.

    ``pos_label`` selects the probability column. When omitted, ``classes``
    must contain exactly two labels and the second label is positive.
    Constant scores with both classes present return ``0.5``.
    """
    if not y_true:
        return 0.0
    resolved, rows = _prepare_proba_inputs(
        y_true,
        y_proba,
        classes=classes,
        metric="roc_auc_binary",
    )
    positive = _resolve_pos_label(resolved, pos_label)
    positive_index = resolved.index(positive)
    scores = [row[positive_index] for row in rows]
    positives = [
        score
        for true_value, score in zip(y_true, scores, strict=True)
        if true_value == positive
    ]
    negatives = [
        score
        for true_value, score in zip(y_true, scores, strict=True)
        if true_value != positive
    ]
    if not positives or not negatives:
        raise ValueError("roc_auc_binary requires both positive and negative labels")
    concordant = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                concordant += 1.0
            elif positive_score == negative_score:
                concordant += 0.5
    return concordant / (len(positives) * len(negatives))


def _call_proba_metric(
    name: str,
    y_true: Sequence[object],
    y_proba: Sequence[Sequence[float] | None],
    *,
    params: Mapping[str, JsonValue],
    classes: Sequence[object] | None,
) -> float:
    rows = _require_proba_rows(y_proba, metric=name)
    if name == "log_loss":
        return log_loss(y_true, rows, classes=classes)
    if name == "brier_score":
        return brier_score(y_true, rows, classes=classes)
    if name == "roc_auc_binary":
        return roc_auc_binary(
            y_true,
            rows,
            classes=classes,
            pos_label=params.get("pos_label"),
        )
    raise ValueError(f"unknown metric: {name!r}")


def _prepare_proba_inputs(
    y_true: Sequence[object],
    y_proba: Sequence[Sequence[float]],
    *,
    classes: Sequence[object] | None,
    metric: str,
) -> tuple[tuple[object, ...], tuple[tuple[float, ...], ...]]:
    _require_classification_labels(y_true, metric=metric)
    resolved = _resolve_classes(y_true, classes, metric=metric)
    rows = tuple(
        _require_proba_row(row, width=len(resolved), index=index, metric=metric)
        for index, row in enumerate(y_proba)
    )
    if len(rows) != len(y_true):
        raise ValueError("y_proba length must match y_true length")
    return resolved, rows


def _resolve_classes(
    y_true: Sequence[object],
    classes: Sequence[object] | None,
    *,
    metric: str,
) -> tuple[object, ...]:
    if classes is None:
        return tuple(sorted({label for label in y_true}, key=repr))
    resolved = tuple(classes)
    if not resolved:
        raise ValueError("classes must contain at least one entry")
    if len(set(resolved)) != len(resolved):
        raise ValueError("classes must be unique")
    _require_classification_labels(resolved, metric=metric)
    known = set(resolved)
    unknown = [label for label in y_true if label not in known]
    if unknown:
        raise ValueError(f"{metric} y_true contains labels not in classes: {unknown!r}")
    return resolved


def _resolve_pos_label(
    classes: tuple[object, ...],
    pos_label: object | None,
) -> object:
    if pos_label is None:
        if len(classes) != 2:
            raise ValueError("roc_auc_binary requires pos_label when classes != 2")
        return classes[1]
    if isinstance(pos_label, bool) or not isinstance(pos_label, str | int):
        raise TypeError("roc_auc_binary pos_label must be str or int")
    if pos_label not in classes:
        raise ValueError(f"roc_auc_binary pos_label {pos_label!r} is not in classes")
    return pos_label


def _require_proba_rows(
    y_proba: Sequence[Sequence[float] | None],
    *,
    metric: str,
) -> list[Sequence[float]]:
    rows: list[Sequence[float]] = []
    for row in y_proba:
        if row is None:
            raise ValueError(f"{metric} requires y_proba for every kept pair")
        rows.append(row)
    return rows


def _require_proba_row(
    row: Sequence[float],
    *,
    width: int,
    index: int,
    metric: str,
) -> tuple[float, ...]:
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
        raise TypeError(
            f"{metric} y_proba[{index}] must be a sequence of probabilities"
        )
    if len(row) != width:
        raise ValueError(
            f"{metric} y_proba[{index}] width must be {width}, got {len(row)}"
        )
    values: list[float] = []
    for cell in row:
        if isinstance(cell, bool) or not isinstance(cell, int | float):
            raise TypeError(f"{metric} y_proba[{index}] must contain finite numbers")
        value = float(cell)
        if not math.isfinite(value):
            raise ValueError(f"{metric} y_proba[{index}] must contain finite numbers")
        values.append(value)
    return tuple(values)


def _clip_and_renormalize(row: Sequence[float]) -> list[float]:
    clipped = [min(max(value, _LOG_LOSS_EPS), 1.0 - _LOG_LOSS_EPS) for value in row]
    total = sum(clipped)
    return [value / total for value in clipped]


def _binary_f1(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    label: object,
) -> float:
    precision = _binary_precision(y_true, y_pred, label)
    recall = _binary_recall(y_true, y_pred, label)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _binary_precision(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    label: object,
) -> float:
    true_positive = 0
    predicted_positive = 0
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        if pred_value == label:
            predicted_positive += 1
            if true_value == label:
                true_positive += 1
    if predicted_positive == 0:
        return 0.0
    return true_positive / predicted_positive


def _binary_recall(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    label: object,
) -> float:
    true_positive = 0
    actual_positive = 0
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        if true_value == label:
            actual_positive += 1
            if pred_value == label:
                true_positive += 1
    if actual_positive == 0:
        return 0.0
    return true_positive / actual_positive


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {type(value)!r}")
    return float(value)


def _require_classification_labels(
    values: Sequence[object],
    *,
    metric: str,
) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, str | int):
            raise TypeError(f"{metric} requires str or int labels, got {value!r}")


_METRIC_BY_NAME: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}
_KNOWN_METRICS = frozenset(_METRIC_BY_NAME) | _PROBA_METRICS
