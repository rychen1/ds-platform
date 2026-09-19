"""Evaluation metrics (modeling Part 4)."""

from __future__ import annotations

import sys

import pytest

from ds_platform.modeling.evaluate import (
    EvaluationReport,
    accuracy,
    evaluate,
    macro_f1,
    mae,
    rmse,
)
from ds_platform.modeling.spec import MetricSpec

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}


def test_accuracy_on_toy_vectors() -> None:
    assert accuracy(["A", "B", "A"], ["A", "A", "A"]) == pytest.approx(2 / 3)


def test_macro_f1_on_toy_vectors() -> None:
    score = macro_f1(["A", "A", "B", "B"], ["A", "B", "B", "B"])
    assert 0.0 < score < 1.0


def test_rmse_and_mae_on_toy_vectors() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 3.0, 3.0]
    assert rmse(y_true, y_pred) == pytest.approx((1.0 / 3.0) ** 0.5)
    assert mae(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_evaluate_counts_missing_predictions() -> None:
    report = evaluate(
        ["A", "B", None, "A"],
        ["A", None, "B", "A"],
        [MetricSpec(name="accuracy")],
    )
    assert isinstance(report, EvaluationReport)
    assert report.n == 4
    assert report.n_missing == 2
    assert report.metrics["accuracy"] == 1.0


def test_evaluate_unknown_metric_errors() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(["A"], ["A"], [MetricSpec(name="ndcg")])


def test_evaluate_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate(["A"], ["A", "B"], [MetricSpec(name="accuracy")])


def test_evaluate_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.evaluate  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
