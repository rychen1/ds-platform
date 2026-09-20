"""Evaluation metrics (modeling Part 4)."""

from __future__ import annotations

import math

import pytest

from ds_platform.modeling.evaluate import (
    EvaluationReport,
    accuracy,
    brier_score,
    evaluate,
    log_loss,
    macro_f1,
    macro_precision,
    macro_recall,
    mae,
    r2,
    rmse,
    roc_auc_binary,
)
from ds_platform.modeling.spec import MetricSpec
from import_boundary_util import assert_import_does_not_pull

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
    assert_import_does_not_pull("ds_platform.modeling.evaluate", _FORBIDDEN)


def test_accuracy_rejects_bool_labels() -> None:
    with pytest.raises(TypeError, match="str or int labels"):
        accuracy([True], [1])


def test_evaluate_ignores_unused_probabilities_for_label_metrics() -> None:
    report = evaluate(
        ["A", "B"],
        ["A", "B"],
        [MetricSpec(name="accuracy")],
        y_proba=[[0.9, 0.1], [0.2, 0.8]],
    )
    assert report.metrics["accuracy"] == 1.0
    assert report.notes == ()


def test_log_loss_perfect_binary_predictions() -> None:
    y_true = ["A", "B", "A"]
    y_proba = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    score = log_loss(y_true, y_proba, classes=("A", "B"))
    assert score == pytest.approx(0.0, abs=1e-12)


def test_log_loss_multiclass_known_value() -> None:
    y_true = ["A", "B", "C"]
    y_proba = [
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
    ]
    expected = -(math.log(0.8) + math.log(0.7) + math.log(0.6)) / 3
    assert log_loss(y_true, y_proba, classes=("A", "B", "C")) == pytest.approx(expected)


def test_evaluate_log_loss_requires_probabilities() -> None:
    with pytest.raises(ValueError, match="requires y_proba"):
        evaluate(["A", "B"], ["A", "B"], [MetricSpec(name="log_loss")])


def test_evaluate_log_loss_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="width"):
        evaluate(
            ["A", "B"],
            ["A", "B"],
            [MetricSpec(name="log_loss")],
            y_proba=[[0.9], [0.1]],
            classes=("A", "B"),
        )


def test_evaluate_log_loss_uses_classes_when_slice_omits_a_label() -> None:
    report = evaluate(
        ["A", "A"],
        ["A", "B"],
        [MetricSpec(name="log_loss")],
        y_proba=[[0.7, 0.3], [0.6, 0.4]],
        classes=("A", "B"),
    )
    expected = -(math.log(0.7) + math.log(0.6)) / 2
    assert report.metrics["log_loss"] == pytest.approx(expected)
    assert report.notes == ("log_loss used y_proba",)


def test_brier_score_perfect_predictions() -> None:
    assert brier_score(["A", "B"], [[1.0, 0.0], [0.0, 1.0]], classes=("A", "B")) == 0.0


def test_brier_score_known_binary_value() -> None:
    # One-hot Brier is (0.2^2 + 0.2^2) + (0.4^2 + 0.4^2) = 0.08 + 0.32 = 0.40
    # mean = 0.20
    score = brier_score(["A", "B"], [[0.8, 0.2], [0.4, 0.6]], classes=("A", "B"))
    assert score == pytest.approx(0.2)


def test_roc_auc_binary_perfect_ranking() -> None:
    score = roc_auc_binary(
        ["neg", "neg", "pos", "pos"],
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        classes=("neg", "pos"),
        pos_label="pos",
    )
    assert score == 1.0


def test_roc_auc_binary_reversed_ranking() -> None:
    score = roc_auc_binary(
        ["neg", "neg", "pos", "pos"],
        [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]],
        classes=("neg", "pos"),
        pos_label="pos",
    )
    assert score == 0.0


def test_roc_auc_binary_constant_scores() -> None:
    score = roc_auc_binary(
        ["neg", "pos", "neg", "pos"],
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        classes=("neg", "pos"),
    )
    assert score == pytest.approx(0.5)


def test_roc_auc_binary_pos_label_selects_column() -> None:
    y_true = ["A", "A", "B", "B"]
    y_proba = [[0.9, 0.5], [0.8, 0.5], [0.1, 0.5], [0.2, 0.5]]
    positive_a = roc_auc_binary(y_true, y_proba, classes=("A", "B"), pos_label="A")
    positive_b = roc_auc_binary(y_true, y_proba, classes=("A", "B"), pos_label="B")
    assert positive_a == 1.0
    assert positive_b == pytest.approx(0.5)


def test_evaluate_roc_auc_binary_accepts_pos_label_param() -> None:
    report = evaluate(
        ["A", "B", "A", "B"],
        ["A", "B", "A", "B"],
        [MetricSpec(name="roc_auc_binary", params={"pos_label": "B"})],
        y_proba=[[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]],
        classes=("A", "B"),
    )
    assert report.metrics["roc_auc_binary"] == 1.0
    assert report.notes == ("roc_auc_binary used y_proba",)


def test_evaluate_roc_auc_binary_rejects_unknown_param() -> None:
    with pytest.raises(ValueError, match="does not accept params"):
        evaluate(
            ["A", "B"],
            ["A", "B"],
            [MetricSpec(name="roc_auc_binary", params={"average": "macro"})],
            y_proba=[[0.9, 0.1], [0.1, 0.9]],
        )


def test_roc_auc_binary_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        roc_auc_binary(["A", "A"], [[0.9, 0.1], [0.8, 0.2]], classes=("A", "B"))


def test_macro_precision_and_recall_on_toy_vectors() -> None:
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    assert macro_precision(y_true, y_pred) == pytest.approx(5 / 6)
    assert macro_recall(y_true, y_pred) == pytest.approx(0.75)


def test_r2_known_value() -> None:
    assert r2([1.0, 2.0, 3.0], [1.0, 3.0, 3.0]) == pytest.approx(0.5)


def test_r2_perfect_and_constant_target() -> None:
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert r2([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) == 1.0
    assert r2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_evaluate_filters_probability_rows_with_missing_pairs() -> None:
    report = evaluate(
        ["A", None, "B"],
        ["A", "B", "B"],
        [MetricSpec(name="log_loss"), MetricSpec(name="accuracy")],
        y_proba=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        classes=("A", "B"),
    )
    assert report.n == 3
    assert report.n_missing == 1
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert report.notes == ("log_loss used y_proba",)


def test_evaluate_mixed_metrics_record_probability_notes() -> None:
    report = evaluate(
        ["A", "B"],
        ["A", "B"],
        [MetricSpec(name="accuracy"), MetricSpec(name="brier_score")],
        y_proba=[[1.0, 0.0], [0.0, 1.0]],
        classes=("A", "B"),
    )
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["brier_score"] == 0.0
    assert report.notes == ("brier_score used y_proba",)


def test_evaluate_dispatches_new_label_metrics() -> None:
    report = evaluate(
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 3.0],
        [MetricSpec(name="r2"), MetricSpec(name="rmse")],
    )
    assert report.metrics["r2"] == pytest.approx(0.5)
    assert report.notes == ()
    class_report = evaluate(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "B"],
        [MetricSpec(name="macro_precision"), MetricSpec(name="macro_recall")],
    )
    assert class_report.metrics["macro_precision"] == pytest.approx(5 / 6)
    assert class_report.metrics["macro_recall"] == pytest.approx(0.75)
