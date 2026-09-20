"""Model comparison and fold aggregation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ds_platform import ArtifactKind, Environment, LocalStore, RelationType, RunContext
from ds_platform.modeling.compare import (
    aggregate_fold_reports,
    compare_evaluations,
    paired_bootstrap,
)
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.records import put_comparison_artifact
from ds_platform.modeling.spec import MetricSpec
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_LEFT_ID = "a" * 64
_RIGHT_ID = "b" * 64
_CREATED = datetime(2026, 9, 20, tzinfo=UTC)


def _report(*, accuracy: float, n: int = 4) -> EvaluationReport:
    return EvaluationReport(metrics={"accuracy": accuracy}, n=n, n_missing=0)


def _run() -> RunContext:
    return RunContext(
        run_id="compare-run",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def test_compare_evaluations_returns_metric_deltas() -> None:
    comparison = compare_evaluations(
        _report(accuracy=0.75),
        _report(accuracy=0.50),
        left_payload_id=_LEFT_ID,
        right_payload_id=_RIGHT_ID,
    )
    assert comparison.method == "holdout_delta"
    assert comparison.deltas["accuracy"] == pytest.approx(0.25)
    assert comparison.n == 4
    assert comparison.interval_low is None


def test_compare_evaluations_requires_matching_metric_names() -> None:
    left = EvaluationReport(metrics={"accuracy": 1.0}, n=2, n_missing=0)
    right = EvaluationReport(metrics={"rmse": 0.1}, n=2, n_missing=0)
    with pytest.raises(ValueError, match="same metric names"):
        compare_evaluations(
            left,
            right,
            left_payload_id=_LEFT_ID,
            right_payload_id=_RIGHT_ID,
        )


def test_paired_bootstrap_is_seed_stable() -> None:
    metrics = (MetricSpec(name="accuracy"),)
    y_true = ["A", "B", "A", "B"]
    y_pred_left = ["A", "B", "A", "B"]
    y_pred_right = ["A", "A", "A", "B"]
    kwargs = {
        "y_true": y_true,
        "y_pred_left": y_pred_left,
        "y_pred_right": y_pred_right,
        "metrics": metrics,
        "left_payload_id": _LEFT_ID,
        "right_payload_id": _RIGHT_ID,
        "seed": 7,
        "n_resamples": 50,
    }
    first = paired_bootstrap(**kwargs)
    second = paired_bootstrap(**kwargs)
    assert first.deltas == second.deltas
    assert first.interval_low == second.interval_low
    assert first.interval_high == second.interval_high
    assert first.method == "paired_bootstrap"
    assert first.seed == 7


def test_aggregate_fold_reports_returns_mean_metrics() -> None:
    aggregated = aggregate_fold_reports(
        [
            EvaluationReport(metrics={"accuracy": 1.0, "rmse": 2.0}, n=2, n_missing=0),
            EvaluationReport(metrics={"accuracy": 0.5, "rmse": 4.0}, n=3, n_missing=1),
        ]
    )
    assert aggregated.metrics["accuracy"] == pytest.approx(0.75)
    assert aggregated.metrics["rmse"] == pytest.approx(3.0)
    assert aggregated.n == 5
    assert aggregated.n_missing == 1
    assert aggregated.notes == ("aggregated from 2 folds",)


def test_put_comparison_artifact_cites_both_evaluations(tmp_path) -> None:
    store = LocalStore(tmp_path)
    comparison = compare_evaluations(
        _report(accuracy=0.8),
        _report(accuracy=0.6),
        left_payload_id=_LEFT_ID,
        right_payload_id=_RIGHT_ID,
    )
    payload_id, record_id = put_comparison_artifact(
        store,
        comparison,
        run=_run(),
        inputs=[_LEFT_ID, _RIGHT_ID],
    )
    record = json.loads(store.get(record_id).decode("utf-8"))
    payload = json.loads(store.get(payload_id).decode("utf-8"))
    assert record["kind"] == ArtifactKind.EVALUATION
    assert set(record["inputs"]) == {_LEFT_ID, _RIGHT_ID}
    assert {item["payload_id"] for item in record["related"]} == {
        _LEFT_ID,
        _RIGHT_ID,
    }
    assert all(item["rel"] == RelationType.EVALUATION_OF for item in record["related"])
    assert payload["method"] == "holdout_delta"


def test_compare_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.compare", _FORBIDDEN)
