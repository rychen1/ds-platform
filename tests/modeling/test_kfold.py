"""K-fold cross-validation orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

import pytest

from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.run import run_kfold
from ds_platform.modeling.spec import (
    DatasetRef,
    ExperimentSpec,
    FeatureSpec,
    MetricSpec,
    ModelSpec,
    SplitSpec,
    TargetSpec,
)
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_DATASET_ID = "c" * 64
_CREATED = datetime(2026, 9, 20, tzinfo=UTC)


class _StructuredView:
    name = "structured"

    def __init__(self, source_payload_id: str) -> None:
        self.source_payload_id = source_payload_id

    def transform(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        as_of: object = None,
    ) -> FeatureTable:
        del as_of
        entity_ids = tuple(str(row["id"]) for row in rows)
        values = cast(
            tuple[tuple[Scalar, ...], ...],
            tuple((row["x"], row["label"]) for row in rows),
        )
        return FeatureTable(
            entity_ids=entity_ids,
            columns=("x", "label"),
            values=values,
            source_payload_ids=(self.source_payload_id,) * len(entity_ids),
        )


class _MajorityClassifier:
    instances: list[_MajorityClassifier] = []

    def __init__(self) -> None:
        self.instances.append(self)
        self._majority: str | int = "A"

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None:
        del features
        counts: dict[str | int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        self._majority = max(counts, key=lambda label: counts[label])

    def predict(self, features: FeatureTable) -> list[str | int]:
        return [self._majority] * len(features.entity_ids)


def _rows() -> list[dict[str, object]]:
    return [
        {"id": "e1", "x": 1, "label": "A"},
        {"id": "e2", "x": 2, "label": "A"},
        {"id": "e3", "x": 3, "label": "B"},
        {"id": "e4", "x": 4, "label": "B"},
        {"id": "e5", "x": 5, "label": "A"},
        {"id": "e6", "x": 6, "label": "B"},
    ]


def _experiment() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="kfold-exp",
        dataset=DatasetRef(payload_id=_DATASET_ID),
        features=FeatureSpec(views=("structured",)),
        target=TargetSpec(column="label", task="classification"),
        model=ModelSpec(family="baseline"),
        split=SplitSpec(method="kfold", seed=11, n_splits=3),
        metrics=(MetricSpec(name="accuracy"),),
        seed=11,
    )


def _run() -> RunContext:
    return RunContext(
        run_id="kfold-run",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def test_run_kfold_aggregates_fold_reports(tmp_path) -> None:
    _MajorityClassifier.instances.clear()
    store = LocalStore(tmp_path)
    result = run_kfold(
        _experiment(),
        rows=_rows(),
        feature_views={"structured": _StructuredView(_DATASET_ID)},
        adapter_factory=_MajorityClassifier,
        store=store,
        run=_run(),
    )
    assert len(result.folds) == 3
    assert result.aggregated_report.notes == ("aggregated from 3 folds",)
    assert "accuracy" in result.aggregated_report.metrics
    assert len(_MajorityClassifier.instances) == 3
    assert store.exists(result.aggregated_evaluation_payload_id)
    for fold in result.folds:
        assert store.exists(fold.split_payload_id)
        assert store.exists(fold.prediction_payload_id)
        assert store.exists(fold.evaluation_payload_id)

    aggregated_payload = json.loads(
        store.get(result.aggregated_evaluation_payload_id).decode("utf-8")
    )
    assert aggregated_payload["notes"] == ["aggregated from 3 folds"]
    assert aggregated_payload["n"] == result.aggregated_report.n


def test_run_kfold_rejects_holdout_spec(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment().model_copy(
        update={"split": SplitSpec(method="holdout", seed=11, test_size=0.3)}
    )
    with pytest.raises(ValueError, match="kfold"):
        run_kfold(
            spec,
            rows=_rows(),
            feature_views={"structured": _StructuredView(_DATASET_ID)},
            adapter_factory=_MajorityClassifier,
            store=store,
            run=_run(),
        )


def test_kfold_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.run", _FORBIDDEN)
