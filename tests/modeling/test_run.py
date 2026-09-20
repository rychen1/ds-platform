"""Holdout run_experiment orchestration (modeling Part 6)."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

import pytest

from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.run import run_experiment
from ds_platform.modeling.spec import (
    DatasetRef,
    ExperimentSpec,
    FeatureSpec,
    MetricSpec,
    ModelSpec,
    SplitSpec,
    TargetSpec,
    experiment_config_hash,
)
from ds_platform.modeling.split import split_entities
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_DATASET_ID = "a" * 64
_CREATED = datetime(2026, 9, 19, 17, 0, tzinfo=UTC)


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
    def __init__(self) -> None:
        self._majority: str | int = "A"

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None:
        del features
        counts: dict[str | int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        self._majority = max(counts, key=lambda label: counts[label])

    def predict(self, features: FeatureTable) -> list[str | int]:
        return [self._majority] * len(features.entity_ids)


class _FloatRegressor:
    def fit(self, features: FeatureTable, y: Sequence[float]) -> None:
        del features, y

    def predict(self, features: FeatureTable) -> list[float]:
        return [1.0] * len(features.entity_ids)


class _RecordingTransform:
    def __init__(self) -> None:
        self.fit_ids: list[str] = []

    def fit(self, table: FeatureTable) -> None:
        self.fit_ids = list(table.entity_ids)

    def transform(self, table: FeatureTable) -> FeatureTable:
        return table


class _ProbabilisticClassifier:
    classes = ("A", "B")

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None:
        del features, y

    def predict(self, features: FeatureTable) -> list[str | int]:
        positives = {"e1", "e2", "e5"}
        return [
            "A" if entity_id in positives else "B" for entity_id in features.entity_ids
        ]

    def predict_proba(self, features: FeatureTable) -> list[list[float]]:
        positives = {"e1", "e2", "e5"}
        return [
            [0.9, 0.1] if entity_id in positives else [0.2, 0.8]
            for entity_id in features.entity_ids
        ]


def _rows() -> list[dict[str, object]]:
    return [
        {"id": "e1", "x": 1, "label": "A"},
        {"id": "e2", "x": 2, "label": "A"},
        {"id": "e3", "x": 3, "label": "B"},
        {"id": "e4", "x": 4, "label": "B"},
        {"id": "e5", "x": 5, "label": "A"},
    ]


def _experiment(**overrides: object) -> ExperimentSpec:
    defaults = {
        "experiment_id": "exp-001",
        "dataset": DatasetRef(payload_id=_DATASET_ID),
        "features": FeatureSpec(views=("structured",)),
        "target": TargetSpec(column="label", task="classification"),
        "model": ModelSpec(family="baseline"),
        "split": SplitSpec(method="holdout", seed=42, test_size=0.4),
        "metrics": (MetricSpec(name="accuracy"),),
        "seed": 42,
    }
    defaults.update(overrides)
    return ExperimentSpec(**defaults)  # type: ignore[arg-type]


def _run() -> RunContext:
    return RunContext(
        run_id="run-exp-001",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def test_run_experiment_persists_artifact_chain(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment()
    result = run_experiment(
        spec,
        rows=_rows(),
        feature_views={"structured": _StructuredView(_DATASET_ID)},
        adapter=_MajorityClassifier(),
        store=store,
        run=_run(),
        serialize_model=pickle.dumps,
    )
    assert result.config_hash == experiment_config_hash(spec)
    assert store.exists(result.feature_payload_id)
    assert store.exists(result.model_payload_id)
    assert store.exists(result.split_payload_id)
    assert store.exists(result.prediction_payload_id)
    assert store.exists(result.evaluation_payload_id)
    assert "accuracy" in result.report.metrics

    assignment = split_entities([str(row["id"]) for row in _rows()], spec.split)[0]
    prediction_lines = (
        store.get(result.prediction_payload_id).decode("utf-8").strip().split("\n")
    )
    predicted_ids = {json.loads(line)["entity_id"] for line in prediction_lines}
    assert predicted_ids == set(assignment.test_ids)


def test_run_experiment_train_and_test_ids_are_disjoint() -> None:
    spec = _experiment()
    entity_ids = [str(row["id"]) for row in _rows()]
    assignment = split_entities(entity_ids, spec.split)[0]
    assert set(assignment.train_ids).isdisjoint(set(assignment.test_ids))


def test_run_experiment_is_deterministic_for_identical_inputs(tmp_path) -> None:
    spec = _experiment()
    rows = _rows()
    views = {"structured": _StructuredView(_DATASET_ID)}
    run = _run()

    first_store = LocalStore(tmp_path / "first")
    first = run_experiment(
        spec,
        rows=rows,
        feature_views=views,
        adapter=_MajorityClassifier(),
        store=first_store,
        run=run,
        serialize_model=pickle.dumps,
    )
    second_store = LocalStore(tmp_path / "second")
    second = run_experiment(
        spec,
        rows=rows,
        feature_views=views,
        adapter=_MajorityClassifier(),
        store=second_store,
        run=run,
        serialize_model=pickle.dumps,
    )
    assert first.feature_payload_id == second.feature_payload_id
    assert first.model_payload_id == second.model_payload_id
    assert first.split_payload_id == second.split_payload_id
    assert first.prediction_payload_id == second.prediction_payload_id
    assert first.evaluation_payload_id == second.evaluation_payload_id


def test_run_experiment_requires_dataset_payload_id(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment(dataset=DatasetRef(logical_key="proj:dataset:v0"))
    with pytest.raises(ValueError, match="dataset.payload_id"):
        run_experiment(
            spec,
            rows=_rows(),
            feature_views={"structured": _StructuredView(_DATASET_ID)},
            adapter=_MajorityClassifier(),
            store=store,
            run=_run(),
            serialize_model=pickle.dumps,
        )


def test_run_experiment_rejects_kfold_spec(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment(
        split=SplitSpec(method="kfold", seed=42, n_splits=3),
    )
    with pytest.raises(ValueError, match="holdout"):
        run_experiment(
            spec,
            rows=_rows(),
            feature_views={"structured": _StructuredView(_DATASET_ID)},
            adapter=_MajorityClassifier(),
            store=store,
            run=_run(),
            serialize_model=pickle.dumps,
        )


def test_run_experiment_rejects_task_adapter_mismatch(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment(target=TargetSpec(column="label", task="classification"))
    with pytest.raises(TypeError, match="Classifier"):
        run_experiment(
            spec,
            rows=_rows(),
            feature_views={"structured": _StructuredView(_DATASET_ID)},
            adapter=_FloatRegressor(),
            store=store,
            run=_run(),
            serialize_model=pickle.dumps,
        )


def test_run_experiment_fitted_transform_sees_only_train_ids(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment(
        split=SplitSpec(method="holdout", seed=42, test_size=0.4, validation_size=0.2)
    )
    transform = _RecordingTransform()
    result = run_experiment(
        spec,
        rows=_rows(),
        feature_views={"structured": _StructuredView(_DATASET_ID)},
        adapter=_MajorityClassifier(),
        store=store,
        run=_run(),
        serialize_model=pickle.dumps,
        fitted_transform=transform,
    )
    assignment = split_entities([str(row["id"]) for row in _rows()], spec.split)[0]
    assert set(transform.fit_ids) == set(assignment.train_ids)
    assert set(transform.fit_ids).isdisjoint(set(assignment.test_ids))
    assert set(transform.fit_ids).isdisjoint(set(assignment.validation_ids))
    assert store.exists(result.evaluation_payload_id)


def test_run_experiment_scores_probabilities_and_records_notes(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment(
        metrics=(
            MetricSpec(name="accuracy"),
            MetricSpec(name="log_loss"),
        )
    )
    result = run_experiment(
        spec,
        rows=_rows(),
        feature_views={"structured": _StructuredView(_DATASET_ID)},
        adapter=_ProbabilisticClassifier(),
        store=store,
        run=_run(),
        serialize_model=pickle.dumps,
    )
    assert "accuracy" in result.report.metrics
    assert "log_loss" in result.report.metrics
    assert result.report.notes == ("log_loss used y_proba",)

    prediction_lines = (
        store.get(result.prediction_payload_id).decode("utf-8").strip().split("\n")
    )
    for line in prediction_lines:
        row = json.loads(line)
        assert row["y_proba"] is not None
        assert len(row["y_proba"]) == 2

    evaluation_payload = json.loads(
        store.get(result.evaluation_payload_id).decode("utf-8")
    )
    assert evaluation_payload["notes"] == ["log_loss used y_proba"]
    assert "log_loss" in evaluation_payload["metrics"]


def test_run_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.run", _FORBIDDEN)
