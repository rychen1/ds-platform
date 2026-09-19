"""End-to-end modeling lifecycle via the public ``ds_platform.modeling`` API."""

from __future__ import annotations

import json
import pickle
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ds_platform import (
    ArtifactKind,
    Environment,
    LocalStore,
    RelationType,
    RunContext,
)
from ds_platform.modeling import (
    Classifier,
    DatasetRef,
    ExperimentSpec,
    FeatureSpec,
    FeatureTable,
    MetricSpec,
    ModelSpec,
    Scalar,
    SplitSpec,
    TargetSpec,
    experiment_config_hash,
    run_experiment,
)

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_DATASET_ID = "c" * 64
_CREATED = datetime(2026, 9, 19, 18, 0, tzinfo=UTC)


class _PrimaryView:
    name = "primary"

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
            tuple((row["feat_a"], row["label"]) for row in rows),
        )
        return FeatureTable(
            entity_ids=entity_ids,
            columns=("feat_a", "label"),
            values=values,
            source_payload_ids=(self.source_payload_id,) * len(entity_ids),
        )


class _SecondaryView:
    name = "secondary"

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
            tuple((row["feat_b"],) for row in rows),
        )
        return FeatureTable(
            entity_ids=entity_ids,
            columns=("feat_b",),
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


def _rows() -> list[dict[str, object]]:
    return [
        {"id": "e1", "feat_a": 1, "feat_b": 10, "label": "A"},
        {"id": "e2", "feat_a": 2, "feat_b": 20, "label": "A"},
        {"id": "e3", "feat_a": 3, "feat_b": 30, "label": "B"},
        {"id": "e4", "feat_a": 4, "feat_b": 40, "label": "B"},
        {"id": "e5", "feat_a": 5, "feat_b": 50, "label": "A"},
        {"id": "e6", "feat_a": 6, "feat_b": 60, "label": "B"},
    ]


def _experiment() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="lifecycle-exp",
        dataset=DatasetRef(payload_id=_DATASET_ID),
        features=FeatureSpec(views=("primary", "secondary")),
        target=TargetSpec(column="label", task="classification"),
        model=ModelSpec(family="baseline"),
        split=SplitSpec(method="holdout", seed=7, test_size=1 / 3),
        metrics=(MetricSpec(name="accuracy"), MetricSpec(name="macro_f1")),
        seed=7,
    )


def _run() -> RunContext:
    return RunContext(
        run_id="run-lifecycle",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def _find_envelope(store_root: Path, payload_id: str) -> dict[str, object]:
    for path in store_root.rglob("*"):
        if not path.is_file() or len(path.name) != 64:
            continue
        try:
            envelope = cast(dict[str, object], json.loads(path.read_bytes()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if envelope.get("payload_id") != payload_id:
            continue
        if "kind" in envelope and "produced_by" in envelope:
            return envelope
    raise AssertionError(f"no envelope found for payload_id {payload_id!r}")


def test_public_modeling_lifecycle_with_two_views(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _experiment()
    views = {
        "primary": _PrimaryView(_DATASET_ID),
        "secondary": _SecondaryView(_DATASET_ID),
    }
    adapter: Classifier = _MajorityClassifier()

    result = run_experiment(
        spec,
        rows=_rows(),
        feature_views=views,
        adapter=adapter,
        store=store,
        run=_run(),
        serialize_model=pickle.dumps,
    )

    assert result.config_hash == experiment_config_hash(spec)

    feature_payload = json.loads(store.get(result.feature_payload_id).decode("utf-8"))
    assert feature_payload["columns"] == ["feat_a", "feat_b"]

    model_payload = store.get(result.model_payload_id)
    reloaded_model = pickle.loads(model_payload)
    assert isinstance(reloaded_model, _MajorityClassifier)
    assert reloaded_model._majority == adapter._majority

    prediction_lines = (
        store.get(result.prediction_payload_id).decode("utf-8").strip().split("\n")
    )
    assert prediction_lines
    for line in prediction_lines:
        row = json.loads(line)
        assert "entity_id" in row
        assert "y_pred" in row

    evaluation_payload = json.loads(
        store.get(result.evaluation_payload_id).decode("utf-8")
    )
    report_metrics = result.report.metrics
    assert evaluation_payload["metrics"]["accuracy"] == report_metrics["accuracy"]
    assert evaluation_payload["metrics"]["macro_f1"] == report_metrics["macro_f1"]

    feature_record = _find_envelope(tmp_path, result.feature_payload_id)
    model_record = _find_envelope(tmp_path, result.model_payload_id)
    prediction_record = _find_envelope(tmp_path, result.prediction_payload_id)
    evaluation_record = _find_envelope(tmp_path, result.evaluation_payload_id)

    assert feature_record["kind"] == ArtifactKind.DATASET
    assert model_record["kind"] == ArtifactKind.MODEL
    assert prediction_record["kind"] == ArtifactKind.PREDICTION
    assert evaluation_record["kind"] == ArtifactKind.EVALUATION

    model_inputs = cast(list[str], model_record["inputs"])
    prediction_inputs = cast(list[str], prediction_record["inputs"])
    evaluation_inputs = cast(list[str], evaluation_record["inputs"])
    assert result.feature_payload_id in model_inputs
    assert result.model_payload_id in prediction_inputs
    assert result.feature_payload_id in prediction_inputs
    assert result.prediction_payload_id in evaluation_inputs
    assert evaluation_record["related"] == [
        {
            "rel": RelationType.EVALUATION_OF,
            "payload_id": result.prediction_payload_id,
        }
    ]

    produced_by = cast(dict[str, object], evaluation_record["produced_by"])
    assert produced_by["config_hash"] == result.config_hash


def test_public_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
