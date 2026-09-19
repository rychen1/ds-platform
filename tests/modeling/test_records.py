"""Modeling artifact record helpers (modeling Part 5)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from ds_platform import (
    ArtifactKind,
    Environment,
    LocalStore,
    RelationType,
    RunContext,
    payload_id,
    record_id_from_bytes,
)
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.records import (
    PredictionRow,
    evaluation_report_bytes,
    prediction_payload_bytes,
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    put_prediction_artifact,
)

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_CREATED = datetime(2026, 9, 19, 16, 0, tzinfo=UTC)
_INPUT_A = "a" * 64
_INPUT_B = "b" * 64
_LOGICAL_KEY = "proj:features/structured:v0"


def _run() -> RunContext:
    return RunContext(
        run_id="records-run",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def test_put_model_artifact_writes_model_kind(tmp_path) -> None:
    store = LocalStore(tmp_path)
    data = b"model-bytes"
    payload_id_value, record_id_value = put_model_artifact(
        store,
        data,
        run=_run(),
        inputs=[_INPUT_A],
        media_type="application/octet-stream",
        logical_key="proj:model/baseline:v0",
    )
    assert payload_id_value == payload_id(data)
    assert store.get(payload_id_value) == data
    record_bytes = store.get(record_id_value)
    record = json.loads(record_bytes.decode("utf-8"))
    assert record["kind"] == ArtifactKind.MODEL
    assert record["inputs"] == [_INPUT_A]


def test_put_feature_dataset_writes_dataset_kind(tmp_path) -> None:
    store = LocalStore(tmp_path)
    data = b'{"entity_ids":["e1"],"columns":["x"]}\n'
    payload_id_value, record_id_value = put_feature_dataset(
        store,
        data,
        run=_run(),
        inputs=[_INPUT_A, _INPUT_B],
        media_type="application/jsonl",
        logical_key=_LOGICAL_KEY,
    )
    record = json.loads(store.get(record_id_value).decode("utf-8"))
    assert record["kind"] == ArtifactKind.DATASET
    assert record["inputs"] == [_INPUT_A, _INPUT_B]
    assert record["logical_key"] == _LOGICAL_KEY


def test_put_prediction_artifact_writes_prediction_kind(tmp_path) -> None:
    store = LocalStore(tmp_path)
    data = prediction_payload_bytes(
        [
            PredictionRow(entity_id="e2", y_pred=0),
            PredictionRow(entity_id="e1", y_pred=1, y_true="A"),
        ]
    )
    payload_id_value, record_id_value = put_prediction_artifact(
        store,
        data,
        run=_run(),
        inputs=[_INPUT_A],
        media_type="application/jsonl",
    )
    record = json.loads(store.get(record_id_value).decode("utf-8"))
    assert record["kind"] == ArtifactKind.PREDICTION
    assert store.get(payload_id_value) == data


def test_put_evaluation_artifact_sets_evaluation_of(tmp_path) -> None:
    store = LocalStore(tmp_path)
    report = EvaluationReport(metrics={"accuracy": 0.75}, n=4, n_missing=0)
    subject_payload_id = _INPUT_B
    payload_id_value, record_id_value = put_evaluation_artifact(
        store,
        report,
        run=_run(),
        subject_payload_id=subject_payload_id,
        inputs=[subject_payload_id, _INPUT_A],
        logical_key="proj:evaluation/baseline:v0",
    )
    assert payload_id_value == payload_id(evaluation_report_bytes(report))
    record = json.loads(store.get(record_id_value).decode("utf-8"))
    assert record["kind"] == ArtifactKind.EVALUATION
    assert record["inputs"] == [subject_payload_id, _INPUT_A]
    assert record["related"] == [
        {
            "rel": RelationType.EVALUATION_OF,
            "payload_id": subject_payload_id,
        }
    ]


def test_prediction_payload_bytes_is_deterministic() -> None:
    rows = [
        PredictionRow(entity_id="e2", y_pred=0),
        PredictionRow(entity_id="e1", y_pred=1, y_true="A"),
    ]
    first = prediction_payload_bytes(rows)
    second = prediction_payload_bytes(list(reversed(rows)))
    assert first == second
    assert first.decode("utf-8").count("\n") == 2


def test_put_record_helpers_repair_corrupted_bytes(tmp_path) -> None:
    store = LocalStore(tmp_path)
    data = b"first-model"
    put_model_artifact(
        store,
        data,
        run=_run(),
        inputs=[_INPUT_A],
        media_type="application/octet-stream",
    )
    pid = payload_id(data)
    colliding = tmp_path / pid[:2] / pid[2:4] / pid
    colliding.write_bytes(b"tampered")
    store.put(pid, data, media_type="application/octet-stream")
    assert store.get(pid) == data


def test_records_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.records  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()


def test_put_record_round_trip_uses_existing_put_record(tmp_path) -> None:
    store = LocalStore(tmp_path)
    data = b"round-trip"
    payload_id_value, record_id_value = put_model_artifact(
        store,
        data,
        run=_run(),
        inputs=[],
        media_type="application/octet-stream",
    )
    record_bytes = store.get(record_id_value)
    assert record_id_value == record_id_from_bytes(record_bytes)
    assert payload_id_value == payload_id(data)
