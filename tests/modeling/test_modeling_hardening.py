"""Hardening tests for modeling phase: splits, evaluation, clustering."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence

import pytest

from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.capabilities import Clusterer, Embedder
from ds_platform.modeling.cluster import run_clustering
from ds_platform.modeling.evaluate import evaluate
from ds_platform.modeling.records import (
    cluster_assignment_payload_bytes,
    put_split_assignment,
)
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import MetricSpec, ModelSpec, SplitSpec
from ds_platform.modeling.split import (
    SplitAssignment,
    apply_representation_split,
    load_split_assignment,
    split_assignment_bytes,
    split_entities,
)
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_SOURCE = "a" * 64


class _FixedClusterer:
    def fit(self, table: RepresentationTable) -> None:
        self.dim = table.dim

    def predict(self, table: RepresentationTable) -> list[int]:
        return list(range(len(table.entity_ids)))


class _FixedEmbedder:
    def fit(
        self,
        entity_ids: Sequence[str],
        records: Sequence[Mapping[str, object]],
    ) -> None:
        del entity_ids, records

    def encode(
        self,
        entity_ids: Sequence[str],
        records: Sequence[Mapping[str, object]],
    ) -> RepresentationTable:
        del records
        vectors = tuple((float(index),) for index, _entity_id in enumerate(entity_ids))
        return RepresentationTable(
            entity_ids=tuple(entity_ids),
            vectors=vectors,
            dim=1,
            source_payload_ids=(_SOURCE,) * len(entity_ids),
        )


def test_clusterer_and_embedder_satisfy_protocols() -> None:
    assert isinstance(_FixedClusterer(), Clusterer)
    assert isinstance(_FixedEmbedder(), Embedder)


def test_split_assignment_includes_spec_metadata() -> None:
    spec = SplitSpec(method="holdout", seed=11, test_size=0.25)
    assignment = split_entities([f"e{index}" for index in range(8)], spec)[0]
    assert assignment.method == "holdout"
    assert assignment.seed == 11
    assert assignment.fold_index is None


def test_kfold_assignments_include_fold_index() -> None:
    spec = SplitSpec(method="kfold", seed=3, n_splits=3)
    assignments = split_entities([f"e{index}" for index in range(9)], spec)
    assert [assignment.fold_index for assignment in assignments] == [0, 1, 2]


def test_split_assignment_round_trip() -> None:
    original = SplitAssignment(
        train_ids=("a", "b"),
        validation_ids=(),
        test_ids=("c",),
        method="holdout",
        seed=1,
    )
    loaded = load_split_assignment(split_assignment_bytes(original))
    assert loaded == original


def test_put_split_assignment_persists_metadata(tmp_path) -> None:
    from datetime import UTC, datetime

    store = LocalStore(tmp_path)
    assignment = SplitAssignment(
        train_ids=("a",),
        validation_ids=(),
        test_ids=("b",),
        method="holdout",
        seed=5,
    )
    run = RunContext(
        run_id="split-run",
        project="proj",
        started_at=datetime(2026, 9, 20, tzinfo=UTC),
        environment=Environment.LOCAL,
    )
    payload_id, _record_id = put_split_assignment(
        store,
        assignment,
        run=run,
        inputs=[_SOURCE],
    )
    loaded = load_split_assignment(store.get(payload_id))
    assert loaded.method == "holdout"
    assert loaded.seed == 5


def test_apply_representation_split_slices_vectors() -> None:
    table = RepresentationTable(
        entity_ids=("e1", "e2", "e3"),
        vectors=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        dim=2,
        source_payload_ids=(_SOURCE, _SOURCE, _SOURCE),
    )
    assignment = SplitAssignment(
        train_ids=("e1", "e3"),
        validation_ids=(),
        test_ids=("e2",),
    )
    train, validation, test = apply_representation_split(table, assignment)
    assert train.entity_ids == ("e1", "e3")
    assert train.vectors == ((1.0, 0.0), (1.0, 1.0))
    assert validation.entity_ids == ()
    assert test.entity_ids == ("e2",)
    assert test.vectors == ((0.0, 1.0),)


def test_evaluate_accepts_extra_metrics() -> None:
    def custom_metric(y_true: list[object], y_pred: list[object]) -> float:
        matches = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
        return matches / len(y_true)

    report = evaluate(
        ["A", "B", "A"],
        ["A", "A", "A"],
        [MetricSpec(name="custom_hit_rate")],
        extra_metrics={"custom_hit_rate": custom_metric},
    )
    assert report.metrics["custom_hit_rate"] == pytest.approx(2 / 3)


def test_run_clustering_persists_assignments(tmp_path) -> None:
    from datetime import UTC, datetime

    store = LocalStore(tmp_path)
    table = RepresentationTable(
        entity_ids=("e1", "e2", "e3"),
        vectors=((1.0,), (2.0,), (3.0,)),
        dim=1,
        source_payload_ids=(_SOURCE, _SOURCE, _SOURCE),
    )
    spec = ModelSpec(family="fixed-kmeans", params={"k": 2})
    run = RunContext(
        run_id="cluster-run",
        project="proj",
        started_at=datetime(2026, 9, 20, tzinfo=UTC),
        environment=Environment.LOCAL,
    )
    result = run_clustering(
        spec,
        clusterer=_FixedClusterer(),
        table=table,
        store=store,
        run=run,
        serialize_model=pickle.dumps,
        inputs=[_SOURCE],
    )
    assert store.exists(result.assignment_payload_id)
    assert store.exists(result.model_payload_id)
    assert result.labels == (0, 1, 2)
    payload = store.get(result.assignment_payload_id).decode("utf-8")
    assert "e1" in payload


def test_cluster_assignment_payload_is_deterministic() -> None:
    first = cluster_assignment_payload_bytes(["b", "a"], [1, 0])
    second = cluster_assignment_payload_bytes(["a", "b"], [0, 1])
    assert first == second


def test_hardening_imports_do_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.cluster", _FORBIDDEN)
