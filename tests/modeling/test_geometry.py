"""Collection geometry, Clusterer, and index artifacts (foundation Phase 7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.modeling.geometry import knn, novelty_scores, pairwise_distances
from ds_platform.modeling.records import put_index_artifact
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import GeometrySpec, spec_config_hash
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "faiss",
    "numpy",
    "restaurant_intelligence",
    "sklearn",
}

_SOURCE = "11" * 32
_CREATED = datetime(2026, 9, 19, 19, 30, tzinfo=UTC)


def _table() -> RepresentationTable:
    return RepresentationTable(
        entity_ids=("e1", "e2", "e3"),
        vectors=((0.0, 0.0), (1.0, 0.0), (10.0, 0.0)),
        dim=2,
        source_payload_ids=(_SOURCE, _SOURCE, _SOURCE),
    )


class _ThresholdClusterer:
    def __init__(self) -> None:
        self._threshold = 5.0

    def fit(self, table: RepresentationTable) -> None:
        self._threshold = 5.0
        del table

    def predict(self, table: RepresentationTable) -> list[str | int]:
        return [0 if vector[0] < self._threshold else 1 for vector in table.vectors]


def test_pairwise_distances_are_symmetric() -> None:
    matrix = pairwise_distances(_table(), metric="l2")
    assert len(matrix) == 3
    for i, row in enumerate(matrix):
        assert row[i] == pytest.approx(0.0)
        for j, value in enumerate(row):
            assert value == pytest.approx(matrix[j][i])


def test_knn_excludes_self() -> None:
    neighbors = knn(_table(), k=1, metric="l2")
    assert neighbors.columns == ("neighbor_1", "distance_1")
    assert neighbors.values[0][0] == "e2"
    for entity_id, row in zip(neighbors.entity_ids, neighbors.values, strict=True):
        assert row[0] != entity_id


def test_novelty_is_higher_for_isolated_point() -> None:
    scores = novelty_scores(_table(), k=1, metric="l2")
    by_id = {
        entity_id: score
        for entity_id, (score, *_) in zip(scores.entity_ids, scores.values, strict=True)
    }
    assert isinstance(by_id["e3"], int | float)
    assert isinstance(by_id["e1"], int | float)
    assert isinstance(by_id["e2"], int | float)
    assert by_id["e3"] > by_id["e1"]
    assert by_id["e3"] > by_id["e2"]


def test_clusterer_thresholds_first_dim() -> None:
    clusterer = _ThresholdClusterer()
    clusterer.fit(_table())
    assert clusterer.predict(_table()) == [0, 0, 1]


def test_put_index_artifact_uses_index_kind(tmp_path) -> None:
    store = LocalStore(tmp_path)
    run = RunContext(
        run_id="run-index",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )
    payload = json.dumps({"neighbors": ["e2"]}).encode("utf-8")
    source_id = "a" * 64
    payload_id, record_id = put_index_artifact(
        store,
        payload,
        run=run,
        inputs=[source_id],
        media_type="application/json",
        logical_key="proj:index/knn:v0",
    )
    record = json.loads(store.get(record_id).decode("utf-8"))
    assert record["kind"] == ArtifactKind.INDEX
    assert record["inputs"] == [source_id]
    assert store.get(payload_id) == payload


def test_geometry_spec_hash() -> None:
    spec = GeometrySpec(metric="cosine", k=2, family="dense")
    assert spec_config_hash(spec) == spec_config_hash(
        GeometrySpec(metric="cosine", k=2, family="dense")
    )


def test_geometry_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.geometry", _FORBIDDEN)
