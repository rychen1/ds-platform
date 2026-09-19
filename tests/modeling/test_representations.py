"""RepresentationTable, Embedder helpers, and encoding artifacts."""

from __future__ import annotations

import pickle
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.encode import EncodingResult, run_encoding
from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.records import representation_payload_bytes
from ds_platform.modeling.representations import (
    RepresentationTable,
    align_representation_tables,
    as_feature_table,
    mean_cosine_similarity,
    representation_mse,
    select_entities,
)
from ds_platform.modeling.spec import EncodingSpec, spec_config_hash

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
    "torch",
}

_SOURCE_A = "a" * 64
_SOURCE_B = "b" * 64
_CREATED = datetime(2026, 9, 19, 18, 30, tzinfo=UTC)


def _table(
    *,
    entity_ids: tuple[str, ...] = ("e1", "e2"),
    vectors: tuple[tuple[float, ...], ...] = ((1.0, 0.0), (0.0, 1.0)),
    dim: int = 2,
    source_payload_ids: tuple[str, ...] = (_SOURCE_A, _SOURCE_A),
) -> RepresentationTable:
    return RepresentationTable(
        entity_ids=entity_ids,
        vectors=vectors,
        dim=dim,
        source_payload_ids=source_payload_ids,
    )


class _MeanEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

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
        vectors: list[tuple[float, ...]] = []
        for record in records:
            numeric = [
                float(value)
                for value in record.values()
                if isinstance(value, int | float)
            ]
            mean = sum(numeric) / len(numeric) if numeric else 0.0
            vectors.append(tuple(mean + index for index in range(self.dim)))
        return RepresentationTable(
            entity_ids=tuple(entity_ids),
            vectors=tuple(vectors),
            dim=self.dim,
            source_payload_ids=(_SOURCE_A,) * len(entity_ids),
        )


def test_representation_table_valid_construction() -> None:
    table = _table()
    assert table.entity_ids == ("e1", "e2")
    assert table.dim == 2


def test_representation_table_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="entity_ids must be unique"):
        _table(entity_ids=("e1", "e1"), source_payload_ids=(_SOURCE_A, _SOURCE_A))


def test_representation_table_rejects_mismatched_dim() -> None:
    with pytest.raises(ValidationError, match="length must match dim"):
        _table(vectors=((1.0, 0.0), (1.0,)), dim=2)


def test_align_representation_tables_inner_join_concatenates() -> None:
    left = _table()
    right = _table(
        entity_ids=("e2", "e3"),
        vectors=((9.0,), (8.0,)),
        dim=1,
        source_payload_ids=(_SOURCE_B, _SOURCE_B),
    )
    aligned = align_representation_tables((left, right))
    assert aligned.entity_ids == ("e2",)
    assert aligned.vectors == ((0.0, 1.0, 9.0),)
    assert aligned.dim == 3
    assert aligned.source_payload_ids == (_SOURCE_A,)


def test_select_entities_preserves_requested_order() -> None:
    selected = select_entities(_table(), ("e2", "e1"))
    assert selected.entity_ids == ("e2", "e1")
    assert selected.vectors == ((0.0, 1.0), (1.0, 0.0))


def test_select_entities_missing_id_raises() -> None:
    with pytest.raises(KeyError, match="e9"):
        select_entities(_table(), ("e9",))


def test_representation_mse_and_cosine() -> None:
    left = _table()
    right = _table(vectors=((1.0, 0.0), (0.0, 1.0)))
    assert representation_mse(left, right) == 0.0
    assert mean_cosine_similarity(left, right) == pytest.approx(1.0)


def test_as_feature_table_uses_prefix() -> None:
    features = as_feature_table(_table(), prefix="z")
    assert isinstance(features, FeatureTable)
    assert features.columns == ("z0", "z1")
    assert features.values == ((1.0, 0.0), (0.0, 1.0))


def test_representation_payload_bytes_are_deterministic() -> None:
    table = _table(vectors=((-0.0, float("inf")), (1.25, float("nan"))))
    first = representation_payload_bytes(table)
    second = representation_payload_bytes(table)
    assert first == second
    assert b"$spec_float" in first


def test_run_encoding_persists_model_and_dataset(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = EncodingSpec(family="mean", dim=2, seed=7)
    run = RunContext(
        run_id="run-encode",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )
    result = run_encoding(
        spec,
        encoder=_MeanEmbedder(dim=2),
        entity_ids=("e1", "e2"),
        records=({"x": 1.0}, {"x": 3.0}),
        store=store,
        run=run,
        serialize_model=pickle.dumps,
        inputs=[_SOURCE_A],
    )
    assert isinstance(result, EncodingResult)
    assert result.config_hash == spec_config_hash(spec)
    assert store.exists(result.representation_payload_id)
    assert store.exists(result.model_payload_id)
    reloaded = pickle.loads(store.get(result.model_payload_id))
    assert isinstance(reloaded, _MeanEmbedder)
    assert reloaded.dim == 2


def test_representations_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.representations  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
