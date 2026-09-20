"""PairTable and ConditionedEncoder alignment (modeling foundation Phase 2)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.pairs import PairTable, align_pairs
from ds_platform.modeling.representations import (
    RepresentationTable,
    as_feature_table,
)
from ds_platform.modeling.spec import ConditioningSpec, spec_config_hash
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "Action",
    "board_game_analysis",
    "numpy",
    "restaurant_intelligence",
    "sklearn",
}

_SOURCE = "c" * 64


def _repr_table(
    entity_ids: tuple[str, ...],
    vectors: tuple[tuple[float, ...], ...],
) -> RepresentationTable:
    return RepresentationTable(
        entity_ids=entity_ids,
        vectors=vectors,
        dim=len(vectors[0]) if vectors else 0,
        source_payload_ids=(_SOURCE,) * len(entity_ids),
    )


class _AddEncoder:
    def fit(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> None:
        del contexts, conditions, pair_ids

    def encode(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> RepresentationTable:
        vectors = tuple(
            tuple(left + right for left, right in zip(context, condition, strict=True))
            for context, condition in zip(
                contexts.vectors,
                conditions.vectors,
                strict=True,
            )
        )
        return RepresentationTable(
            entity_ids=tuple(pair_ids),
            vectors=vectors,
            dim=contexts.dim,
            source_payload_ids=contexts.source_payload_ids,
        )


class _SumRegressor:
    def fit(self, features: FeatureTable, y: Sequence[float]) -> None:
        del features, y

    def predict(self, features: FeatureTable) -> list[float]:
        totals: list[float] = []
        for row in features.values:
            total = 0.0
            for cell in row:
                if isinstance(cell, bool) or not isinstance(cell, int | float):
                    raise TypeError(f"expected numeric cell, got {cell!r}")
                total += float(cell)
            totals.append(total)
        return totals


def test_pair_table_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="pair_ids must be unique"):
        PairTable(
            pair_ids=("q1", "q1"),
            context_ids=("s1", "s2"),
            condition_ids=("a1", "a2"),
            source_payload_ids=(_SOURCE, _SOURCE),
        )


def test_align_pairs_rekeys_to_pair_ids() -> None:
    contexts = _repr_table(("s1", "s2"), ((1.0, 0.0), (2.0, 0.0)))
    conditions = _repr_table(("a1", "a2"), ((0.0, 1.0), (0.0, 3.0)))
    pairs = PairTable(
        pair_ids=("q2", "q1"),
        context_ids=("s2", "s1"),
        condition_ids=("a2", "a1"),
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    aligned_contexts, aligned_conditions = align_pairs(contexts, conditions, pairs)
    assert aligned_contexts.entity_ids == ("q2", "q1")
    assert aligned_contexts.vectors == ((2.0, 0.0), (1.0, 0.0))
    assert aligned_conditions.vectors == ((0.0, 3.0), (0.0, 1.0))


def test_align_pairs_missing_context_raises() -> None:
    contexts = _repr_table(("s1",), ((1.0,),))
    conditions = _repr_table(("a1",), ((0.0,),))
    pairs = PairTable(
        pair_ids=("q1",),
        context_ids=("s9",),
        condition_ids=("a1",),
        source_payload_ids=(_SOURCE,),
    )
    with pytest.raises(KeyError, match="s9"):
        align_pairs(contexts, conditions, pairs)


def test_conditioned_encoder_output_ids_match_pair_ids() -> None:
    contexts = _repr_table(("s1", "s2"), ((1.0, 2.0), (3.0, 4.0)))
    conditions = _repr_table(("a1", "a2"), ((10.0, 20.0), (30.0, 40.0)))
    pairs = PairTable(
        pair_ids=("q1", "q2"),
        context_ids=("s1", "s2"),
        condition_ids=("a1", "a2"),
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    aligned_contexts, aligned_conditions = align_pairs(contexts, conditions, pairs)
    encoded = _AddEncoder().encode(
        aligned_contexts,
        aligned_conditions,
        pairs.pair_ids,
    )
    assert encoded.entity_ids == pairs.pair_ids
    assert encoded.vectors == ((11.0, 22.0), (33.0, 44.0))


def test_v1_regressor_accepts_flattened_pairs() -> None:
    contexts = _repr_table(("s1",), ((1.0, 2.0),))
    conditions = _repr_table(("a1",), ((3.0, 4.0),))
    pairs = PairTable(
        pair_ids=("q1",),
        context_ids=("s1",),
        condition_ids=("a1",),
        source_payload_ids=(_SOURCE,),
    )
    aligned_contexts, aligned_conditions = align_pairs(contexts, conditions, pairs)
    from ds_platform.modeling.representations import align_representation_tables

    features = as_feature_table(
        align_representation_tables((aligned_contexts, aligned_conditions))
    )
    adapter = _SumRegressor()
    adapter.fit(features, [10.0])
    assert adapter.predict(features) == [10.0]


def test_conditioning_spec_hash_is_stable() -> None:
    spec = ConditioningSpec(family="add", seed=1, params={"scale": 1.5})
    assert spec_config_hash(spec) == spec_config_hash(
        ConditioningSpec(family="add", seed=1, params={"scale": 1.5})
    )


def test_pairs_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.pairs", _FORBIDDEN)
    import ds_platform.modeling.pairs as pairs

    assert "Action" not in dir(pairs)
