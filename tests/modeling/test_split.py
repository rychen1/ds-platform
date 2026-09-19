"""Entity split helpers (modeling Part 4)."""

from __future__ import annotations

import sys
from datetime import date

import pytest

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.spec import SplitSpec
from ds_platform.modeling.split import SplitAssignment, apply_split, split_entities

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_SOURCE = "a" * 64


def _holdout_spec(**overrides: object) -> SplitSpec:
    defaults = {
        "method": "holdout",
        "seed": 42,
        "test_size": 0.25,
    }
    defaults.update(overrides)
    return SplitSpec(**defaults)  # type: ignore[arg-type]


def _kfold_spec(**overrides: object) -> SplitSpec:
    defaults = {
        "method": "kfold",
        "seed": 7,
        "n_splits": 3,
    }
    defaults.update(overrides)
    return SplitSpec(**defaults)  # type: ignore[arg-type]


def test_split_entities_holdout_is_deterministic() -> None:
    entity_ids = [f"e{index}" for index in range(8)]
    spec = _holdout_spec()
    first = split_entities(entity_ids, spec)
    second = split_entities(entity_ids, spec)
    assert first == second


def test_split_entities_holdout_ids_are_disjoint() -> None:
    entity_ids = [f"e{index}" for index in range(8)]
    assignment = split_entities(entity_ids, _holdout_spec(test_size=0.25))[0]
    train = set(assignment.train_ids)
    test = set(assignment.test_ids)
    assert train.isdisjoint(test)
    assert train | test == set(entity_ids)
    assert assignment.validation_ids == ()


def test_split_entities_stratify_preserves_classes_in_test() -> None:
    entity_ids = ["e1", "e2", "e3", "e4"]
    labels = ["A", "A", "B", "B"]
    assignment = split_entities(
        entity_ids,
        _holdout_spec(test_size=0.5, stratify=True),
        labels=labels,
    )[0]
    test_labels = {
        labels[entity_ids.index(entity_id)] for entity_id in assignment.test_ids
    }
    assert test_labels == {"A", "B"}


def test_split_entities_temporal_cutoff_drops_later_entities() -> None:
    entity_ids = ["e1", "e2", "e3"]
    timestamps = [date(2026, 1, 1), date(2026, 6, 1), date(2026, 12, 1)]
    assignment = split_entities(
        entity_ids,
        _holdout_spec(as_of=date(2026, 6, 1)),
        timestamps=timestamps,
    )[0]
    kept = set(assignment.train_ids) | set(assignment.test_ids)
    assert kept == {"e1", "e2"}
    assert "e3" not in kept


def test_split_entities_requires_timestamps_when_as_of_set() -> None:
    with pytest.raises(ValueError, match="timestamps are required"):
        split_entities(["e1"], _holdout_spec(as_of=date(2026, 1, 1)))


def test_split_entities_kfold_returns_n_assignments() -> None:
    entity_ids = [f"e{index}" for index in range(6)]
    assignments = split_entities(entity_ids, _kfold_spec())
    assert len(assignments) == 3
    for assignment in assignments:
        assert assignment.validation_ids == ()
        assert set(assignment.train_ids) | set(assignment.test_ids) == set(entity_ids)
        assert set(assignment.train_ids).isdisjoint(set(assignment.test_ids))


def test_apply_split_preserves_columns_and_sources() -> None:
    table = FeatureTable(
        entity_ids=("e1", "e2", "e3", "e4"),
        columns=("x", "y"),
        values=((1, 2), (3, 4), (5, 6), (7, 8)),
        source_payload_ids=(_SOURCE, _SOURCE, _SOURCE, _SOURCE),
    )
    assignment = SplitAssignment(
        train_ids=("e1", "e3"),
        validation_ids=(),
        test_ids=("e2", "e4"),
    )
    train, validation, test = apply_split(table, assignment)
    assert train.columns == table.columns
    assert train.entity_ids == ("e1", "e3")
    assert train.values == ((1, 2), (5, 6))
    assert train.source_payload_ids == (_SOURCE, _SOURCE)
    assert validation.entity_ids == ()
    assert test.entity_ids == ("e2", "e4")
    assert test.values == ((3, 4), (7, 8))


def test_split_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.split  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
