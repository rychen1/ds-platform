"""Entity split helpers (modeling Part 4)."""

from __future__ import annotations

import subprocess
import sys
from datetime import date

import pytest
from pydantic import ValidationError

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.spec import SplitSpec
from ds_platform.modeling.split import (
    SplitAssignment,
    apply_split,
    split_entities,
    split_groups,
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


def test_split_entities_temporal_method_puts_later_ids_in_test() -> None:
    entity_ids = ["early", "mid", "late"]
    timestamps = [date(2026, 1, 1), date(2026, 6, 1), date(2026, 12, 1)]
    assignment = split_entities(
        entity_ids,
        SplitSpec(method="temporal", seed=0, test_size=1 / 3),
        timestamps=timestamps,
    )[0]
    assert assignment.train_ids == ("early", "mid")
    assert assignment.test_ids == ("late",)


def test_split_entities_stratify_is_stable_under_label_reorder() -> None:
    spec = SplitSpec(method="holdout", seed=42, test_size=0.5, stratify=True)
    first = split_entities(
        [f"a{index}" for index in range(6)] + [f"b{index}" for index in range(6)],
        spec,
        labels=["A"] * 6 + ["B"] * 6,
    )[0]
    second = split_entities(
        [f"b{index}" for index in range(6)] + [f"a{index}" for index in range(6)],
        spec,
        labels=["B"] * 6 + ["A"] * 6,
    )[0]
    assert set(first.test_ids) == set(second.test_ids)


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


def test_split_groups_holdout_is_order_invariant() -> None:
    spec = _holdout_spec(seed=0, test_size=1 / 3)
    forward = split_groups(["a", "a", "b", "b", "c", "c"], spec)[0]
    reverse = split_groups(list(reversed(["a", "a", "b", "b", "c", "c"])), spec)[0]
    shuffled = split_groups(["c", "a", "b", "c", "a", "b"], spec)[0]
    assert set(forward.train_ids) == set(reverse.train_ids) == set(shuffled.train_ids)
    assert set(forward.test_ids) == set(reverse.test_ids) == set(shuffled.test_ids)


def test_split_groups_is_stable_under_python_hash_seed() -> None:
    script = """
from ds_platform.modeling.spec import SplitSpec
from ds_platform.modeling.split import split_groups

spec = SplitSpec(method="holdout", seed=0, test_size=0.3)
assignment = split_groups(["a", "a", "b", "b", "c", "c"], spec)[0]
print(",".join(sorted(assignment.train_ids)))
"""
    import os

    outputs = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("0", "1", "424242")
    }
    assert len(set(outputs)) == 1


def test_split_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.split", _FORBIDDEN)


def test_holdout_requires_at_least_two_entities() -> None:
    with pytest.raises(ValueError, match="at least two entities"):
        split_entities(["only"], _holdout_spec())


def test_as_of_filter_dropping_all_entities_errors() -> None:
    with pytest.raises(ValueError, match="no entities remain"):
        split_entities(
            ["e1"],
            _holdout_spec(as_of=date(2020, 1, 1)),
            timestamps=[date(2021, 1, 1)],
        )


def test_split_groups_rejects_conflicting_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="conflicting labels"):
        split_groups(
            ["g1", "g1"],
            _holdout_spec(test_size=0.5),
            labels=["A", "B"],
        )


def test_holdout_three_way_ids_are_disjoint() -> None:
    entity_ids = [f"e{index}" for index in range(12)]
    assignment = split_entities(
        entity_ids,
        _holdout_spec(test_size=0.25, validation_size=0.25),
    )[0]
    train = set(assignment.train_ids)
    validation = set(assignment.validation_ids)
    test = set(assignment.test_ids)
    assert train and validation and test
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(entity_ids)


def test_holdout_three_way_is_seed_stable() -> None:
    entity_ids = [f"e{index}" for index in range(12)]
    spec = _holdout_spec(test_size=0.25, validation_size=0.25)
    assert split_entities(entity_ids, spec) == split_entities(entity_ids, spec)


def test_stratified_three_way_keeps_classes_in_train() -> None:
    entity_ids = [f"a{index}" for index in range(6)] + [
        f"b{index}" for index in range(6)
    ]
    labels = ["A"] * 6 + ["B"] * 6
    assignment = split_entities(
        entity_ids,
        _holdout_spec(test_size=1 / 3, validation_size=1 / 3, stratify=True),
        labels=labels,
    )[0]
    train_labels = {
        labels[entity_ids.index(entity_id)] for entity_id in assignment.train_ids
    }
    assert train_labels == {"A", "B"}
    assert assignment.validation_ids
    assert assignment.test_ids


def test_stratified_three_way_errors_when_train_would_drop_a_class() -> None:
    with pytest.raises(ValueError, match="non-empty train assignment"):
        split_entities(
            ["a1", "a2", "b1", "b2"],
            _holdout_spec(test_size=0.4, validation_size=0.4, stratify=True),
            labels=["A", "A", "B", "B"],
        )


def test_temporal_three_way_uses_later_ids_for_test() -> None:
    entity_ids = ["t1", "t2", "t3", "t4"]
    timestamps = [
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 1),
        date(2026, 4, 1),
    ]
    assignment = split_entities(
        entity_ids,
        SplitSpec(
            method="temporal",
            seed=0,
            test_size=0.25,
            validation_size=0.25,
        ),
        timestamps=timestamps,
    )[0]
    assert assignment.train_ids == ("t1", "t2")
    assert assignment.validation_ids == ("t3",)
    assert assignment.test_ids == ("t4",)


def test_kfold_rejects_validation_size() -> None:
    with pytest.raises(ValidationError, match="kfold split does not support"):
        SplitSpec(method="kfold", seed=1, n_splits=3, validation_size=0.2)
