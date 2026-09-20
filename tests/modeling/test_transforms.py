"""Leakage-safe fitted transforms."""

from __future__ import annotations

import pytest

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.transforms import FittedTransform, apply_fitted_transform
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_SOURCE = "a" * 64


class _RecordingTransform:
    def __init__(self) -> None:
        self.fit_ids: list[str] = []

    def fit(self, table: FeatureTable) -> None:
        self.fit_ids = list(table.entity_ids)

    def transform(self, table: FeatureTable) -> FeatureTable:
        return table


class _ReorderTransform:
    def fit(self, table: FeatureTable) -> None:
        del table

    def transform(self, table: FeatureTable) -> FeatureTable:
        if len(table.entity_ids) < 2:
            return table
        order = (1, 0, *range(2, len(table.entity_ids)))
        return FeatureTable(
            entity_ids=tuple(table.entity_ids[index] for index in order),
            columns=table.columns,
            values=tuple(table.values[index] for index in order),
            source_payload_ids=tuple(
                table.source_payload_ids[index] for index in order
            ),
        )


def _table(entity_ids: tuple[str, ...]) -> FeatureTable:
    return FeatureTable(
        entity_ids=entity_ids,
        columns=("x",),
        values=tuple((float(index),) for index in range(len(entity_ids))),
        source_payload_ids=(_SOURCE,) * len(entity_ids),
    )


def test_fitted_transform_protocol() -> None:
    assert isinstance(_RecordingTransform(), FittedTransform)


def test_apply_fitted_transform_fits_train_only() -> None:
    transform = _RecordingTransform()
    train = _table(("e1", "e2"))
    validation = _table(("e3",))
    test = _table(("e4", "e5"))
    apply_fitted_transform(transform, train, validation, test)
    assert transform.fit_ids == ["e1", "e2"]
    assert set(transform.fit_ids).isdisjoint({"e3", "e4", "e5"})


def test_apply_fitted_transform_preserves_empty_validation() -> None:
    transform = _RecordingTransform()
    train, validation, test = apply_fitted_transform(
        transform,
        _table(("e1", "e2")),
        _table(()),
        _table(("e3",)),
    )
    assert train.entity_ids == ("e1", "e2")
    assert validation.entity_ids == ()
    assert test.entity_ids == ("e3",)
    assert transform.fit_ids == ["e1", "e2"]


def test_apply_fitted_transform_rejects_reordered_entities() -> None:
    with pytest.raises(ValueError, match="preserve train entity_ids"):
        apply_fitted_transform(
            _ReorderTransform(),
            _table(("e1", "e2")),
            _table(()),
            _table(("e3",)),
        )


def test_transforms_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.transforms", _FORBIDDEN)
