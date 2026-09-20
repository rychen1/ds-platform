"""Train-only fitted transforms. Not a feature store or sklearn Pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ds_platform.modeling.features import FeatureTable


@runtime_checkable
class FittedTransform(Protocol):
    """Learned table transform that must be fit on the train split only."""

    def fit(self, table: FeatureTable) -> None: ...

    def transform(self, table: FeatureTable) -> FeatureTable: ...


def apply_fitted_transform(
    transform: FittedTransform,
    train: FeatureTable,
    validation: FeatureTable,
    test: FeatureTable,
) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
    """Fit ``transform`` on ``train`` only, then transform all three slices.

    There is no helper that fits on concatenated rows. Transforms must keep
    each slice's ``entity_ids`` in the same order.
    """
    transform.fit(train)
    transformed_train = transform.transform(train)
    transformed_validation = transform.transform(validation)
    transformed_test = transform.transform(test)
    _require_preserved_entity_ids(train, transformed_train, split="train")
    _require_preserved_entity_ids(
        validation,
        transformed_validation,
        split="validation",
    )
    _require_preserved_entity_ids(test, transformed_test, split="test")
    return transformed_train, transformed_validation, transformed_test


def _require_preserved_entity_ids(
    original: FeatureTable,
    transformed: FeatureTable,
    *,
    split: str,
) -> None:
    if tuple(transformed.entity_ids) != tuple(original.entity_ids):
        raise ValueError(f"fitted transform must preserve {split} entity_ids")
