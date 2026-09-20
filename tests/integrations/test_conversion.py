"""Tests for integration FeatureTable conversion helpers."""

from __future__ import annotations

import pytest
from helpers import classification_table

from ds_platform.integrations._conversion import (
    feature_table_to_dataframe,
    resolve_column_names,
)


def test_feature_table_to_dataframe_preserves_columns_and_order() -> None:
    table = classification_table()
    frame = feature_table_to_dataframe(table)
    assert list(frame.columns) == ["x1", "x2", "color"]
    assert list(frame.index) == list(table.entity_ids)


def test_feature_table_to_dataframe_maps_missing_values() -> None:
    table = classification_table(include_missing=True)
    frame = feature_table_to_dataframe(table)
    assert frame.loc["e2", "x2"] != frame.loc["e2", "x2"]  # NaN != NaN
    import pandas as pd

    assert pd.isna(frame.loc["e2", "x2"])


def test_resolve_column_names_rejects_unknown_columns() -> None:
    table = classification_table()
    with pytest.raises(KeyError, match="column not found"):
        resolve_column_names(table, ("missing",))
