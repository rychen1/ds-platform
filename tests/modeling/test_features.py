"""FeatureTable, FeatureView, and feature helper functions (modeling Part 2)."""

from __future__ import annotations

import sys
from datetime import date

import pytest
from pydantic import ValidationError

from ds_platform.modeling.features import (
    FeatureTable,
    FeatureView,
    Scalar,
    align_feature_tables,
    extract_column,
    select_columns,
)

_FORBIDDEN = {
    "board_game_analysis",
    "boto3",
    "catboost",
    "lightgbm",
    "mlflow",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
    "torch",
    "xgboost",
}

_SOURCE_A = "a" * 64
_SOURCE_B = "b" * 64


def _table(
    *,
    entity_ids: tuple[str, ...] = ("e1", "e2"),
    columns: tuple[str, ...] = ("x", "y"),
    values: tuple[tuple[Scalar, ...], ...] = ((1, 2), (3, 4)),
    source_payload_ids: tuple[str, ...] = (_SOURCE_A, _SOURCE_A),
) -> FeatureTable:
    return FeatureTable(
        entity_ids=entity_ids,
        columns=columns,
        values=values,
        source_payload_ids=source_payload_ids,
    )


def test_feature_table_valid_construction() -> None:
    table = _table()
    assert table.entity_ids == ("e1", "e2")
    assert table.columns == ("x", "y")
    assert table.values == ((1, 2), (3, 4))
    assert table.source_payload_ids == (_SOURCE_A, _SOURCE_A)


def test_feature_table_rejects_mismatched_entity_row_lengths() -> None:
    with pytest.raises(ValidationError, match="values row count"):
        FeatureTable(
            entity_ids=("e1", "e2"),
            columns=("x",),
            values=((1,),),
            source_payload_ids=(_SOURCE_A, _SOURCE_A),
        )


def test_feature_table_rejects_mismatched_column_widths() -> None:
    with pytest.raises(ValidationError, match="values row 0 width"):
        FeatureTable(
            entity_ids=("e1",),
            columns=("x", "y"),
            values=((1,),),
            source_payload_ids=(_SOURCE_A,),
        )


def test_feature_table_rejects_mismatched_source_payload_ids() -> None:
    with pytest.raises(ValidationError, match="source_payload_ids length"):
        FeatureTable(
            entity_ids=("e1",),
            columns=("x",),
            values=((1,),),
            source_payload_ids=(_SOURCE_A, _SOURCE_B),
        )


def test_feature_table_rejects_duplicate_entity_ids() -> None:
    with pytest.raises(ValidationError, match="entity_ids must be unique"):
        FeatureTable(
            entity_ids=("e1", "e1"),
            columns=("x",),
            values=((1,), (2,)),
            source_payload_ids=(_SOURCE_A, _SOURCE_B),
        )


def test_feature_table_rejects_duplicate_columns() -> None:
    with pytest.raises(ValidationError, match="columns must be unique"):
        FeatureTable(
            entity_ids=("e1",),
            columns=("x", "x"),
            values=((1, 2),),
            source_payload_ids=(_SOURCE_A,),
        )


def test_feature_table_allows_empty_table() -> None:
    table = FeatureTable(
        entity_ids=(),
        columns=(),
        values=(),
        source_payload_ids=(),
    )
    assert table.entity_ids == ()
    assert table.columns == ()
    assert table.values == ()
    assert table.source_payload_ids == ()


class _CountingView:
    name = "counting"

    def __init__(self) -> None:
        self.last_as_of: date | None = None

    def transform(
        self,
        rows: list[dict[str, object]],
        *,
        as_of: date | None = None,
    ) -> FeatureTable:
        self.last_as_of = as_of
        entity_ids = tuple(str(row["id"]) for row in rows)
        return FeatureTable(
            entity_ids=entity_ids,
            columns=("n",),
            values=tuple((len(rows),) for _ in entity_ids),
            source_payload_ids=(_SOURCE_A,) * len(entity_ids),
        )


def test_feature_view_protocol_shape() -> None:
    view = _CountingView()
    assert isinstance(view, FeatureView)
    table = view.transform([{"id": "e1"}], as_of=date(2026, 3, 14))
    assert view.last_as_of == date(2026, 3, 14)
    assert table.columns == ("n",)


def test_select_columns_preserves_alignment_and_sources() -> None:
    table = _table()
    selected = select_columns(table, ("y", "x"))
    assert selected.entity_ids == table.entity_ids
    assert selected.columns == ("y", "x")
    assert selected.values == ((2, 1), (4, 3))
    assert selected.source_payload_ids == table.source_payload_ids


def test_select_columns_missing_column_raises_key_error() -> None:
    with pytest.raises(KeyError, match="column not found: 'missing'"):
        select_columns(_table(), ("x", "missing"))


def test_select_columns_empty_selection() -> None:
    selected = select_columns(_table(), ())
    assert selected.columns == ()
    assert selected.values == ((), ())
    assert selected.source_payload_ids == _table().source_payload_ids


def test_extract_column_returns_aligned_values() -> None:
    assert extract_column(_table(), "y") == (2, 4)


def test_extract_column_missing_raises_key_error() -> None:
    with pytest.raises(KeyError, match="column not found: 'missing'"):
        extract_column(_table(), "missing")


def test_align_feature_tables_inner_join_and_order() -> None:
    left = FeatureTable(
        entity_ids=("e2", "e1", "e3"),
        columns=("shared", "left_only"),
        values=((1, 10), (2, 20), (3, 30)),
        source_payload_ids=(_SOURCE_A, _SOURCE_B, _SOURCE_A),
    )
    right = FeatureTable(
        entity_ids=("e1", "e2", "e4"),
        columns=("shared", "right_only"),
        values=((100, 1000), (200, 2000), (400, 4000)),
        source_payload_ids=(_SOURCE_B, _SOURCE_B, _SOURCE_B),
    )
    aligned = align_feature_tables([left, right])
    assert aligned.entity_ids == ("e2", "e1")
    assert aligned.columns == (
        "shared",
        "left_only",
        "v1__shared",
        "right_only",
    )
    assert aligned.values == (
        (1, 10, 200, 2000),
        (2, 20, 100, 1000),
    )
    assert aligned.source_payload_ids == (_SOURCE_A, _SOURCE_B)


def test_align_feature_tables_prefixes_colliding_columns() -> None:
    first = FeatureTable(
        entity_ids=("e1",),
        columns=("score",),
        values=((1,),),
        source_payload_ids=(_SOURCE_A,),
    )
    second = FeatureTable(
        entity_ids=("e1",),
        columns=("score", "extra"),
        values=((2, 3),),
        source_payload_ids=(_SOURCE_B,),
    )
    aligned = align_feature_tables([first, second])
    assert aligned.columns == ("score", "v1__score", "extra")
    assert aligned.values == ((1, 2, 3),)
    assert aligned.source_payload_ids == (_SOURCE_A,)


def test_align_feature_tables_is_deterministic() -> None:
    tables = [
        FeatureTable(
            entity_ids=("e1", "e2"),
            columns=("a",),
            values=((1,), (2,)),
            source_payload_ids=(_SOURCE_A, _SOURCE_A),
        ),
        FeatureTable(
            entity_ids=("e2", "e1"),
            columns=("b",),
            values=((20,), (10,)),
            source_payload_ids=(_SOURCE_B, _SOURCE_B),
        ),
    ]
    first = align_feature_tables(tables)
    second = align_feature_tables(tables)
    assert first == second


def test_align_feature_tables_empty_inputs() -> None:
    assert align_feature_tables([]) == FeatureTable(
        entity_ids=(),
        columns=(),
        values=(),
        source_payload_ids=(),
    )
    single = _table()
    assert align_feature_tables([single]) is single


def test_align_feature_tables_no_overlap_returns_empty() -> None:
    left = FeatureTable(
        entity_ids=("e1",),
        columns=("x",),
        values=((1,),),
        source_payload_ids=(_SOURCE_A,),
    )
    right = FeatureTable(
        entity_ids=("e2",),
        columns=("y",),
        values=((2,),),
        source_payload_ids=(_SOURCE_B,),
    )
    aligned = align_feature_tables([left, right])
    assert aligned.entity_ids == ()
    assert aligned.columns == ()
    assert aligned.values == ()
    assert aligned.source_payload_ids == ()


def test_align_feature_tables_rejects_unsupported_how() -> None:
    with pytest.raises(ValueError, match="unsupported align how"):
        align_feature_tables([_table()], how="outer")  # type: ignore[arg-type]


def test_features_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.features  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
