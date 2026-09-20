"""FeatureTable conversion helpers for vendor integrations."""

from __future__ import annotations

from collections.abc import Sequence

from ds_platform.modeling.features import FeatureTable, Scalar


def feature_table_to_dataframe(table: FeatureTable):
    """Convert a :class:`FeatureTable` to a pandas DataFrame.

    Column order and names match ``table.columns``. Row order matches
    ``table.entity_ids``. ``None`` cells become NaN. Booleans become ints.
    """
    import pandas as pd

    if not table.columns:
        return pd.DataFrame(index=list(table.entity_ids))

    columns: dict[str, list[object]] = {name: [] for name in table.columns}
    column_indexes = list(range(len(table.columns)))
    for row in table.values:
        for column_index in column_indexes:
            cell = row[column_index]
            columns[table.columns[column_index]].append(_cell_to_frame_value(cell))

    frame = pd.DataFrame(columns, index=list(table.entity_ids))
    frame.index.name = "entity_id"
    for column_name in frame.columns:
        series = frame[column_name]
        if bool(series.isna().all()):
            frame[column_name] = series.astype(float)
    return frame


def apply_categorical_columns(frame, column_names: Sequence[str]):
    """Return a copy with named columns cast to pandas ``category`` dtype."""

    if not column_names:
        return frame
    result = frame.copy()
    for column_name in column_names:
        if column_name not in result.columns:
            raise KeyError(f"categorical column not found: {column_name!r}")
        result[column_name] = result[column_name].astype("category")
    return result


def resolve_column_names(
    table: FeatureTable,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate integration column names against ``table.columns``."""
    if columns is None:
        return ()
    resolved: list[str] = []
    available = set(table.columns)
    for column in columns:
        if column not in available:
            raise KeyError(f"column not found: {column!r}")
        resolved.append(column)
    return tuple(resolved)


def _cell_to_frame_value(cell: Scalar) -> object:
    if cell is None:
        import numpy as np

        return np.nan
    if isinstance(cell, bool):
        return int(cell)
    return cell
