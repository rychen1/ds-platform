"""In-memory feature representations for modeling (not a feature store)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from ds_platform.hashing import SHA256_HEX_PATTERN

type Scalar = str | int | float | bool | None

_SHA256_HEX = re.compile(SHA256_HEX_PATTERN)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureTable(_FrozenModel):
    """Rectangular in-memory feature matrix keyed by entity id.

    Invariants (enforced at construction):

    - ``len(values) == len(entity_ids) == len(source_payload_ids)``
    - each row in ``values`` has ``len(columns)`` cells
    - ``entity_ids`` and ``columns`` contain no duplicates
    - every cell is a :data:`Scalar`
    """

    entity_ids: tuple[str, ...]
    columns: tuple[str, ...]
    values: tuple[tuple[Scalar, ...], ...]
    source_payload_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> FeatureTable:
        if len(self.values) != len(self.entity_ids):
            raise ValueError("values row count must match entity_ids length")
        if len(self.source_payload_ids) != len(self.entity_ids):
            raise ValueError("source_payload_ids length must match entity_ids length")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must be unique")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")
        width = len(self.columns)
        for row_index, row in enumerate(self.values):
            if len(row) != width:
                raise ValueError(
                    f"values row {row_index} width must match columns length"
                )
            for cell_index, cell in enumerate(row):
                if not _is_scalar(cell):
                    raise ValueError(
                        f"values[{row_index}][{cell_index}] must be a Scalar"
                    )
                if isinstance(cell, float) and not math.isfinite(cell):
                    raise ValueError(
                        f"values[{row_index}][{cell_index}] must be a finite float"
                    )
        return self


@runtime_checkable
class FeatureView(Protocol):
    """Named, deterministic row transform producing a :class:`FeatureTable`."""

    @property
    def name(self) -> str: ...

    def transform(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        as_of: date | None = None,
    ) -> FeatureTable: ...


def align_feature_tables(
    tables: Sequence[FeatureTable],
    *,
    how: Literal["inner"] = "inner",
    on_missing: Literal["inner", "error"] = "inner",
) -> FeatureTable:
    """Join tables on ``entity_ids`` with deterministic column naming.

    Entity order follows the first table, restricted to entities present in
    every input table (inner join). Entity ids are not re-sorted.

    Column names from the first table are kept unchanged. For each later
    table, a column keeps its name when it does not collide with an existing
    name; on collision it is prefixed ``v{index}__{column}`` where ``index``
    is the table's zero-based position in ``tables``.

    ``source_payload_ids`` for each aligned row come from the first table.
    """
    if how != "inner":
        raise ValueError(f"unsupported align how: {how!r}")
    if not tables:
        return FeatureTable(
            entity_ids=(),
            columns=(),
            values=(),
            source_payload_ids=(),
        )
    if len(tables) == 1:
        return tables[0]

    first = tables[0]
    common_ids = set(first.entity_ids)
    for table in tables[1:]:
        common_ids &= set(table.entity_ids)
    if on_missing == "error":
        for table in tables:
            missing = set(table.entity_ids) - common_ids
            if missing:
                raise ValueError(f"align is missing entities: {sorted(missing)}")
    elif on_missing != "inner":
        raise ValueError(f"unsupported align on_missing: {on_missing!r}")
    aligned_ids = tuple(
        entity_id for entity_id in first.entity_ids if entity_id in common_ids
    )
    if not aligned_ids:
        return FeatureTable(
            entity_ids=(),
            columns=(),
            values=(),
            source_payload_ids=(),
        )

    merged_columns: list[str] = []
    for table_index, table in enumerate(tables):
        for column in table.columns:
            output_name = column
            if output_name in merged_columns:
                output_name = f"v{table_index}__{column}"
            if output_name in merged_columns:
                raise ValueError(
                    f"column name collision after prefixing: {output_name!r}"
                )
            merged_columns.append(output_name)

    row_maps = [
        {
            entity_id: row
            for entity_id, row in zip(table.entity_ids, table.values, strict=True)
        }
        for table in tables
    ]
    first_sources = dict(zip(first.entity_ids, first.source_payload_ids, strict=True))

    merged_values: list[tuple[Scalar, ...]] = []
    merged_sources: list[str] = []
    for entity_id in aligned_ids:
        row: list[Scalar] = []
        for table_index, _table in enumerate(tables):
            source_row = row_maps[table_index][entity_id]
            row.extend(source_row)
        merged_values.append(tuple(row))
        merged_sources.append(first_sources[entity_id])

    return FeatureTable(
        entity_ids=aligned_ids,
        columns=tuple(merged_columns),
        values=tuple(merged_values),
        source_payload_ids=tuple(merged_sources),
    )


def select_columns(table: FeatureTable, columns: Sequence[str]) -> FeatureTable:
    """Return ``table`` restricted to ``columns`` in the requested order.

    Raises ``KeyError`` when a requested column is absent.
    """
    if not columns:
        return FeatureTable(
            entity_ids=table.entity_ids,
            columns=(),
            values=tuple(() for _ in table.values),
            source_payload_ids=table.source_payload_ids,
        )
    index_by_name = {name: index for index, name in enumerate(table.columns)}
    selected_indexes: list[int] = []
    for column in columns:
        try:
            selected_indexes.append(index_by_name[column])
        except KeyError as exc:
            raise KeyError(f"column not found: {column!r}") from exc
    selected_values = tuple(
        tuple(row[index] for index in selected_indexes) for row in table.values
    )
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=tuple(columns),
        values=selected_values,
        source_payload_ids=table.source_payload_ids,
    )


def extract_column(table: FeatureTable, column: str) -> tuple[Scalar, ...]:
    """Return one column aligned to ``table.entity_ids``.

    Raises ``KeyError`` when ``column`` is absent.
    """
    try:
        column_index = table.columns.index(column)
    except ValueError as exc:
        raise KeyError(f"column not found: {column!r}") from exc
    return tuple(row[column_index] for row in table.values)


def require_source_payload_ids(source_payload_ids: Sequence[str]) -> None:
    """Reject strings that are not lowercase SHA-256 hex payload ids.

    In-memory tables do not call this. Envelope ``inputs`` are already
    hash-checked. Call this when a persist path needs the same rule on
    table-local source ids.
    """
    for source_id in source_payload_ids:
        if _SHA256_HEX.fullmatch(source_id) is None:
            raise ValueError(
                f"source_payload_ids must be lowercase sha256 hex: {source_id!r}"
            )


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None
