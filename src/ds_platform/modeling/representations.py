"""In-memory vector representations keyed by opaque entity ids."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ds_platform.modeling.features import FeatureTable

type DistanceMetric = Literal["cosine", "l2"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RepresentationTable(_FrozenModel):
    """Id-aligned matrix of dense float vectors.

    Invariants (enforced at construction):

    - ``len(vectors) == len(entity_ids) == len(source_payload_ids)``
    - every vector has length ``dim``
    - ``entity_ids`` contain no duplicates
    - ``dim`` is non-negative
    """

    entity_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    dim: int
    source_payload_ids: tuple[str, ...]
    encoding_hash: str | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> RepresentationTable:
        if self.dim < 0:
            raise ValueError("dim must be non-negative")
        if len(self.vectors) != len(self.entity_ids):
            raise ValueError("vectors row count must match entity_ids length")
        if len(self.source_payload_ids) != len(self.entity_ids):
            raise ValueError("source_payload_ids length must match entity_ids length")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must be unique")
        for row_index, vector in enumerate(self.vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"vectors row {row_index} length must match dim {self.dim}"
                )
            for component in vector:
                if not math.isfinite(component):
                    raise ValueError(
                        f"vectors row {row_index} must contain finite floats"
                    )
        return self


def align_representation_tables(
    tables: Sequence[RepresentationTable],
    *,
    on_missing: Literal["inner", "error"] = "inner",
) -> RepresentationTable:
    """Inner-join tables on ``entity_ids`` and concatenate vectors.

    Entity order follows the first table, restricted to entities present in
    every input table. ``source_payload_ids`` come from the first table.
    """
    if not tables:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=0,
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
    dim = sum(table.dim for table in tables)
    if not aligned_ids:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=dim,
            source_payload_ids=(),
        )

    row_maps = [
        {
            entity_id: vector
            for entity_id, vector in zip(table.entity_ids, table.vectors, strict=True)
        }
        for table in tables
    ]
    first_sources = dict(zip(first.entity_ids, first.source_payload_ids, strict=True))
    merged_vectors = tuple(
        tuple(
            component
            for table_index, _table in enumerate(tables)
            for component in row_maps[table_index][entity_id]
        )
        for entity_id in aligned_ids
    )
    return RepresentationTable(
        entity_ids=aligned_ids,
        vectors=merged_vectors,
        dim=dim,
        source_payload_ids=tuple(first_sources[entity_id] for entity_id in aligned_ids),
    )


def select_entities(
    table: RepresentationTable,
    entity_ids: Sequence[str],
) -> RepresentationTable:
    """Return ``table`` restricted to ``entity_ids`` in the requested order.

    Raises ``KeyError`` when a requested entity is absent.
    """
    index_by_id = {entity_id: index for index, entity_id in enumerate(table.entity_ids)}
    selected: list[int] = []
    for entity_id in entity_ids:
        try:
            selected.append(index_by_id[entity_id])
        except KeyError as exc:
            raise KeyError(f"entity not found: {entity_id!r}") from exc
    return RepresentationTable(
        entity_ids=tuple(entity_ids),
        vectors=tuple(table.vectors[index] for index in selected),
        dim=table.dim,
        source_payload_ids=tuple(table.source_payload_ids[index] for index in selected),
    )


def representation_mse(
    y_true: RepresentationTable,
    y_pred: RepresentationTable,
) -> float:
    """Return mean squared error over aligned entity vectors."""
    left, right = _align_pair(y_true, y_pred)
    if not left.entity_ids:
        return 0.0
    if left.dim == 0:
        return 0.0
    total = 0.0
    count = 0
    for true_vector, pred_vector in zip(left.vectors, right.vectors, strict=True):
        for true_value, pred_value in zip(true_vector, pred_vector, strict=True):
            delta = true_value - pred_value
            total += delta * delta
            count += 1
    return total / count


def mean_cosine_similarity(
    left: RepresentationTable,
    right: RepresentationTable,
) -> float:
    """Return mean cosine similarity over aligned entity vectors."""
    aligned_left, aligned_right = _align_pair(left, right)
    if not aligned_left.entity_ids:
        return 0.0
    scores = [
        _cosine_similarity(left_vector, right_vector)
        for left_vector, right_vector in zip(
            aligned_left.vectors,
            aligned_right.vectors,
            strict=True,
        )
    ]
    return sum(scores) / len(scores)


def as_feature_table(
    table: RepresentationTable,
    *,
    prefix: str = "z",
) -> FeatureTable:
    """Flatten vectors into a :class:`FeatureTable` for v1 scalar probes."""
    columns = tuple(f"{prefix}{index}" for index in range(table.dim))
    values = tuple(tuple(component for component in vector) for vector in table.vectors)
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=columns,
        values=values,
        source_payload_ids=table.source_payload_ids,
    )


def vector_distance(
    left: Sequence[float],
    right: Sequence[float],
    *,
    metric: DistanceMetric,
) -> float:
    """Return the distance between two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    if metric == "l2":
        squared = sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
        return math.sqrt(squared)
    if metric == "cosine":
        return 1.0 - _cosine_similarity(left, right)
    raise ValueError(f"unsupported distance metric: {metric!r}")


def _align_pair(
    left: RepresentationTable,
    right: RepresentationTable,
) -> tuple[RepresentationTable, RepresentationTable]:
    if left.dim != right.dim:
        raise ValueError("representation tables must have the same dim")
    common = set(left.entity_ids) & set(right.entity_ids)
    ordered = tuple(entity_id for entity_id in left.entity_ids if entity_id in common)
    return select_entities(left, ordered), select_entities(right, ordered)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = 0.0
    left_norm_sq = 0.0
    right_norm_sq = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm_sq += left_value * left_value
        right_norm_sq += right_value * right_value
    if left_norm_sq == 0.0 or right_norm_sq == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm_sq * right_norm_sq)
