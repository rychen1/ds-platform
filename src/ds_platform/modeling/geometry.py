"""Collection-level distances, neighbors, and novelty scores."""

from __future__ import annotations

from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.representations import (
    DistanceMetric,
    RepresentationTable,
    vector_distance,
)


def pairwise_distances(
    table: RepresentationTable,
    *,
    metric: DistanceMetric,
) -> tuple[tuple[float, ...], ...]:
    """Return a dense distance matrix in ``table.entity_ids`` order."""
    return tuple(
        tuple(vector_distance(left, right, metric=metric) for right in table.vectors)
        for left in table.vectors
    )


def knn(
    table: RepresentationTable,
    *,
    k: int,
    metric: DistanceMetric,
) -> FeatureTable:
    """Return ``k`` nearest neighbors for each entity, excluding self.

    Columns are ``neighbor_1..k`` and ``distance_1..k``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    n_entities = len(table.entity_ids)
    if n_entities <= 1:
        raise ValueError("knn requires at least two entities")
    if k >= n_entities:
        raise ValueError("k must be smaller than the number of entities")

    columns = tuple(
        [f"neighbor_{index}" for index in range(1, k + 1)]
        + [f"distance_{index}" for index in range(1, k + 1)]
    )
    values: list[tuple[Scalar, ...]] = []
    distances = pairwise_distances(table, metric=metric)
    for row_index, row_distances in enumerate(distances):
        ranked = sorted(
            (
                (distance, neighbor_id)
                for neighbor_index, (distance, neighbor_id) in enumerate(
                    zip(row_distances, table.entity_ids, strict=True)
                )
                if neighbor_index != row_index
            ),
            key=lambda item: (item[0], item[1]),
        )
        neighbors = ranked[:k]
        neighbor_ids = tuple(neighbor_id for _distance, neighbor_id in neighbors)
        neighbor_distances = tuple(distance for distance, _neighbor_id in neighbors)
        values.append(neighbor_ids + neighbor_distances)
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=columns,
        values=tuple(values),
        source_payload_ids=table.source_payload_ids,
    )


def novelty_scores(
    table: RepresentationTable,
    *,
    k: int,
    metric: DistanceMetric,
) -> FeatureTable:
    """Return mean kNN distance per entity in column ``novelty``."""
    neighbors = knn(table, k=k, metric=metric)
    distance_indexes = [
        neighbors.columns.index(f"distance_{index}") for index in range(1, k + 1)
    ]
    values = tuple((_mean_distance(row, distance_indexes),) for row in neighbors.values)
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=("novelty",),
        values=values,
        source_payload_ids=table.source_payload_ids,
    )


def _as_float(value: Scalar) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {value!r}")
    return float(value)


def _mean_distance(row: tuple[Scalar, ...], indexes: list[int]) -> float:
    return sum(_as_float(row[index]) for index in indexes) / len(indexes)
