"""Subject-by-perspective representation tables."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ds_platform.modeling.capabilities import PerspectiveComposer
from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.representations import (
    RepresentationTable,
    vector_distance,
)

_PAIR_KEY_SEP = "\x1f"
_PATH_KEY_SEP = "\x1e"

type PerspectiveEntityKey = Literal["subject", "perspective", "pair"]
type DistanceMetric = Literal["cosine", "l2"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PerspectiveTable(_FrozenModel):
    """Many representations of one subject, indexed by perspective id.

    Invariants (enforced at construction):

    - row counts align across all parallel tuples
    - every vector has length ``dim``
    - ``(subject_id, perspective_id)`` pairs are unique
    """

    subject_ids: tuple[str, ...]
    perspective_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    dim: int
    source_payload_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> PerspectiveTable:
        if self.dim < 0:
            raise ValueError("dim must be non-negative")
        n_rows = len(self.subject_ids)
        if len(self.perspective_ids) != n_rows:
            raise ValueError("perspective_ids length must match subject_ids length")
        if len(self.vectors) != n_rows:
            raise ValueError("vectors row count must match subject_ids length")
        if len(self.source_payload_ids) != n_rows:
            raise ValueError("source_payload_ids length must match subject_ids length")
        seen: set[tuple[str, str]] = set()
        for subject_id, perspective_id in zip(
            self.subject_ids,
            self.perspective_ids,
            strict=True,
        ):
            key = (subject_id, perspective_id)
            if key in seen:
                raise ValueError("subject_id and perspective_id pairs must be unique")
            seen.add(key)
        for row_index, vector in enumerate(self.vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"vectors row {row_index} length must match dim {self.dim}"
                )
            if any(not math.isfinite(component) for component in vector):
                raise ValueError(f"vectors row {row_index} must contain finite floats")
        return self


class PerspectivePath(_FrozenModel):
    """Ordered chain of opaque perspective ids for same-subject composition."""

    steps: tuple[str, ...] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def _steps_non_empty_strings(cls, steps: tuple[str, ...]) -> tuple[str, ...]:
        if any(not step for step in steps):
            raise ValueError("steps must be non-empty strings")
        return steps


def join_pair_key(subject_id: str, perspective_id: str) -> str:
    """Return the opaque export key for one ``(subject, perspective)`` pair."""
    if _PAIR_KEY_SEP in subject_id or _PAIR_KEY_SEP in perspective_id:
        raise ValueError(
            "subject_id and perspective_id must not contain the pair-key separator"
        )
    return f"{subject_id}{_PAIR_KEY_SEP}{perspective_id}"


def split_pair_key(key: str) -> tuple[str, str]:
    """Split a key produced by :func:`join_pair_key`."""
    if key.count(_PAIR_KEY_SEP) != 1:
        raise ValueError("pair key must contain exactly one separator")
    subject_id, perspective_id = key.split(_PAIR_KEY_SEP, 1)
    return subject_id, perspective_id


def perspectives_for_subject(
    table: PerspectiveTable,
    subject_id: str,
) -> RepresentationTable:
    """Return perspectives of ``subject_id`` keyed by perspective id."""
    indexes = [
        index
        for index, row_subject in enumerate(table.subject_ids)
        if row_subject == subject_id
    ]
    if not indexes:
        raise KeyError(f"subject not found: {subject_id!r}")
    return RepresentationTable(
        entity_ids=tuple(table.perspective_ids[index] for index in indexes),
        vectors=tuple(table.vectors[index] for index in indexes),
        dim=table.dim,
        source_payload_ids=tuple(table.source_payload_ids[index] for index in indexes),
    )


def subjects_for_perspective(
    table: PerspectiveTable,
    perspective_id: str,
) -> RepresentationTable:
    """Return subjects that have ``perspective_id`` keyed by subject id."""
    indexes = [
        index
        for index, row_perspective in enumerate(table.perspective_ids)
        if row_perspective == perspective_id
    ]
    if not indexes:
        raise KeyError(f"perspective not found: {perspective_id!r}")
    return RepresentationTable(
        entity_ids=tuple(table.subject_ids[index] for index in indexes),
        vectors=tuple(table.vectors[index] for index in indexes),
        dim=table.dim,
        source_payload_ids=tuple(table.source_payload_ids[index] for index in indexes),
    )


def pairwise_perspective_distances(
    table: PerspectiveTable,
    *,
    metric: DistanceMetric,
) -> FeatureTable:
    """Return same-subject pairwise distances as a :class:`FeatureTable`.

    Rows are subjects that have at least two perspectives, in first-seen
    order. Columns are lexicographic pair names ``pA__pB``. Missing pairs
    on a subject are ``None``.
    """
    by_subject = _rows_by_subject(table)
    subjects = tuple(
        subject_id
        for subject_id in _unique_in_order(table.subject_ids)
        if len(by_subject[subject_id]) >= 2
    )
    pair_keys = sorted(
        {
            tuple(sorted((left, right)))
            for rows in by_subject.values()
            if len(rows) >= 2
            for left_index, (left, _left_vector, _left_source) in enumerate(rows)
            for right, _right_vector, _right_source in rows[left_index + 1 :]
        }
    )
    pair_names = tuple(_pair_column(left, right) for left, right in pair_keys)
    values: list[tuple[Scalar, ...]] = []
    sources: list[str] = []
    for subject_id in subjects:
        lookup = {
            perspective_id: vector
            for perspective_id, vector, _source in by_subject[subject_id]
        }
        row: list[Scalar] = []
        for left_id, right_id in pair_keys:
            if left_id in lookup and right_id in lookup:
                row.append(
                    vector_distance(lookup[left_id], lookup[right_id], metric=metric)
                )
            else:
                row.append(None)
        values.append(tuple(row))
        sources.append(by_subject[subject_id][0][2])
    return FeatureTable(
        entity_ids=subjects,
        columns=pair_names,
        values=tuple(values),
        source_payload_ids=tuple(sources),
    )


def to_representation_table(
    table: PerspectiveTable,
    *,
    entity_key: PerspectiveEntityKey,
) -> RepresentationTable:
    """Export a perspective table as a :class:`RepresentationTable`.

    ``entity_key="pair"`` uses :func:`join_pair_key`. ``subject`` and
    ``perspective`` require the chosen key to be unique.
    """
    if entity_key == "pair":
        entity_ids = tuple(
            join_pair_key(subject_id, perspective_id)
            for subject_id, perspective_id in zip(
                table.subject_ids,
                table.perspective_ids,
                strict=True,
            )
        )
    elif entity_key == "subject":
        entity_ids = table.subject_ids
    elif entity_key == "perspective":
        entity_ids = table.perspective_ids
    else:
        raise ValueError(f"unsupported entity_key: {entity_key!r}")
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError(f"entity_key {entity_key!r} is not unique in this table")
    return RepresentationTable(
        entity_ids=entity_ids,
        vectors=table.vectors,
        dim=table.dim,
        source_payload_ids=table.source_payload_ids,
    )


def compose_perspectives(
    table: PerspectiveTable,
    paths: Sequence[PerspectivePath],
    *,
    composer: PerspectiveComposer,
    subject_ids: Sequence[str] | None = None,
) -> RepresentationTable:
    """Compose same-subject perspective chains.

    Each output row is keyed by ``join_pair_key(subject_id, path)`` where
    path steps are joined with a unit-separator. Missing ``(subject, step)``
    links raise ``KeyError``. Cross-subject nesting is a consumer encoding
    problem: this helper only composes vectors already present for one
    subject.
    """
    if not paths:
        raise ValueError("paths must contain at least one entry")
    lookup = {
        (subject_id, perspective_id): (vector, source)
        for subject_id, perspective_id, vector, source in zip(
            table.subject_ids,
            table.perspective_ids,
            table.vectors,
            table.source_payload_ids,
            strict=True,
        )
    }
    chosen_subjects = (
        tuple(subject_ids)
        if subject_ids is not None
        else _unique_in_order(table.subject_ids)
    )
    entity_ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    output_dim: int | None = None
    for subject_id in chosen_subjects:
        for path in paths:
            chain: list[tuple[float, ...]] = []
            path_source: str | None = None
            for step in path.steps:
                try:
                    vector, source = lookup[(subject_id, step)]
                except KeyError as exc:
                    raise KeyError(
                        f"missing perspective {step!r} for subject {subject_id!r}"
                    ) from exc
                chain.append(vector)
                if path_source is None:
                    path_source = source
            composed = tuple(composer.compose(chain))
            if output_dim is None:
                output_dim = len(composed)
            elif len(composed) != output_dim:
                raise ValueError("composer must return vectors of equal length")
            entity_ids.append(_path_entity_id(subject_id, path))
            vectors.append(composed)
            sources.append(path_source or "")
    return RepresentationTable(
        entity_ids=tuple(entity_ids),
        vectors=tuple(vectors),
        dim=0 if output_dim is None else output_dim,
        source_payload_ids=tuple(sources),
    )


def _path_entity_id(subject_id: str, path: PerspectivePath) -> str:
    path_key = _PATH_KEY_SEP.join(path.steps)
    return join_pair_key(subject_id, path_key)


def _pair_column(left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{first}__{second}"


def _unique_in_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _rows_by_subject(
    table: PerspectiveTable,
) -> dict[str, list[tuple[str, tuple[float, ...], str]]]:
    grouped: dict[str, list[tuple[str, tuple[float, ...], str]]] = {}
    for subject_id, perspective_id, vector, source in zip(
        table.subject_ids,
        table.perspective_ids,
        table.vectors,
        table.source_payload_ids,
        strict=True,
    ):
        grouped.setdefault(subject_id, []).append((perspective_id, vector, source))
    return grouped
