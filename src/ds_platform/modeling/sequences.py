"""Ordered event tables grouped by an opaque sequence key."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from ds_platform.modeling.representations import RepresentationTable, select_entities


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SequenceTable(_FrozenModel):
    """Ordered events belonging to opaque groups.

    Invariants (enforced at construction):

    - row counts align across all parallel tuples
    - ``event_ids`` are unique
    - ``(group_id, position)`` pairs are unique
    - positions are non-negative
    """

    group_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    positions: tuple[int, ...]
    actor_ids: tuple[str | None, ...]
    source_payload_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> SequenceTable:
        n_rows = len(self.group_ids)
        if len(self.event_ids) != n_rows:
            raise ValueError("event_ids length must match group_ids length")
        if len(self.positions) != n_rows:
            raise ValueError("positions length must match group_ids length")
        if len(self.actor_ids) != n_rows:
            raise ValueError("actor_ids length must match group_ids length")
        if len(self.source_payload_ids) != n_rows:
            raise ValueError("source_payload_ids length must match group_ids length")
        if len(set(self.event_ids)) != n_rows:
            raise ValueError("event_ids must be unique")
        seen_positions: set[tuple[str, int]] = set()
        for group_id, position in zip(self.group_ids, self.positions, strict=True):
            if position < 0:
                raise ValueError("positions must be non-negative")
            key = (group_id, position)
            if key in seen_positions:
                raise ValueError("group_id and position pairs must be unique")
            seen_positions.add(key)
        return self


def events_for_group(table: SequenceTable, group_id: str) -> SequenceTable:
    """Return events belonging to ``group_id`` in table order."""
    indexes = [
        index
        for index, row_group in enumerate(table.group_ids)
        if row_group == group_id
    ]
    if not indexes:
        raise KeyError(f"group not found: {group_id!r}")
    return _take_indexes(table, indexes)


def order_events(table: SequenceTable) -> SequenceTable:
    """Return ``table`` sorted by first-seen group order, then position."""
    group_order = {
        group_id: order
        for order, group_id in enumerate(_unique_in_order(table.group_ids))
    }
    indexes = sorted(
        range(len(table.event_ids)),
        key=lambda index: (group_order[table.group_ids[index]], table.positions[index]),
    )
    return _take_indexes(table, indexes)


def attach_representations(
    sequences: SequenceTable,
    events: RepresentationTable,
) -> RepresentationTable:
    """Align event representations to ``sequences.event_ids``.

    Raises ``KeyError`` when an event id is absent from ``events``.
    """
    return select_entities(events, sequences.event_ids)


def pool_groups(
    event_reprs: RepresentationTable,
    sequences: SequenceTable,
) -> RepresentationTable:
    """Mean-pool event representations by group id.

    Group order follows first appearance in ``sequences``. Learned pooling
    belongs in a consumer :class:`~ds_platform.modeling.capabilities.SequenceEncoder`.
    """
    aligned = attach_representations(sequences, event_reprs)
    by_group: dict[str, list[int]] = {}
    for index, group_id in enumerate(sequences.group_ids):
        by_group.setdefault(group_id, []).append(index)
    group_ids = _unique_in_order(sequences.group_ids)
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    for group_id in group_ids:
        indexes = by_group[group_id]
        n_events = len(indexes)
        pooled = tuple(
            sum(aligned.vectors[index][dim] for index in indexes) / n_events
            for dim in range(aligned.dim)
        )
        vectors.append(pooled)
        sources.append(aligned.source_payload_ids[indexes[0]])
    return RepresentationTable(
        entity_ids=group_ids,
        vectors=tuple(vectors),
        dim=aligned.dim,
        source_payload_ids=tuple(sources),
    )


def _take_indexes(table: SequenceTable, indexes: list[int]) -> SequenceTable:
    return SequenceTable(
        group_ids=tuple(table.group_ids[index] for index in indexes),
        event_ids=tuple(table.event_ids[index] for index in indexes),
        positions=tuple(table.positions[index] for index in indexes),
        actor_ids=tuple(table.actor_ids[index] for index in indexes),
        source_payload_ids=tuple(table.source_payload_ids[index] for index in indexes),
    )


def _unique_in_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
