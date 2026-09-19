"""Entity-level train/validation/test splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.sequences import SequenceTable
from ds_platform.modeling.spec import SplitSpec


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitAssignment(_FrozenModel):
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def split_entities(
    entity_ids: Sequence[str],
    spec: SplitSpec,
    *,
    labels: Sequence[object] | None = None,
    timestamps: Sequence[date] | None = None,
) -> tuple[SplitAssignment, ...]:
    """Split entity ids for holdout or k-fold cross-validation.

    When ``spec.as_of`` is set, ``timestamps`` must be provided with one
    entry per entity. Entities with ``timestamp > spec.as_of`` are dropped
    before shuffling. Missing timestamps raise ``ValueError``.

    Holdout returns one assignment with empty ``validation_ids``. K-fold
    returns ``spec.n_splits`` assignments, each with empty ``validation_ids``.
    """
    ids, split_labels, split_times = _prepare_entities(
        entity_ids,
        labels=labels,
        timestamps=timestamps,
        as_of=spec.as_of,
    )
    if spec.stratify and split_labels is None:
        raise ValueError("labels are required when stratify is enabled")
    if spec.method == "temporal" and spec.stratify:
        raise ValueError("temporal split does not support stratify")

    if spec.method == "holdout":
        assignment = _holdout_assignment(ids, split_labels, spec)
        return (assignment,)
    if spec.method == "kfold":
        return _kfold_assignments(ids, split_labels, spec)
    if spec.method == "temporal":
        if split_times is None:
            raise ValueError("timestamps are required for temporal split")
        return (_temporal_assignment(ids, split_times, spec),)
    raise ValueError(f"unsupported split method: {spec.method!r}")


def split_groups(
    group_ids: Sequence[str],
    spec: SplitSpec,
    *,
    labels: Sequence[object] | None = None,
    timestamps: Sequence[date] | None = None,
) -> tuple[SplitAssignment, ...]:
    """Split unique group ids with canonical ordering.

    Duplicate group ids are dropped after the first occurrence. When
    ``labels`` or ``timestamps`` are provided they align to the input
    ``group_ids`` sequence; values from the first occurrence of each
    group are kept. Unique groups are sorted lexicographically before
    holdout or k-fold selection so incidental input order does not change
    train/test assignment for the same seed. Returned assignment ids are
    group ids.
    """
    unique_ids: list[str] = []
    unique_labels: list[object] | None = [] if labels is not None else None
    unique_timestamps: list[date] | None = [] if timestamps is not None else None
    seen: set[str] = set()
    if labels is not None and len(labels) != len(group_ids):
        raise ValueError("labels length must match group_ids length")
    if timestamps is not None and len(timestamps) != len(group_ids):
        raise ValueError("timestamps length must match group_ids length")
    for index, group_id in enumerate(group_ids):
        if group_id in seen:
            continue
        seen.add(group_id)
        unique_ids.append(group_id)
        if unique_labels is not None and labels is not None:
            unique_labels.append(labels[index])
        if unique_timestamps is not None and timestamps is not None:
            unique_timestamps.append(timestamps[index])
    order = sorted(range(len(unique_ids)), key=lambda index: unique_ids[index])
    unique_ids = [unique_ids[index] for index in order]
    if unique_labels is not None:
        unique_labels = [unique_labels[index] for index in order]
    if unique_timestamps is not None:
        unique_timestamps = [unique_timestamps[index] for index in order]
    return split_entities(
        unique_ids,
        spec,
        labels=unique_labels,
        timestamps=unique_timestamps,
    )


def apply_group_split(
    table: SequenceTable,
    assignment: SplitAssignment,
) -> tuple[SequenceTable, SequenceTable, SequenceTable]:
    """Return train, validation, and test sequence tables by group id.

    Row order follows ``table``. Events of one group stay together.
    """
    train = _slice_sequence_table(table, set(assignment.train_ids))
    validation = _slice_sequence_table(table, set(assignment.validation_ids))
    test = _slice_sequence_table(table, set(assignment.test_ids))
    return train, validation, test


def apply_split(
    table: FeatureTable,
    assignment: SplitAssignment,
) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
    """Return train, validation, and test tables sliced from ``table``.

    Row order follows ``table.entity_ids``. ``source_payload_ids`` and
    ``columns`` are preserved in each slice.
    """
    train = _slice_table(table, set(assignment.train_ids))
    validation = _slice_table(table, set(assignment.validation_ids))
    test = _slice_table(table, set(assignment.test_ids))
    return train, validation, test


def _prepare_entities(
    entity_ids: Sequence[str],
    *,
    labels: Sequence[object] | None,
    timestamps: Sequence[date] | None,
    as_of: date | None,
) -> tuple[list[str], list[object] | None, list[date] | None]:
    ids = list(entity_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("entity_ids must be unique")
    if labels is not None and len(labels) != len(ids):
        raise ValueError("labels length must match entity_ids length")

    if as_of is None:
        kept_times: list[date] | None
        kept_times = list(timestamps) if timestamps is not None else None
        if kept_times is not None and len(kept_times) != len(ids):
            raise ValueError("timestamps length must match entity_ids length")
        if labels is None:
            return ids, None, kept_times
        return ids, list(labels), kept_times

    if timestamps is None:
        raise ValueError("timestamps are required when split.as_of is set")
    if len(timestamps) != len(ids):
        raise ValueError("timestamps length must match entity_ids length")

    kept_ids: list[str] = []
    kept_labels: list[object] | None = [] if labels is not None else None
    filtered_times: list[date] = []
    for index, entity_id in enumerate(ids):
        timestamp = timestamps[index]
        if timestamp is None:
            raise ValueError("timestamp required for each entity when as_of is set")
        if timestamp <= as_of:
            kept_ids.append(entity_id)
            filtered_times.append(timestamp)
            if kept_labels is not None and labels is not None:
                kept_labels.append(labels[index])
    return kept_ids, kept_labels, filtered_times


def _holdout_assignment(
    entity_ids: list[str],
    labels: list[object] | None,
    spec: SplitSpec,
) -> SplitAssignment:
    if spec.test_size is None:
        raise ValueError("test_size is required for holdout split")
    test_size = spec.test_size
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not entity_ids:
        return SplitAssignment(train_ids=(), validation_ids=(), test_ids=())

    if spec.stratify:
        test_ids = _stratified_holdout_test_ids(entity_ids, labels, spec, test_size)
    else:
        test_ids = _random_holdout_test_ids(entity_ids, spec, test_size)

    test_set = set(test_ids)
    train_ids = tuple(
        entity_id for entity_id in entity_ids if entity_id not in test_set
    )
    return SplitAssignment(
        train_ids=train_ids,
        validation_ids=(),
        test_ids=test_ids,
    )


def _random_holdout_test_ids(
    entity_ids: list[str],
    spec: SplitSpec,
    test_size: float,
) -> tuple[str, ...]:
    shuffled = list(entity_ids)
    random.Random(spec.seed).shuffle(shuffled)
    n_test = _holdout_test_count(len(shuffled), test_size)
    return tuple(shuffled[:n_test])


def _stratified_holdout_test_ids(
    entity_ids: list[str],
    labels: list[object] | None,
    spec: SplitSpec,
    test_size: float,
) -> tuple[str, ...]:
    assert labels is not None
    by_label: dict[object, list[str]] = defaultdict(list)
    for entity_id, label in zip(entity_ids, labels, strict=True):
        by_label[label].append(entity_id)

    test_ids: list[str] = []
    for label_index, label in enumerate(sorted(by_label, key=repr)):
        shuffled = list(by_label[label])
        random.Random(spec.seed + label_index).shuffle(shuffled)
        n_test = _holdout_test_count(len(shuffled), test_size)
        test_ids.extend(shuffled[:n_test])
    return tuple(test_ids)


def _temporal_assignment(
    entity_ids: list[str],
    timestamps: list[date],
    spec: SplitSpec,
) -> SplitAssignment:
    if spec.test_size is None:
        raise ValueError("test_size is required for temporal split")
    if not 0 < spec.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    ordered = sorted(
        zip(timestamps, entity_ids, strict=True),
        key=lambda item: (item[0], item[1]),
    )
    ids = [entity_id for _timestamp, entity_id in ordered]
    if not ids:
        return SplitAssignment(train_ids=(), validation_ids=(), test_ids=())
    n_test = _holdout_test_count(len(ids), spec.test_size)
    return SplitAssignment(
        train_ids=tuple(ids[:-n_test]),
        validation_ids=(),
        test_ids=tuple(ids[-n_test:]),
    )


def _holdout_test_count(n_entities: int, test_size: float) -> int:
    if n_entities <= 1:
        return n_entities
    n_test = int(round(n_entities * test_size))
    return max(1, min(n_entities - 1, n_test))


def _kfold_assignments(
    entity_ids: list[str],
    labels: list[object] | None,
    spec: SplitSpec,
) -> tuple[SplitAssignment, ...]:
    if spec.n_splits is None:
        raise ValueError("n_splits is required for kfold split")
    if spec.n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(entity_ids) < spec.n_splits:
        raise ValueError("entity count must be at least n_splits")

    if spec.stratify:
        folds = _stratified_folds(entity_ids, labels, spec.n_splits, spec.seed)
    else:
        folds = _random_folds(entity_ids, spec.n_splits, spec.seed)

    assignments: list[SplitAssignment] = []
    for fold_index in range(spec.n_splits):
        test_ids = tuple(folds[fold_index])
        train_ids = tuple(
            entity_id
            for index, fold in enumerate(folds)
            if index != fold_index
            for entity_id in fold
        )
        assignments.append(
            SplitAssignment(
                train_ids=train_ids,
                validation_ids=(),
                test_ids=test_ids,
            )
        )
    return tuple(assignments)


def _random_folds(
    entity_ids: list[str],
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    shuffled = list(entity_ids)
    random.Random(seed).shuffle(shuffled)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for index, entity_id in enumerate(shuffled):
        folds[index % n_splits].append(entity_id)
    return folds


def _stratified_folds(
    entity_ids: list[str],
    labels: list[object] | None,
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    assert labels is not None
    by_label: dict[object, list[str]] = defaultdict(list)
    for entity_id, label in zip(entity_ids, labels, strict=True):
        by_label[label].append(entity_id)

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for label_index, label in enumerate(sorted(by_label, key=repr)):
        shuffled = list(by_label[label])
        random.Random(seed + label_index).shuffle(shuffled)
        for index, entity_id in enumerate(shuffled):
            folds[index % n_splits].append(entity_id)
    return folds


def _slice_sequence_table(
    table: SequenceTable,
    group_id_set: set[str],
) -> SequenceTable:
    if not group_id_set:
        return SequenceTable(
            group_ids=(),
            event_ids=(),
            positions=(),
            actor_ids=(),
            source_payload_ids=(),
        )
    indexes = [
        index
        for index, group_id in enumerate(table.group_ids)
        if group_id in group_id_set
    ]
    return SequenceTable(
        group_ids=tuple(table.group_ids[index] for index in indexes),
        event_ids=tuple(table.event_ids[index] for index in indexes),
        positions=tuple(table.positions[index] for index in indexes),
        actor_ids=tuple(table.actor_ids[index] for index in indexes),
        source_payload_ids=tuple(table.source_payload_ids[index] for index in indexes),
    )


def _slice_table(table: FeatureTable, entity_id_set: set[str]) -> FeatureTable:
    if not entity_id_set:
        return FeatureTable(
            entity_ids=(),
            columns=table.columns,
            values=(),
            source_payload_ids=(),
        )
    selected_indexes = [
        index
        for index, entity_id in enumerate(table.entity_ids)
        if entity_id in entity_id_set
    ]
    return FeatureTable(
        entity_ids=tuple(table.entity_ids[index] for index in selected_indexes),
        columns=table.columns,
        values=tuple(table.values[index] for index in selected_indexes),
        source_payload_ids=tuple(
            table.source_payload_ids[index] for index in selected_indexes
        ),
    )
