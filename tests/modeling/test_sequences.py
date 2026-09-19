"""SequenceTable, grouped split, and SequenceEncoder helpers."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.sequences import (
    SequenceTable,
    attach_representations,
    events_for_group,
    order_events,
    pool_groups,
)
from ds_platform.modeling.spec import SequenceSpec, SplitSpec, spec_config_hash
from ds_platform.modeling.split import apply_group_split, split_groups

_FORBIDDEN = {
    "Trajectory",
    "board_game_analysis",
    "numpy",
    "restaurant_intelligence",
}

_SOURCE = "e" * 64


def _sequences() -> SequenceTable:
    return SequenceTable(
        group_ids=("g1", "g2", "g1", "g2"),
        event_ids=("t2", "u1", "t1", "u2"),
        positions=(1, 0, 0, 1),
        actor_ids=("pA", "pB", "pA", None),
        source_payload_ids=(_SOURCE,) * 4,
    )


def _events() -> RepresentationTable:
    return RepresentationTable(
        entity_ids=("t1", "t2", "u1", "u2"),
        vectors=((1.0, 0.0), (3.0, 0.0), (5.0, 0.0), (7.0, 0.0)),
        dim=2,
        source_payload_ids=(_SOURCE,) * 4,
    )


class _PrefixMeanEncoder:
    def fit(self, sequences: SequenceTable, events: RepresentationTable) -> None:
        del sequences, events

    def encode(
        self,
        sequences: SequenceTable,
        events: RepresentationTable,
    ) -> RepresentationTable:
        ordered = order_events(sequences)
        aligned = attach_representations(ordered, events)
        vectors: list[tuple[float, ...]] = []
        running: dict[str, list[tuple[float, ...]]] = {}
        for group_id, vector in zip(ordered.group_ids, aligned.vectors, strict=True):
            running.setdefault(group_id, []).append(vector)
            prefix = running[group_id]
            pooled = tuple(
                sum(item[dim] for item in prefix) / len(prefix)
                for dim in range(aligned.dim)
            )
            vectors.append(pooled)
        return RepresentationTable(
            entity_ids=ordered.event_ids,
            vectors=tuple(vectors),
            dim=aligned.dim,
            source_payload_ids=aligned.source_payload_ids,
        )


def test_sequence_table_rejects_duplicate_group_position() -> None:
    with pytest.raises(ValidationError, match="position pairs must be unique"):
        SequenceTable(
            group_ids=("g1", "g1"),
            event_ids=("t1", "t2"),
            positions=(0, 0),
            actor_ids=(None, None),
            source_payload_ids=(_SOURCE, _SOURCE),
        )


def test_order_events_and_events_for_group() -> None:
    ordered = order_events(_sequences())
    assert ordered.event_ids == ("t1", "t2", "u1", "u2")
    assert events_for_group(ordered, "g1").event_ids == ("t1", "t2")


def test_attach_and_pool_groups() -> None:
    sequences = order_events(_sequences())
    attached = attach_representations(sequences, _events())
    assert attached.entity_ids == ("t1", "t2", "u1", "u2")
    pooled = pool_groups(attached, sequences)
    assert pooled.entity_ids == ("g1", "g2")
    assert pooled.vectors[0] == (2.0, 0.0)
    assert pooled.vectors[1] == (6.0, 0.0)


def test_split_groups_keeps_events_together() -> None:
    sequences = _sequences()
    assignment = split_groups(
        sequences.group_ids,
        SplitSpec(method="holdout", seed=3, test_size=0.5),
    )[0]
    train, _validation, test = apply_group_split(sequences, assignment)
    train_groups = set(train.group_ids)
    test_groups = set(test.group_ids)
    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == {"g1", "g2"}
    for group_id in train_groups:
        assert set(events_for_group(train, group_id).event_ids) == set(
            events_for_group(sequences, group_id).event_ids
        )


def test_prefix_mean_encoder_is_deterministic() -> None:
    sequences = _sequences()
    events = _events()
    encoder = _PrefixMeanEncoder()
    first = encoder.encode(sequences, events)
    second = encoder.encode(sequences, events)
    assert first.vectors == second.vectors
    assert first.entity_ids == order_events(sequences).event_ids


def test_sequence_spec_hash() -> None:
    spec = SequenceSpec(family="prefix_mean", seed=2, max_events=8)
    assert spec_config_hash(spec) == spec_config_hash(
        SequenceSpec(family="prefix_mean", seed=2, max_events=8)
    )


def test_sequences_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.sequences as sequences

    assert _FORBIDDEN.intersection(sys.modules) == set()
    assert "Trajectory" not in dir(sequences)
