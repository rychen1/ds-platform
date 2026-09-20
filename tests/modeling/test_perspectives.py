"""PerspectiveTable, distances, and composition (foundation Phases 3–4)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from ds_platform.modeling.perspectives import (
    PerspectivePath,
    PerspectiveTable,
    compose_perspectives,
    join_pair_key,
    pairwise_perspective_distances,
    perspectives_for_subject,
    split_pair_key,
    subjects_for_perspective,
    to_representation_table,
)
from ds_platform.modeling.spec import PerspectiveSpec, spec_config_hash
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "restaurant_intelligence",
}

_SOURCE = "d" * 64


def _table() -> PerspectiveTable:
    return PerspectiveTable(
        subject_ids=("s1", "s1", "s2"),
        perspective_ids=("pA", "pB", "pA"),
        vectors=((1.0, 0.0), (0.0, 1.0), (2.0, 0.0)),
        dim=2,
        source_payload_ids=(_SOURCE, _SOURCE, _SOURCE),
    )


class _MeanComposer:
    def compose(self, chain: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
        dim = len(chain[0])
        return tuple(
            sum(vector[index] for vector in chain) / len(chain) for index in range(dim)
        )


def test_perspective_table_rejects_duplicate_pairs() -> None:
    with pytest.raises(ValidationError, match="pairs must be unique"):
        PerspectiveTable(
            subject_ids=("s1", "s1"),
            perspective_ids=("pA", "pA"),
            vectors=((1.0,), (2.0,)),
            dim=1,
            source_payload_ids=(_SOURCE, _SOURCE),
        )


def test_perspectives_for_subject_and_subjects_for_perspective() -> None:
    table = _table()
    of_subject = perspectives_for_subject(table, "s1")
    assert of_subject.entity_ids == ("pA", "pB")
    assert of_subject.vectors == ((1.0, 0.0), (0.0, 1.0))
    of_perspective = subjects_for_perspective(table, "pA")
    assert of_perspective.entity_ids == ("s1", "s2")


def test_pairwise_perspective_distances_only_multi_view_subjects() -> None:
    distances = pairwise_perspective_distances(_table(), metric="l2")
    assert distances.entity_ids == ("s1",)
    assert distances.columns == ("pA__pB",)
    assert distances.values[0][0] == pytest.approx(2.0**0.5)


def test_join_and_split_pair_key_round_trip() -> None:
    key = join_pair_key("s1", "pA")
    assert split_pair_key(key) == ("s1", "pA")


def test_to_representation_table_pair_key() -> None:
    exported = to_representation_table(_table(), entity_key="pair")
    assert split_pair_key(exported.entity_ids[0]) == ("s1", "pA")
    with pytest.raises(ValueError, match="not unique"):
        to_representation_table(_table(), entity_key="perspective")


def test_compose_path_length_one_matches_subject_select() -> None:
    table = _table()
    selected = perspectives_for_subject(table, "s1")
    composed = compose_perspectives(
        table,
        (PerspectivePath(steps=("pA",)), PerspectivePath(steps=("pB",))),
        composer=_MeanComposer(),
        subject_ids=("s1",),
    )
    by_path = {
        split_pair_key(entity_id)[1]: vector
        for entity_id, vector in zip(composed.entity_ids, composed.vectors, strict=True)
    }
    assert by_path["pA"] == selected.vectors[0]
    assert by_path["pB"] == selected.vectors[1]


def test_compose_missing_step_raises() -> None:
    with pytest.raises(KeyError, match="pZ"):
        compose_perspectives(
            _table(),
            (PerspectivePath(steps=("pA", "pZ")),),
            composer=_MeanComposer(),
            subject_ids=("s1",),
        )


def test_compose_mean_is_deterministic() -> None:
    table = _table()
    paths = (PerspectivePath(steps=("pA", "pB")),)
    first = compose_perspectives(
        table,
        paths,
        composer=_MeanComposer(),
        subject_ids=("s1",),
    )
    second = compose_perspectives(
        table,
        paths,
        composer=_MeanComposer(),
        subject_ids=("s1",),
    )
    assert first.vectors == ((0.5, 0.5),)
    assert first.vectors == second.vectors


def test_perspective_spec_hash() -> None:
    spec = PerspectiveSpec(subject_field="subject", perspective_field="observer")
    assert spec_config_hash(spec) == spec_config_hash(
        PerspectiveSpec(subject_field="subject", perspective_field="observer")
    )


def test_perspectives_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.perspectives", _FORBIDDEN)
    assert not hasattr(
        __import__("ds_platform.modeling.perspectives", fromlist=["GameState"]),
        "GameState",
    )
