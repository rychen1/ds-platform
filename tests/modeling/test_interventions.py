"""Intervention queries and open-loop rollout (foundation Phase 6)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.interventions import (
    InterventionQuery,
    apply_intervention,
    representation_delta,
    rollout,
)
from ds_platform.modeling.records import (
    put_evaluation_artifact,
    put_feature_dataset,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import RepresentationTable, representation_mse
from ds_platform.modeling.sequences import SequenceTable
from import_boundary_util import assert_import_does_not_pull

_FORBIDDEN = {
    "GameTree",
    "board_game_analysis",
    "numpy",
    "restaurant_intelligence",
}

_SOURCE = "f" * 64
_CREATED = datetime(2026, 9, 19, 19, 0, tzinfo=UTC)


def _reprs(
    entity_ids: tuple[str, ...],
    vectors: tuple[tuple[float, ...], ...],
) -> RepresentationTable:
    return RepresentationTable(
        entity_ids=entity_ids,
        vectors=vectors,
        dim=len(vectors[0]),
        source_payload_ids=(_SOURCE,) * len(entity_ids),
    )


class _AddEncoder:
    def fit(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> None:
        del contexts, conditions, pair_ids

    def encode(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> RepresentationTable:
        vectors = tuple(
            tuple(left + right for left, right in zip(context, condition, strict=True))
            for context, condition in zip(
                contexts.vectors,
                conditions.vectors,
                strict=True,
            )
        )
        return RepresentationTable(
            entity_ids=tuple(pair_ids),
            vectors=vectors,
            dim=contexts.dim,
            source_payload_ids=contexts.source_payload_ids,
        )


def test_intervention_query_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="context_ids length"):
        InterventionQuery(
            query_ids=("q1",),
            context_ids=("s1", "s2"),
            factual_condition_ids=("a1",),
            alternative_condition_ids=("a2",),
        )


def test_apply_intervention_differs_when_conditions_differ() -> None:
    contexts = _reprs(("s1",), ((1.0, 2.0),))
    conditions = _reprs(("a1", "a2"), ((10.0, 0.0), (0.0, 5.0)))
    query = InterventionQuery(
        query_ids=("q1",),
        context_ids=("s1",),
        factual_condition_ids=("a1",),
        alternative_condition_ids=("a2",),
    )
    factual, alternative = apply_intervention(
        _AddEncoder(),
        contexts,
        conditions,
        query,
    )
    assert factual.entity_ids == ("q1",)
    assert factual.vectors == ((11.0, 2.0),)
    assert alternative.vectors == ((1.0, 7.0),)
    assert factual.vectors != alternative.vectors
    delta = representation_delta(factual, alternative)
    assert delta.vectors == ((-10.0, 5.0),)


def test_rollout_length_matches_events() -> None:
    start = _reprs(("g1",), ((0.0, 0.0),))
    steps = SequenceTable(
        group_ids=("g1", "g1"),
        event_ids=("t1", "t2"),
        positions=(0, 1),
        actor_ids=(None, None),
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    conditions = _reprs(("t1", "t2"), ((1.0, 0.0), (0.0, 2.0)))
    predicted = rollout(_AddEncoder(), start, steps, conditions)
    assert predicted.entity_ids == ("t1", "t2")
    assert predicted.vectors == ((1.0, 0.0), (1.0, 2.0))


def test_intervention_persists_delta_and_evaluation(tmp_path) -> None:
    store = LocalStore(tmp_path)
    contexts = _reprs(("s1",), ((1.0, 0.0),))
    conditions = _reprs(("a1", "a2"), ((0.0, 1.0), (2.0, 0.0)))
    query = InterventionQuery(
        query_ids=("q1",),
        context_ids=("s1",),
        factual_condition_ids=("a1",),
        alternative_condition_ids=("a2",),
    )
    factual, alternative = apply_intervention(
        _AddEncoder(),
        contexts,
        conditions,
        query,
    )
    delta = representation_delta(factual, alternative)
    run = RunContext(
        run_id="run-cf",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )
    delta_payload_id, _delta_record_id = put_feature_dataset(
        store,
        representation_payload_bytes(delta),
        run=run,
        inputs=[],
        media_type="application/json",
    )
    report = EvaluationReport(
        metrics={"representation_mse": representation_mse(factual, alternative)},
        n=1,
        n_missing=0,
    )
    _eval_payload_id, eval_record_id = put_evaluation_artifact(
        store,
        report,
        run=run,
        subject_payload_id=delta_payload_id,
        inputs=[delta_payload_id],
    )
    assert store.exists(delta_payload_id)
    assert store.exists(eval_record_id)


def test_interventions_have_no_environment_api() -> None:
    import ds_platform.modeling.interventions as interventions

    assert not hasattr(interventions, "step")
    assert not hasattr(interventions, "GameTree")


def test_interventions_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.interventions", _FORBIDDEN)
