"""Counterfactual queries over a conditioned encoder. Not an environment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from ds_platform.modeling.capabilities import ConditionedEncoder
from ds_platform.modeling.pairs import PairTable, align_pairs
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.sequences import SequenceTable, events_for_group, order_events


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InterventionQuery(_FrozenModel):
    """Same-context alternative-condition query.

    Invariants (enforced at construction):

    - all id tuples have equal length
    - ``query_ids`` are unique
    """

    query_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    factual_condition_ids: tuple[str, ...]
    alternative_condition_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> InterventionQuery:
        n_queries = len(self.query_ids)
        if len(self.context_ids) != n_queries:
            raise ValueError("context_ids length must match query_ids length")
        if len(self.factual_condition_ids) != n_queries:
            raise ValueError("factual_condition_ids length must match query_ids length")
        if len(self.alternative_condition_ids) != n_queries:
            raise ValueError(
                "alternative_condition_ids length must match query_ids length"
            )
        if len(set(self.query_ids)) != n_queries:
            raise ValueError("query_ids must be unique")
        return self


def apply_intervention(
    model: ConditionedEncoder,
    contexts: RepresentationTable,
    conditions: RepresentationTable,
    query: InterventionQuery,
) -> tuple[RepresentationTable, RepresentationTable]:
    """Encode factual and alternative conditions for the same contexts.

    Both outputs are keyed by ``query.query_ids``.
    """
    sources = ("",) * len(query.query_ids)
    factual_pairs = PairTable(
        pair_ids=query.query_ids,
        context_ids=query.context_ids,
        condition_ids=query.factual_condition_ids,
        source_payload_ids=sources,
    )
    alternative_pairs = PairTable(
        pair_ids=query.query_ids,
        context_ids=query.context_ids,
        condition_ids=query.alternative_condition_ids,
        source_payload_ids=sources,
    )
    factual_contexts, factual_conditions = align_pairs(
        contexts,
        conditions,
        factual_pairs,
    )
    alternative_contexts, alternative_conditions = align_pairs(
        contexts,
        conditions,
        alternative_pairs,
    )
    factual = model.encode(factual_contexts, factual_conditions, query.query_ids)
    alternative = model.encode(
        alternative_contexts,
        alternative_conditions,
        query.query_ids,
    )
    if tuple(factual.entity_ids) != query.query_ids:
        raise ValueError("conditioned encoder must return rows keyed by query_ids")
    if tuple(alternative.entity_ids) != query.query_ids:
        raise ValueError("conditioned encoder must return rows keyed by query_ids")
    return factual, alternative


def representation_delta(
    factual: RepresentationTable,
    alternative: RepresentationTable,
) -> RepresentationTable:
    """Return elementwise ``alternative - factual`` aligned to ``factual`` ids."""
    if factual.dim != alternative.dim:
        raise ValueError("representation tables must have the same dim")
    if tuple(factual.entity_ids) != tuple(alternative.entity_ids):
        raise ValueError("representation tables must share the same entity_ids")
    vectors = tuple(
        tuple(
            alt_value - fact_value
            for fact_value, alt_value in zip(fact_vector, alt_vector, strict=True)
        )
        for fact_vector, alt_vector in zip(
            factual.vectors,
            alternative.vectors,
            strict=True,
        )
    )
    return RepresentationTable(
        entity_ids=factual.entity_ids,
        vectors=vectors,
        dim=factual.dim,
        source_payload_ids=factual.source_payload_ids,
    )


def rollout(
    model: ConditionedEncoder,
    start: RepresentationTable,
    steps: SequenceTable,
    conditions: RepresentationTable,
) -> RepresentationTable:
    """Open-loop multi-step prediction along ``steps``.

    ``start`` is keyed by group id. Each event uses ``conditions[event_id]``
    and feeds the predicted vector forward. Closed-loop simulator rollouts
    remain consumer-side.
    """
    ordered = order_events(steps)
    start_lookup = dict(zip(start.entity_ids, start.vectors, strict=True))
    start_sources = dict(zip(start.entity_ids, start.source_payload_ids, strict=True))
    current = {
        group_id: start_lookup[group_id]
        for group_id in _unique_in_order(ordered.group_ids)
    }
    event_ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    output_dim: int | None = None
    for group_id in _unique_in_order(ordered.group_ids):
        if group_id not in start_lookup:
            raise KeyError(f"start representation not found for group {group_id!r}")
        group_events = events_for_group(ordered, group_id)
        state = current[group_id]
        for event_id, source in zip(
            group_events.event_ids,
            group_events.source_payload_ids,
            strict=True,
        ):
            context = RepresentationTable(
                entity_ids=(event_id,),
                vectors=(state,),
                dim=len(state),
                source_payload_ids=(start_sources[group_id],),
            )
            try:
                condition_index = conditions.entity_ids.index(event_id)
            except ValueError as exc:
                raise KeyError(
                    f"condition representation not found for event {event_id!r}"
                ) from exc
            condition = RepresentationTable(
                entity_ids=(event_id,),
                vectors=(conditions.vectors[condition_index],),
                dim=conditions.dim,
                source_payload_ids=(conditions.source_payload_ids[condition_index],),
            )
            predicted = model.encode(context, condition, (event_id,))
            if tuple(predicted.entity_ids) != (event_id,):
                raise ValueError("conditioned encoder must return the queried event id")
            if output_dim is None:
                output_dim = predicted.dim
            elif predicted.dim != output_dim:
                raise ValueError("rollout predictions must have a consistent dim")
            state = predicted.vectors[0]
            event_ids.append(event_id)
            vectors.append(state)
            sources.append(source)
    return RepresentationTable(
        entity_ids=tuple(event_ids),
        vectors=tuple(vectors),
        dim=0 if output_dim is None else output_dim,
        source_payload_ids=tuple(sources),
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
