"""Paired context/condition rows for conditioned encoding."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from ds_platform.modeling.representations import RepresentationTable, select_entities


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PairTable(_FrozenModel):
    """Opaque context/condition pairs keyed by pair id.

    Invariants (enforced at construction):

    - ``len(context_ids) == len(condition_ids) == len(source_payload_ids)``
    - those lengths match ``len(pair_ids)``
    - ``pair_ids`` contain no duplicates
    """

    pair_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    source_payload_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> PairTable:
        n_pairs = len(self.pair_ids)
        if len(self.context_ids) != n_pairs:
            raise ValueError("context_ids length must match pair_ids length")
        if len(self.condition_ids) != n_pairs:
            raise ValueError("condition_ids length must match pair_ids length")
        if len(self.source_payload_ids) != n_pairs:
            raise ValueError("source_payload_ids length must match pair_ids length")
        if len(set(self.pair_ids)) != n_pairs:
            raise ValueError("pair_ids must be unique")
        return self


def align_pairs(
    contexts: RepresentationTable,
    conditions: RepresentationTable,
    pairs: PairTable,
) -> tuple[RepresentationTable, RepresentationTable]:
    """Return context and condition tables aligned to ``pairs.pair_ids``.

    Raises ``KeyError`` when a pair's context or condition id is absent.
    """
    aligned_contexts = select_entities(contexts, pairs.context_ids)
    aligned_conditions = select_entities(conditions, pairs.condition_ids)
    rekeyed_contexts = RepresentationTable(
        entity_ids=pairs.pair_ids,
        vectors=aligned_contexts.vectors,
        dim=aligned_contexts.dim,
        source_payload_ids=pairs.source_payload_ids,
    )
    rekeyed_conditions = RepresentationTable(
        entity_ids=pairs.pair_ids,
        vectors=aligned_conditions.vectors,
        dim=aligned_conditions.dim,
        source_payload_ids=pairs.source_payload_ids,
    )
    return rekeyed_contexts, rekeyed_conditions
