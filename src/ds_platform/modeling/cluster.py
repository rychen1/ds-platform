"""Thin clustering orchestration for representation artifacts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from ds_platform.modeling.capabilities import Clusterer
from ds_platform.modeling.records import (
    cluster_assignment_payload_bytes,
    put_cluster_assignment,
    put_model_artifact,
)
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import ModelSpec, spec_config_hash
from ds_platform.store import Store
from ds_platform.types import RunContext


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClusteringResult(_FrozenModel):
    run_id: str
    config_hash: str
    assignment_payload_id: str
    model_payload_id: str
    labels: tuple[str | int, ...]


def run_clustering(
    spec: ModelSpec,
    *,
    clusterer: Clusterer,
    table: RepresentationTable,
    store: Store,
    run: RunContext,
    serialize_model: Callable[[object], bytes],
    model_media_type: str = "application/octet-stream",
    inputs: Sequence[str] = (),
) -> ClusteringResult:
    """Fit a clusterer on a representation table and persist assignments.

    This is optional lineage glue. Callers may use :class:`Clusterer` and
    the record helpers directly.
    """
    config_hash = spec_config_hash(spec)
    run_with_hash = run.model_copy(update={"config_hash": config_hash})

    clusterer.fit(table)
    labels = tuple(clusterer.predict(table))
    if len(labels) != len(table.entity_ids):
        raise ValueError("clusterer must return one label per entity")

    input_ids = list(inputs)
    assignment_payload_id, _assignment_record_id = put_cluster_assignment(
        store,
        cluster_assignment_payload_bytes(table.entity_ids, labels),
        run=run_with_hash,
        inputs=input_ids,
        media_type="application/json",
    )
    model_payload_id, _model_record_id = put_model_artifact(
        store,
        serialize_model(clusterer),
        run=run_with_hash,
        inputs=[assignment_payload_id, *input_ids],
        media_type=model_media_type,
    )
    return ClusteringResult(
        run_id=run_with_hash.run_id,
        config_hash=config_hash,
        assignment_payload_id=assignment_payload_id,
        model_payload_id=model_payload_id,
        labels=labels,
    )
