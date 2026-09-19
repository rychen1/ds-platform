"""Thin encoding orchestration for representation artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from ds_platform.modeling.capabilities import Embedder
from ds_platform.modeling.records import (
    put_feature_dataset,
    put_model_artifact,
    representation_payload_bytes,
)
from ds_platform.modeling.spec import EncodingSpec, spec_config_hash
from ds_platform.store import Store
from ds_platform.types import RunContext


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EncodingResult(_FrozenModel):
    run_id: str
    config_hash: str
    representation_payload_id: str
    model_payload_id: str


def run_encoding(
    spec: EncodingSpec,
    *,
    encoder: Embedder,
    entity_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
    store: Store,
    run: RunContext,
    serialize_model: Callable[[object], bytes],
    model_media_type: str = "application/octet-stream",
    inputs: Sequence[str] = (),
) -> EncodingResult:
    """Fit and encode records, then persist representation and model artifacts.

    This is optional lineage glue. Callers may use :class:`Embedder` and
    the record helpers directly.
    """
    if len(entity_ids) != len(records):
        raise ValueError("entity_ids length must match records length")

    config_hash = spec_config_hash(spec)
    run_with_hash = run.model_copy(update={"config_hash": config_hash})

    encoder.fit(entity_ids, records)
    table = encoder.encode(entity_ids, records)
    if table.dim != spec.dim:
        raise ValueError(
            f"encoder dim {table.dim} does not match EncodingSpec.dim {spec.dim}"
        )
    if tuple(table.entity_ids) != tuple(entity_ids):
        raise ValueError("encoder must return representations keyed by entity_ids")

    input_ids = list(inputs)
    representation_payload_id, _representation_record_id = put_feature_dataset(
        store,
        representation_payload_bytes(table),
        run=run_with_hash,
        inputs=input_ids,
        media_type="application/json",
    )
    model_payload_id, _model_record_id = put_model_artifact(
        store,
        serialize_model(encoder),
        run=run_with_hash,
        inputs=[representation_payload_id, *input_ids],
        media_type=model_media_type,
    )
    return EncodingResult(
        run_id=run_with_hash.run_id,
        config_hash=config_hash,
        representation_payload_id=representation_payload_id,
        model_payload_id=model_payload_id,
    )
