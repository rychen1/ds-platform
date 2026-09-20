"""Thin artifact record helpers for modeling outputs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from ds_platform.hashing import format_aware_utc, payload_id
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.modeling.compare import ComparisonReport
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.features import Scalar
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import FeatureSchema
from ds_platform.modeling.split import SplitAssignment, split_assignment_bytes
from ds_platform.store import Store, put_record
from ds_platform.types import (
    ArtifactKind,
    ArtifactRecord,
    ContractRef,
    Evidence,
    RelatedRef,
    RelationType,
    RunContext,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredictionRow(_FrozenModel):
    """One supervised prediction aligned to an opaque entity id."""

    entity_id: str = Field(min_length=1)
    y_pred: Scalar
    y_true: Scalar | None = None
    y_proba: tuple[float, ...] | None = None
    evidence: tuple[Evidence, ...] = ()


def put_model_artifact(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist model bytes and return ``(payload_id, record_id)``."""
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.MODEL,
        run=run,
        inputs=inputs,
        media_type=media_type,
        logical_key=logical_key,
        contract_ref=contract_ref,
        created_at=created_at,
    )


def representation_payload_bytes(table: RepresentationTable) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a representation table.

    Floats use the spec canonical encoding so payload identity is
    bit-stable. The artifact envelope itself remains float-free.
    """
    return spec_canonical_json_bytes(table)


def put_index_artifact(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist index bytes as ``kind=index``. The platform does not query them."""
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.INDEX,
        run=run,
        inputs=inputs,
        media_type=media_type,
        logical_key=logical_key,
        contract_ref=contract_ref,
        created_at=created_at,
    )


def put_representation_artifact(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist representation-table bytes as ``kind=representation``."""
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.REPRESENTATION,
        run=run,
        inputs=inputs,
        media_type=media_type,
        logical_key=logical_key,
        contract_ref=contract_ref,
        created_at=created_at,
    )


def feature_schema_bytes(schema: FeatureSchema) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a feature schema."""
    return spec_canonical_json_bytes(schema)


def put_feature_schema(
    store: Store,
    schema: FeatureSchema,
    *,
    run: RunContext,
    inputs: Sequence[str],
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist a feature schema as a ``document`` artifact."""
    return _put_artifact(
        store,
        feature_schema_bytes(schema),
        kind=ArtifactKind.DOCUMENT,
        run=run,
        inputs=inputs,
        media_type="application/json",
        logical_key=logical_key,
        created_at=created_at,
    )


def put_split_assignment(
    store: Store,
    assignment: SplitAssignment,
    *,
    run: RunContext,
    inputs: Sequence[str],
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist a split assignment as a ``document`` artifact."""
    return _put_artifact(
        store,
        split_assignment_bytes(assignment),
        kind=ArtifactKind.DOCUMENT,
        run=run,
        inputs=inputs,
        media_type="application/json",
        logical_key=logical_key,
        created_at=created_at,
    )


def cluster_assignment_payload_bytes(
    entity_ids: Sequence[str],
    labels: Sequence[str | int],
) -> bytes:
    """Return deterministic JSON bytes mapping entity ids to cluster labels."""
    if len(entity_ids) != len(labels):
        raise ValueError("entity_ids length must match labels length")
    ordered = sorted(
        zip(entity_ids, labels, strict=True),
        key=lambda item: item[0],
    )
    payload = {
        "assignments": [
            {"entity_id": entity_id, "label": label} for entity_id, label in ordered
        ]
    }
    return spec_canonical_json_bytes(payload)


def put_cluster_assignment(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str = "application/json",
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist cluster assignment bytes as a ``document`` artifact."""
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.DOCUMENT,
        run=run,
        inputs=inputs,
        media_type=media_type,
        logical_key=logical_key,
        created_at=created_at,
    )


def put_feature_dataset(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist feature-table bytes as ``kind=dataset``."""
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.DATASET,
        run=run,
        inputs=inputs,
        media_type=media_type,
        logical_key=logical_key,
        contract_ref=contract_ref,
        created_at=created_at,
    )


def put_prediction_artifact(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist prediction payload bytes and return ``(payload_id, record_id)``."""
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.PREDICTION,
        run=run,
        inputs=inputs,
        media_type=media_type,
        logical_key=logical_key,
        contract_ref=contract_ref,
        created_at=created_at,
    )


def prediction_payload_bytes(rows: Sequence[PredictionRow]) -> bytes:
    """Return deterministic UTF-8 JSONL bytes for prediction rows.

    Rows are sorted by ``entity_id`` so payload identity does not depend on
    input order. Floats use spec canonical encoding. This is a convenience
    schema for scalar supervised predictions; callers may pass other bytes
    to ``put_prediction_artifact``.
    """
    ordered = sorted(rows, key=lambda row: row.entity_id)
    encoded_rows = [spec_canonical_json_bytes(row).decode("utf-8") for row in ordered]
    if not encoded_rows:
        return b""
    return ("\n".join(encoded_rows) + "\n").encode("utf-8")


def put_evaluation_artifact(
    store: Store,
    report: EvaluationReport,
    *,
    run: RunContext,
    subject_payload_id: str,
    inputs: Sequence[str],
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist an evaluation report with ``evaluation_of`` provenance."""
    data = evaluation_report_bytes(report)
    return _put_artifact(
        store,
        data,
        kind=ArtifactKind.EVALUATION,
        run=run,
        inputs=inputs,
        media_type="application/json",
        logical_key=logical_key,
        related=[
            RelatedRef(
                rel=RelationType.EVALUATION_OF,
                payload_id=subject_payload_id,
            )
        ],
        created_at=created_at,
    )


def evaluation_report_bytes(report: EvaluationReport) -> bytes:
    """Return deterministic UTF-8 JSON bytes for an evaluation report."""
    return spec_canonical_json_bytes(report)


def comparison_report_bytes(report: ComparisonReport) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a comparison report."""
    return spec_canonical_json_bytes(report)


def put_comparison_artifact(
    store: Store,
    report: ComparisonReport,
    *,
    run: RunContext,
    inputs: Sequence[str],
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    """Persist a comparison report as ``kind=evaluation``.

    ``inputs`` should cite the compared evaluation artifacts. ``related``
    records ``evaluation_of`` both ``left_payload_id`` and
    ``right_payload_id``.
    """
    return _put_artifact(
        store,
        comparison_report_bytes(report),
        kind=ArtifactKind.EVALUATION,
        run=run,
        inputs=inputs,
        media_type="application/json",
        logical_key=logical_key,
        related=[
            RelatedRef(
                rel=RelationType.EVALUATION_OF,
                payload_id=report.left_payload_id,
            ),
            RelatedRef(
                rel=RelationType.EVALUATION_OF,
                payload_id=report.right_payload_id,
            ),
        ],
        created_at=created_at,
    )


def _put_artifact(
    store: Store,
    data: bytes,
    *,
    kind: ArtifactKind,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    related: Sequence[RelatedRef] | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    artifact_payload_id = payload_id(data)
    store.put(artifact_payload_id, data, media_type=media_type)
    record = ArtifactRecord(
        payload_id=artifact_payload_id,
        media_type=media_type,
        kind=kind,
        produced_by=run,
        created_at=_created_at(run, created_at),
        logical_key=logical_key,
        contract_ref=contract_ref,
        inputs=tuple(inputs),
        related=tuple(related or ()),
    )
    record_id_value = put_record(store, record)
    return artifact_payload_id, record_id_value


def _created_at(run: RunContext, created_at: datetime | None) -> datetime:
    instant = created_at if created_at is not None else run.started_at
    if instant.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    format_aware_utc(instant)
    return instant.astimezone(UTC)
