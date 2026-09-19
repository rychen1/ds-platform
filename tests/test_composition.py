"""Non-BGG composition: raw → document → dataset → quality → claim_set."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    Citation,
    Claim,
    ClaimLayer,
    ContractRef,
    Environment,
    LocalStore,
    RelatedRef,
    RelationType,
    RunContext,
    payload_id,
    put_record,
    record_id,
    sha256_hex,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCHEMA_PATH = FIXTURES / "example_cafe.schema.json"
CREATED = datetime(2026, 9, 19, 16, 10, tzinfo=UTC)

RAW_BYTES = b'{"html": "<menu>soup</menu>"}'
DOCUMENT_BYTES = b'{\n  "name": "Example Cafe"\n}\n'
DATASET_BYTES = b'{"name":"Example Cafe"}\n'
QUALITY_BYTES = b'{"null_fields": 0}\n'


def _run() -> RunContext:
    return RunContext(
        run_id="composition-run",
        project="demo",
        started_at=CREATED,
        environment=Environment.LOCAL,
    )


def _cafe_contract() -> ContractRef:
    data = SCHEMA_PATH.read_bytes()
    uri = json.loads(data)["$id"]
    assert isinstance(uri, str)
    return ContractRef(uri=uri, schema_hash=sha256_hex(data))


def _store_payload(
    store: LocalStore,
    data: bytes,
    *,
    media_type: str,
    kind: ArtifactKind,
    logical_key: str,
    inputs: list[str] | None = None,
    related: list[RelatedRef] | None = None,
    contract_ref: ContractRef | None = None,
) -> tuple[str, str, ArtifactRecord]:
    pid = payload_id(data)
    store.put(pid, data, media_type=media_type)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=media_type,
        kind=kind,
        produced_by=_run(),
        created_at=CREATED,
        logical_key=logical_key,
        contract_ref=contract_ref,
        inputs=inputs or [],
        related=related or [],
    )
    rid = put_record(store, record)
    return pid, rid, record


def test_non_xml_chain_uses_existing_types_only(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    raw_pid, _raw_rid, raw = _store_payload(
        store,
        RAW_BYTES,
        media_type="application/json",
        kind=ArtifactKind.RAW,
        logical_key="demo:raw/html/menu-1",
    )
    document_pid, _doc_rid, document = _store_payload(
        store,
        DOCUMENT_BYTES,
        media_type="application/json",
        kind=ArtifactKind.DOCUMENT,
        logical_key="demo:document/place/example-cafe",
        inputs=[raw_pid],
        contract_ref=_cafe_contract(),
    )
    dataset_pid, _ds_rid, dataset = _store_payload(
        store,
        DATASET_BYTES,
        media_type="application/jsonl",
        kind=ArtifactKind.DATASET,
        logical_key="demo:dataset:v0",
        inputs=[document_pid],
    )
    quality_pid, _q_rid, quality = _store_payload(
        store,
        QUALITY_BYTES,
        media_type="application/json",
        kind=ArtifactKind.QUALITY,
        logical_key="demo:quality/dataset:v0",
        related=[RelatedRef(rel=RelationType.QUALITY_FOR, payload_id=dataset_pid)],
    )
    claim = Claim(
        statement="Example Cafe is named in the source menu document",
        layer=ClaimLayer.INFERRED,
        confidence_policy="fixture",
        subject_payload_id=document_pid,
        citations=[Citation(payload_id=document_pid, pointer="/name")],
    )
    claim_bytes = (json.dumps([claim.model_dump(mode="json")], indent=2) + "\n").encode(
        "utf-8"
    )
    claim_pid, _c_rid, claims = _store_payload(
        store,
        claim_bytes,
        media_type="application/json",
        kind=ArtifactKind.CLAIM_SET,
        logical_key="demo:claims/place/example-cafe",
        related=[RelatedRef(rel=RelationType.CLAIMS_ABOUT, payload_id=document_pid)],
    )

    assert raw_pid != document_pid != dataset_pid
    assert document.inputs == [raw_pid]
    assert document.contract_ref == _cafe_contract()
    assert dataset.inputs == [document_pid]
    assert quality.related[0].rel is RelationType.QUALITY_FOR
    assert quality.related[0].payload_id == dataset_pid
    assert claims.related[0].rel is RelationType.CLAIMS_ABOUT
    assert claims.related[0].payload_id == document_pid
    assert claim_pid != document_pid
    assert quality_pid != dataset_pid

    assert store.get(raw_pid) == RAW_BYTES
    assert store.get(document_pid) == DOCUMENT_BYTES
    assert store.get(dataset_pid) == DATASET_BYTES
    assert payload_id(store.get(dataset_pid)) == dataset_pid
    assert payload_id(store.get(document_pid)) == document_pid

    for record in (raw, document, dataset, quality, claims):
        dumped = record.model_dump()
        assert "location" not in dumped
        assert "url" not in dumped


def test_put_record_round_trips_canonical_bytes(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    pid = payload_id(DOCUMENT_BYTES)
    store.put(pid, DOCUMENT_BYTES, media_type="application/json")
    record = ArtifactRecord(
        payload_id=pid,
        media_type="application/json",
        kind=ArtifactKind.DOCUMENT,
        produced_by=_run(),
        created_at=CREATED,
        logical_key="demo:document/place/example-cafe",
        inputs=[payload_id(RAW_BYTES)],
        contract_ref=_cafe_contract(),
    )
    rid = put_record(store, record)
    assert rid == record_id(record)
    loaded = ArtifactRecord.model_validate_json(store.get(rid))
    assert record_id(loaded) == rid
    assert "location" not in loaded.model_dump()
    assert store.get(pid) == DOCUMENT_BYTES


def test_composition_import_graph_has_no_vendor_stack() -> None:
    forbidden = {
        "boto3",
        "dagster",
        "dask",
        "duckdb",
        "mlflow",
        "pandas",
        "polars",
        "pyarrow",
        "pyspark",
        "ray",
        "sqlmesh",
        "streamlit",
    }
    assert forbidden.intersection(sys.modules) == set()
