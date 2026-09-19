import json
import sys
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pytest
from pydantic import ValidationError

from ds_platform.hashing import payload_id
from ds_platform.types import (
    ArtifactKind,
    ArtifactRecord,
    Citation,
    Claim,
    ClaimLayer,
    ContractRef,
    Environment,
    Evidence,
    InferenceRecord,
    RelatedRef,
    RelationType,
    RunContext,
    artifact_record_json_schema,
    attribution_json_schema,
    dump_json_schema,
    new_run_id,
)

_PID = payload_id(b"types-fixture")
_CREATED = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _run() -> RunContext:
    return RunContext(
        run_id=new_run_id(),
        project="ri",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def _record(**overrides: object) -> ArtifactRecord:
    fields: dict[str, object] = {
        "payload_id": _PID,
        "media_type": "application/json",
        "kind": ArtifactKind.DOCUMENT,
        "produced_by": _run(),
        "created_at": _CREATED,
    }
    fields.update(overrides)
    return ArtifactRecord.model_validate(fields)


@pytest.mark.parametrize(
    "key",
    ["bga:corpus:v0", "ri:ranker:champion", "bga:raw/bgg/2026-03"],
)
def test_logical_key_accepts_project_namespaced_aliases(key: str) -> None:
    record = _record(logical_key=key)
    assert record.logical_key == key


@pytest.mark.parametrize(
    "key",
    ["not-namespaced", "BGA:Corpus", "bga:raw:bgg:2026-03", "bga:", ":corpus"],
)
def test_logical_key_rejects_malformed_aliases(key: str) -> None:
    with pytest.raises(ValidationError):
        _record(logical_key=key)


def test_artifact_record_rejects_unknown_kind_and_bad_ids() -> None:
    with pytest.raises(ValidationError):
        _record(kind="not-a-kind")
    with pytest.raises(ValidationError):
        _record(payload_id="not-a-hash")
    with pytest.raises(ValidationError):
        _record(inputs=["zzz"])


def test_artifact_record_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(
            {
                "payload_id": _PID,
                "media_type": "application/json",
                "kind": "document",
                "produced_by": _run().model_dump(),
                "created_at": _CREATED,
                "is_champion": True,
            }
        )


def test_claim_requires_layer_and_evidence_requires_excerpt() -> None:
    citation = Citation(payload_id=_PID)
    with pytest.raises(ValidationError):
        Claim(statement="the rule is X")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Evidence(citation=citation, excerpt="")
    claim = Claim(
        statement="the rule is X",
        layer=ClaimLayer.INFERRED,
        evidence=[Evidence(citation=citation, excerpt="from the rulebook")],
    )
    assert claim.layer is ClaimLayer.INFERRED
    assert claim.evidence is not None
    assert claim.evidence[0].citation.payload_id == _PID


def test_inference_record_requires_model_or_prompt_subject() -> None:
    with pytest.raises(ValidationError):
        InferenceRecord(
            input_payload_id=_PID,
            provider="openai",
            params_hash="temp-0",
            run_id="run-1",
        )
    record = InferenceRecord(
        prompt_payload_id=_PID,
        input_payload_id=_PID,
        provider="openai",
        params_hash="temp-0",
        run_id="run-1",
        cost="0.02",
        token_count=12,
    )
    assert record.prompt_payload_id == _PID


def test_contract_ref_and_related_refs_validate() -> None:
    record = _record(
        contract_ref=ContractRef(
            uri="https://example.test/game.schema.json",
            schema_hash=_PID,
        ),
        related=[RelatedRef(rel=RelationType.CLAIMS_ABOUT, payload_id=_PID)],
    )
    assert record.contract_ref is not None
    assert record.related[0].rel is RelationType.CLAIMS_ABOUT


def test_new_run_id_is_uuid4_hex() -> None:
    issued = new_run_id()
    assert len(issued) == 32
    int(issued, 16)


def test_committed_json_schemas_match_models() -> None:
    package_root = files("ds_platform").joinpath("schemas")
    artifact_path = Path(str(package_root.joinpath("artifact_record.schema.json")))
    attribution_path = Path(str(package_root.joinpath("attribution.schema.json")))
    assert artifact_path.read_text() == dump_json_schema(artifact_record_json_schema())
    assert attribution_path.read_text() == dump_json_schema(attribution_json_schema())
    assert json.loads(artifact_path.read_text())["title"] == "ArtifactRecord"


def test_core_import_graph_has_no_forbidden_libraries() -> None:
    import ds_platform  # noqa: F401

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
    loaded = forbidden.intersection(sys.modules)
    assert loaded == set()
