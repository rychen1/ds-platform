from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ds_platform.hashing import canonical_json_bytes, payload_id, record_id, sha256_hex
from ds_platform.store import LocalStore, Location
from ds_platform.types import (
    ArtifactKind,
    ArtifactRecord,
    Environment,
    RelatedRef,
    RelationType,
    RunContext,
)

_PAYLOAD = b"subject-bytes"
_PID = payload_id(_PAYLOAD)
_CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _run() -> RunContext:
    return RunContext(
        run_id="abc123",
        project="bga",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def _record(**overrides: object) -> ArtifactRecord:
    fields: dict[str, object] = {
        "payload_id": _PID,
        "media_type": "application/xml",
        "kind": ArtifactKind.RAW,
        "produced_by": _run(),
        "created_at": _CREATED,
    }
    fields.update(overrides)
    return ArtifactRecord.model_validate(fields)


def test_payload_id_is_independent_of_filename_and_path(tmp_path: Path) -> None:
    left = tmp_path / "one" / "dump.xml"
    right = tmp_path / "two" / "other-name.xml"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(_PAYLOAD)
    right.write_bytes(_PAYLOAD)
    assert payload_id(left.read_bytes()) == payload_id(right.read_bytes()) == _PID


def test_metadata_change_changes_record_id_not_payload_id() -> None:
    quality_subject = payload_id(b"quality-subject")
    first = _record(logical_key="bga:corpus:v0")
    second = _record(
        logical_key="bga:corpus:v1",
        related=[
            RelatedRef(rel=RelationType.QUALITY_FOR, payload_id=quality_subject),
        ],
    )
    assert first.payload_id == second.payload_id == _PID
    assert record_id(first) != record_id(second)


def test_location_is_not_part_of_record_identity() -> None:
    record = _record()
    dumped = canonical_json_bytes(record)
    assert b"location" not in dumped
    assert b"record_id" not in dumped
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(
            {**record.model_dump(), "location": "file:///tmp/nope"}
        )
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(
            {**record.model_dump(), "record_id": sha256_hex(b"x")}
        )


def test_location_object_is_not_hashed_into_record() -> None:
    record = _record()
    rid = record_id(record)
    location = Location(uri="file:///tmp/store/ab/cd/" + _PID)
    assert location.uri not in canonical_json_bytes(record).decode()
    assert rid == record_id(record)


def test_record_id_equals_payload_id_of_canonical_sidecar_bytes() -> None:
    record = _record(logical_key="ri:ranker:champion")
    sidecar = canonical_json_bytes(record)
    assert record_id(record) == payload_id(sidecar) == sha256_hex(sidecar)


def test_same_bytes_in_two_store_roots_share_payload_id(tmp_path: Path) -> None:
    first = LocalStore(tmp_path / "a")
    second = LocalStore(tmp_path / "b")
    left = first.put(_PID, _PAYLOAD, media_type="application/xml")
    right = second.put(_PID, _PAYLOAD, media_type="application/xml")
    assert left.uri != right.uri
    assert first.get(_PID) == second.get(_PID) == _PAYLOAD
