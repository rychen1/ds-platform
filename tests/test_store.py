from datetime import UTC, datetime
from pathlib import Path

import pytest

from ds_platform.hashing import canonical_json_bytes, payload_id, record_id
from ds_platform.store import (
    ArtifactNotFoundError,
    HashMismatchError,
    LocalStore,
    PayloadConflictError,
)
from ds_platform.types import ArtifactKind, ArtifactRecord, Environment, RunContext

_PAYLOAD = b"store-bytes"
_PID = payload_id(_PAYLOAD)


def test_put_get_round_trips_bytes(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    location = store.put(_PID, _PAYLOAD, media_type="application/octet-stream")
    assert store.get(_PID) == _PAYLOAD
    assert location.uri.startswith("file://")
    assert location.uri.endswith(_PID)


def test_content_addressed_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.put(_PID, _PAYLOAD, media_type="text/plain")
    expected = tmp_path.resolve() / _PID[:2] / _PID[2:4] / _PID
    assert expected.is_file()
    assert expected.read_bytes() == _PAYLOAD


def test_exists_and_locate(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    missing = payload_id(b"absent")
    assert store.exists(_PID) is False
    assert store.locate(_PID) is None
    assert store.exists(missing) is False
    store.put(_PID, _PAYLOAD, media_type="text/plain")
    assert store.exists(_PID) is True
    located = store.locate(_PID)
    assert located is not None
    assert located.uri.endswith(_PID)
    assert store.locate(missing) is None


def test_get_missing_raises(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        store.get(_PID)


def test_put_rejects_hash_mismatch(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(HashMismatchError):
        store.put(payload_id(b"other"), _PAYLOAD, media_type="text/plain")


def test_put_is_idempotent_for_identical_bytes(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    first = store.put(_PID, _PAYLOAD, media_type="text/plain")
    second = store.put(_PID, _PAYLOAD, media_type="application/xml")
    assert first == second
    assert store.get(_PID) == _PAYLOAD


def test_put_does_not_silently_mutate_existing_payload(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.put(_PID, _PAYLOAD, media_type="text/plain")
    # Same id cannot be claimed by different bytes; caller must hash first.
    with pytest.raises(HashMismatchError):
        store.put(_PID, b"mutated-bytes", media_type="text/plain")
    colliding = tmp_path / _PID[:2] / _PID[2:4] / _PID
    colliding.write_bytes(b"tampered")
    with pytest.raises(PayloadConflictError):
        store.put(_PID, _PAYLOAD, media_type="text/plain")
    assert store.get(_PID) == b"tampered"


def test_records_are_stored_through_the_same_store(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.put(_PID, _PAYLOAD, media_type="application/xml")
    record = ArtifactRecord(
        payload_id=_PID,
        media_type="application/xml",
        kind=ArtifactKind.RAW,
        produced_by=RunContext(
            run_id="run-store",
            project="bga",
            started_at=datetime(2026, 2, 2, 2, 2, 2, tzinfo=UTC),
            environment=Environment.CI,
        ),
        created_at=datetime(2026, 2, 2, 2, 2, 2, tzinfo=UTC),
        logical_key="bga:raw/bgg/thing-123",
    )
    sidecar = canonical_json_bytes(record)
    rid = record_id(record)
    store.put(rid, sidecar, media_type="application/json")
    loaded = store.get(rid)
    assert loaded == sidecar
    assert record_id(ArtifactRecord.model_validate_json(loaded)) == rid
    assert store.get(_PID) == _PAYLOAD


def test_invalid_payload_id_is_rejected(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError, match="payload_id"):
        store.exists("../etc/passwd")
    with pytest.raises(ValueError, match="payload_id"):
        store.locate("not-a-hash")
