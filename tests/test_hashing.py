from datetime import UTC, datetime

import pytest

from ds_platform.hashing import (
    canonical_json_bytes,
    payload_id,
    record_id_from_bytes,
    sha256_hex,
)


def test_identical_bytes_produce_identical_payload_id() -> None:
    data = b"<boardgame>same</boardgame>"
    assert payload_id(data) == payload_id(data)
    assert payload_id(data) == sha256_hex(data)


def test_different_bytes_produce_different_payload_id() -> None:
    assert payload_id(b"alpha") != payload_id(b"beta")


def test_payload_id_is_lowercase_hex_without_prefix() -> None:
    digest = payload_id(b"fixture")
    assert len(digest) == 64
    assert digest == digest.lower()
    assert not digest.startswith("sha256:")
    int(digest, 16)


def test_canonical_json_is_stable_across_key_order() -> None:
    first = canonical_json_bytes({"b": 2, "a": 1})
    second = canonical_json_bytes({"a": 1, "b": 2})
    assert first == second
    assert first == b'{"a":1,"b":2}'


def test_canonical_json_omits_nulls_sorts_nested_keys() -> None:
    encoded = canonical_json_bytes({"z": {"b": "x", "a": None}, "m": None, "a": True})
    assert encoded == b'{"a":true,"z":{"b":"x"}}'


def test_canonical_json_formats_datetime_as_utc_microseconds_z() -> None:
    instant = datetime(2026, 3, 14, 15, 9, 26, 123456, tzinfo=UTC)
    encoded = canonical_json_bytes({"created_at": instant})
    assert encoded == b'{"created_at":"2026-03-14T15:09:26.123456Z"}'


def test_canonical_json_rejects_naive_datetime() -> None:
    with pytest.raises(TypeError, match="timezone-aware"):
        canonical_json_bytes({"created_at": datetime(2026, 3, 14, 15, 9, 26)})


def test_canonical_json_omits_empty_lists() -> None:
    encoded = canonical_json_bytes({"a": 1, "inputs": [], "related": []})
    assert encoded == b'{"a":1}'


def test_record_id_from_bytes_matches_sha256() -> None:
    data = b'{"kind":"raw"}'
    assert record_id_from_bytes(data) == sha256_hex(data)
