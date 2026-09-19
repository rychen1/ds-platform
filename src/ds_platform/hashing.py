"""Canonical JSON serialization and SHA-256 identity hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

SHA256_HEX_PATTERN = r"^[a-f0-9]{64}$"


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data`` (no algorithm prefix)."""
    return hashlib.sha256(data).hexdigest()


def payload_id(data: bytes) -> str:
    """Identity of a payload: SHA-256 of the exact bytes, never a path.

    v0 identity is one byte sequence per artifact. Multi-file datasets must
    be packed by the caller (single Parquet, zip, or a manifest the caller
    hashes as bytes). Directory paths are locations, not identities.
    """
    return sha256_hex(data)


def record_id_from_bytes(data: bytes) -> str:
    """Identity of already-canonical record document bytes.

    This is the identity of a stored sidecar. Do not re-canonicalize a
    parsed model from another schema version and expect the same digest.
    """
    return sha256_hex(data)


def record_id(record: BaseModel) -> str:
    """Identity of a current-schema record: SHA-256 of canonical JSON."""
    return sha256_hex(canonical_json_bytes(record))


def canonical_json_bytes(value: Mapping[str, Any] | BaseModel) -> bytes:
    """Serialize a mapping or Pydantic model to frozen canonical JSON bytes.

    Rules: UTF-8, lexicographically sorted object keys, no insignificant
    whitespace, enums as values, timezone-aware UTC datetimes as
    ``YYYY-MM-DDTHH:MM:SS.ffffffZ``, null/unset optional fields omitted,
    empty lists and tuples omitted, ``schema_version`` omitted when ``0``,
    floats rejected.
    """
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def format_aware_utc(value: datetime) -> str:
    """Return an injective UTC timestamp with microsecond precision."""
    if value.tzinfo is None:
        raise TypeError("canonical JSON requires timezone-aware datetimes")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        dumped = value.model_dump(mode="python", exclude_none=True)
        if dumped.get("schema_version") == 0:
            dumped.pop("schema_version")
        return _canonical_value(dumped)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in value.items()
            if not _omit_canonical_item(key, item)
        }
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, datetime):
        return format_aware_utc(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        raise TypeError("floats are not allowed in canonical JSON")
    if value is None:
        return None
    raise TypeError(f"unsupported type for canonical JSON: {type(value)!r}")


def _omit_canonical_item(key: str, item: Any) -> bool:
    if item is None:
        return True
    if item == [] or item == ():
        return True
    return key == "schema_version" and item == 0
