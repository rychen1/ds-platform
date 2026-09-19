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
    """Identity of a payload: SHA-256 of the exact bytes, never a path."""
    return sha256_hex(data)


def record_id_from_bytes(data: bytes) -> str:
    """Identity of already-canonical record document bytes."""
    return sha256_hex(data)


def record_id(record: BaseModel) -> str:
    """Identity of a record: SHA-256 of its canonical JSON encoding."""
    return sha256_hex(canonical_json_bytes(record))


def canonical_json_bytes(value: Mapping[str, Any] | BaseModel) -> bytes:
    """Serialize a mapping or Pydantic model to frozen canonical JSON bytes.

    Rules: UTF-8, lexicographically sorted object keys, no insignificant
    whitespace, enums as values, UTC datetimes as ``YYYY-MM-DDTHH:MM:SSZ``,
    null/unset optional fields omitted, floats rejected.
    """
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, datetime):
        return _format_datetime(value)
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


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
