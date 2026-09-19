"""Content-addressed store protocol and local filesystem implementation."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ds_platform.hashing import (
    SHA256_HEX_PATTERN,
    canonical_json_bytes,
    record_id,
    sha256_hex,
)
from ds_platform.types import ArtifactRecord

_SHA256_HEX = re.compile(SHA256_HEX_PATTERN)


class HashMismatchError(ValueError):
    """``payload_id`` does not match the SHA-256 of the supplied bytes."""


class PayloadConflictError(ValueError):
    """A different payload already occupies this content-addressed key."""


class ArtifactNotFoundError(KeyError):
    """No payload exists for the given ``payload_id``."""


class Location(BaseModel):
    """A retrieval URI. Never an artifact identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str


class Store(Protocol):
    def put(self, payload_id: str, data: bytes, *, media_type: str) -> Location: ...

    def get(self, payload_id: str) -> bytes: ...

    def exists(self, payload_id: str) -> bool: ...

    def locate(self, payload_id: str) -> Location | None: ...


def put_record(store: Store, record: ArtifactRecord) -> str:
    """Persist canonical record JSON under ``record_id``. Not a service."""
    data = canonical_json_bytes(record)
    rid = record_id(record)
    store.put(rid, data, media_type="application/json")
    return rid


class LocalStore:
    """Write-once filesystem store: ``{root}/{ab}/{cd}/{sha256}``."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def put(self, payload_id: str, data: bytes, *, media_type: str) -> Location:
        del media_type  # persisted on the record, not in the kv
        _require_payload_id(payload_id)
        digest = sha256_hex(data)
        if payload_id != digest:
            raise HashMismatchError(
                f"payload_id {payload_id} does not match sha256 {digest}"
            )
        path = self._object_path(payload_id)
        if path.is_file():
            existing = path.read_bytes()
            if existing == data:
                return self._location(path)
            if sha256_hex(existing) == payload_id:
                raise PayloadConflictError(
                    f"payload {payload_id} already exists with different bytes"
                )
            # Existing bytes do not match the content-addressed key; repair.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return self._location(path)

    def get(self, payload_id: str) -> bytes:
        path = self._existing_path(payload_id)
        if path is None:
            raise ArtifactNotFoundError(payload_id)
        data = path.read_bytes()
        digest = sha256_hex(data)
        if digest != payload_id:
            raise HashMismatchError(f"stored bytes for {payload_id} hash to {digest}")
        return data

    def exists(self, payload_id: str) -> bool:
        return self._existing_path(payload_id) is not None

    def locate(self, payload_id: str) -> Location | None:
        path = self._existing_path(payload_id)
        if path is None:
            return None
        return self._location(path)

    def _object_path(self, payload_id: str) -> Path:
        return self._root / payload_id[:2] / payload_id[2:4] / payload_id

    def _existing_path(self, payload_id: str) -> Path | None:
        _require_payload_id(payload_id)
        path = self._object_path(payload_id)
        if path.is_file():
            return path
        return None

    def _location(self, path: Path) -> Location:
        return Location(uri=path.resolve().as_uri())


def _require_payload_id(payload_id: str) -> None:
    if _SHA256_HEX.fullmatch(payload_id) is None:
        raise ValueError(f"payload_id must be a lowercase sha256 hex: {payload_id!r}")
