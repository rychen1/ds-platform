"""Integration model artifact envelope for native vendor payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ds_platform.modeling._spec_json import spec_canonical_json_bytes

_ARTIFACT_SEPARATOR = b"\n---\n"


def pack_integration_model(
    *,
    library: str,
    task: str,
    metadata: Mapping[str, Any],
    model_bytes: bytes,
) -> bytes:
    """Wrap native model bytes with deterministic integration metadata."""
    header = {
        "schema_version": 1,
        "library": library,
        "task": task,
        **dict(metadata),
    }
    return spec_canonical_json_bytes(header) + _ARTIFACT_SEPARATOR + model_bytes


def unpack_integration_model(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Split an integration artifact into metadata and native model bytes."""
    try:
        header_bytes, model_bytes = data.split(_ARTIFACT_SEPARATOR, 1)
    except ValueError as exc:
        raise ValueError("invalid integration model artifact") from exc
    header = json.loads(header_bytes.decode("utf-8"))
    if header.get("schema_version") != 1:
        raise ValueError(
            f"unsupported integration artifact schema: {header.get('schema_version')!r}"
        )
    return header, model_bytes
