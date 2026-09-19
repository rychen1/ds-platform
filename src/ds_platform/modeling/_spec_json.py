"""Private canonical JSON for future modeling experiment specs.

Unlike artifact ``canonical_json_bytes`` in ``ds_platform.hashing``, this
encoder accepts ``float`` leaves, timezone-aware datetimes, and
``datetime.date`` values so experiment configuration can be hashed into
``RunContext.config_hash``.

Do **not** use this for ``ArtifactRecord`` envelopes. Artifact record
hashing intentionally rejects floats and remains unchanged.

Float canonicalization
----------------------
Each float leaf becomes a tagged object ``{"$spec_float": "<text>"}`` so
floats never appear as JSON numbers (which would conflate with integers or
invite parser drift).

``<text>`` is one of:

- ``NaN`` — not-a-number
- ``Infinity`` / ``-Infinity`` — positive / negative infinity
- ``-0`` — negative zero (distinct from positive zero)
- otherwise the **lowercase 16-character big-endian IEEE-754 binary64 hex**
  digest produced by ``struct.pack(">d", value).hex()``

Hex is deterministic across runs and Python builds, independent of ``repr``
or locale, and round-trips the exact float bit pattern.

Date canonicalization
---------------------
``datetime.datetime`` leaves use the same injective UTC form as artifact
records. Naive datetimes are rejected. ``datetime.date`` leaves serialize
as ISO 8601 calendar dates: ``YYYY-MM-DD``.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from ds_platform.hashing import format_aware_utc

_SPEC_FLOAT_TAG = "$spec_float"


def spec_canonical_json_bytes(value: Mapping[str, Any] | BaseModel) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a spec-shaped mapping or model."""
    return json.dumps(
        spec_canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def spec_canonical_value(value: Any) -> Any:
    """Return the canonical JSON-serializable tree for a spec value."""
    if isinstance(value, BaseModel):
        return spec_canonical_value(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Mapping):
        return {
            str(key): spec_canonical_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, tuple):
        return [spec_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [spec_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return spec_canonical_value(value.value)
    if isinstance(value, datetime):
        return format_aware_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return canonical_spec_float(value)
    if value is None:
        return None
    raise TypeError(f"unsupported type for spec canonical JSON: {type(value)!r}")


def canonical_spec_float(value: float) -> dict[str, str]:
    """Encode one float as the tagged spec canonical form."""
    if math.isnan(value):
        text = "NaN"
    elif math.isinf(value):
        text = "Infinity" if value > 0 else "-Infinity"
    elif value == 0.0 and math.copysign(1.0, value) < 0:
        text = "-0"
    else:
        text = struct.pack(">d", value).hex()
    return {_SPEC_FLOAT_TAG: text}
