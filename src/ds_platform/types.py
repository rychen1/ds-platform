"""Core semantic types for artifacts, provenance, and attribution."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, SupportsIndex

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ds_platform.hashing import SHA256_HEX_PATTERN

LOGICAL_KEY_PATTERN = (
    r"^[a-z][a-z0-9-]{0,31}:"
    r"[a-z][a-z0-9._/-]{0,127}"
    r"(?::[a-z0-9][a-z0-9._-]{0,63})?$"
)
MEDIA_TYPE_PATTERN = r"^[a-z][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$"

Sha256Hex = Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
LogicalKey = Annotated[str, Field(pattern=LOGICAL_KEY_PATTERN)]
MediaType = Annotated[str, Field(pattern=MEDIA_TYPE_PATTERN)]


class ArtifactKind(StrEnum):
    RAW = "raw"
    DATASET = "dataset"
    DOCUMENT = "document"
    MODEL = "model"
    PROMPT = "prompt"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    QUALITY = "quality"
    CLAIM_SET = "claim_set"
    REVIEW = "review"
    RUN = "run"
    INDEX = "index"
    REPRESENTATION = "representation"


class ClaimLayer(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    INFERRED = "inferred"
    ASSERTED = "asserted"


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    CLOUD = "cloud"


class RelationType(StrEnum):
    QUALITY_FOR = "quality_for"
    EVALUATION_OF = "evaluation_of"
    CLAIMS_ABOUT = "claims_about"
    REVIEWS = "reviews"
    CHAMPION_OF = "champion_of"


class _CoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContractRef(_CoreModel):
    uri: str = Field(min_length=1)
    schema_hash: Sha256Hex


class RelatedRef(_CoreModel):
    rel: RelationType
    payload_id: Sha256Hex
    record_id: Sha256Hex | None = None


class Citation(_CoreModel):
    payload_id: Sha256Hex
    record_id: Sha256Hex | None = None
    pointer: str | None = None


class Evidence(_CoreModel):
    citation: Citation
    excerpt: str = Field(min_length=1)
    method: str | None = None


class Claim(_CoreModel):
    statement: str = Field(min_length=1)
    layer: ClaimLayer
    confidence_policy: str | None = None
    evidence: Sequence[Evidence] | None = None
    citations: Sequence[Citation] | None = None
    subject_payload_id: Sha256Hex | None = None

    @field_validator("evidence", "citations", mode="before")
    @classmethod
    def _tupleize_optional(cls, value: object) -> object:
        if value is None:
            return None
        return tuple(value)  # type: ignore[call-overload]


class CodeRef(_CoreModel):
    git_sha: str = Field(min_length=1)
    dirty: bool


class ExternalRunRef(_CoreModel):
    """One identifier from an external orchestrator, registry, or tracer."""

    system: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


class RunContext(_CoreModel):
    run_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    started_at: datetime
    environment: Environment
    code_ref: CodeRef | None = None
    config_hash: Sha256Hex | None = None
    completed_at: datetime | None = None
    external_run_ids: tuple[ExternalRunRef, ...] = ()

    @field_validator("started_at", "completed_at")
    @classmethod
    def _aware_datetime(cls, value: datetime | None) -> datetime | None:
        return _require_aware(value)


class InferenceRecord(_CoreModel):
    model_payload_id: Sha256Hex | None = None
    prompt_payload_id: Sha256Hex | None = None
    input_payload_id: Sha256Hex
    output_payload_id: Sha256Hex | None = None
    output_inline: str | None = None
    provider: str = Field(min_length=1)
    params_hash: str = Field(min_length=1)
    token_count: int | str | None = None
    latency_ms: int | str | None = None
    cost: int | str | None = None
    run_id: str = Field(min_length=1)
    otel_span_id: str | None = None

    @model_validator(mode="after")
    def _require_subject(self) -> InferenceRecord:
        if self.model_payload_id is None and self.prompt_payload_id is None:
            raise ValueError("model_payload_id or prompt_payload_id is required")
        return self


class ArtifactRecord(_CoreModel):
    """Slim sidecar for one payload.

    ``inputs`` are payload ids consumed by the producing run.
    ``derived_from`` is reserved and unused in v0; do not give it a second
    meaning. Typed ``related`` refs are the closed ``RelationType`` set.

    ``logical_key`` is a non-unique project label, not an identity and not
    a catalog lookup. The same key may label many payloads.

    v0 artifacts are a single byte sequence. Multi-file layouts must be
    packed by the caller before hashing.
    """

    schema_version: Literal[0] = 0
    payload_id: Sha256Hex
    media_type: MediaType
    kind: ArtifactKind
    produced_by: RunContext
    created_at: datetime
    name: str | None = None
    logical_key: LogicalKey | None = None
    contract_ref: ContractRef | None = None
    inherited_policy_refs: Sequence[Sha256Hex] = ()
    inputs: Sequence[Sha256Hex] = ()
    derived_from: Sequence[Sha256Hex] = ()
    related: Sequence[RelatedRef] = ()

    @field_validator(
        "inherited_policy_refs",
        "inputs",
        "derived_from",
        "related",
        mode="before",
    )
    @classmethod
    def _tupleize(cls, value: object) -> object:
        if value is None:
            return ()
        return tuple(value)  # type: ignore[call-overload]

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        aware = _require_aware(value)
        assert aware is not None
        return aware

    @model_validator(mode="after")
    def _validate_lineage(self) -> ArtifactRecord:
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("inputs must not contain duplicates")
        for ref in self.related:
            if ref.payload_id == self.payload_id:
                raise ValueError(
                    "related must not reference this record's payload_id"
                )
        return self


def new_run_id() -> str:
    """Issue a platform run id (UUID4 hex, no dashes)."""
    return uuid.uuid4().hex


class FrozenJSON(dict[str, Any]):
    """JSON object that rejects in-place mutation."""

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("frozen mapping")

    def __delitem__(self, key: str) -> None:
        raise TypeError("frozen mapping")

    def clear(self) -> None:
        raise TypeError("frozen mapping")

    def pop(self, key: str, default: Any = None) -> Any:
        raise TypeError("frozen mapping")

    def popitem(self) -> tuple[str, Any]:
        raise TypeError("frozen mapping")

    def setdefault(self, key: str, default: Any = None) -> Any:
        raise TypeError("frozen mapping")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen mapping")


class FrozenList(list[Any]):
    """JSON array that rejects in-place mutation."""

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("frozen sequence")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("frozen sequence")

    def append(self, value: Any) -> None:
        raise TypeError("frozen sequence")

    def extend(self, values: Any) -> None:
        raise TypeError("frozen sequence")

    def insert(self, index: SupportsIndex, value: Any) -> None:
        raise TypeError("frozen sequence")

    def pop(self, index: SupportsIndex = -1) -> Any:
        raise TypeError("frozen sequence")

    def remove(self, value: Any) -> None:
        raise TypeError("frozen sequence")

    def clear(self) -> None:
        raise TypeError("frozen sequence")

    def reverse(self) -> None:
        raise TypeError("frozen sequence")

    def sort(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen sequence")

    def __iadd__(self, other: Any) -> FrozenList:
        raise TypeError("frozen sequence")

    def __imul__(self, other: Any) -> FrozenList:
        raise TypeError("frozen sequence")


def freeze_json_value(value: Any) -> Any:
    """Return a deep-immutable JSON-shaped value."""
    if isinstance(value, Mapping):
        return FrozenJSON(
            {str(key): freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return FrozenList(freeze_json_value(item) for item in value)
    return value


def artifact_record_json_schema() -> dict[str, Any]:
    schema = ArtifactRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ArtifactRecord"
    return schema


def attribution_json_schema() -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for model in (Citation, Evidence, Claim, InferenceRecord):
        raw = model.model_json_schema(ref_template="#/$defs/{model}")
        nested = raw.pop("$defs", {})
        defs.update(nested)
        defs[model.__name__] = raw
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ds-platform attribution types",
        "$defs": defs,
    }


def dump_json_schema(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def _require_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value
