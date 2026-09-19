"""Core semantic types for artifacts, provenance, and attribution."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    model_config = ConfigDict(extra="forbid")


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
    evidence: list[Evidence] | None = None
    citations: list[Citation] | None = None
    subject_payload_id: Sha256Hex | None = None


class CodeRef(_CoreModel):
    git_sha: str = Field(min_length=1)
    dirty: bool


class ExternalRunIds(_CoreModel):
    dagster: str | None = None
    mlflow: str | None = None
    sqlmesh: str | None = None
    otel_trace: str | None = None


class RunContext(_CoreModel):
    run_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    started_at: datetime
    environment: Environment
    code_ref: CodeRef | None = None
    config_hash: Sha256Hex | None = None
    completed_at: datetime | None = None
    external_run_ids: ExternalRunIds | None = None


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
    payload_id: Sha256Hex
    media_type: MediaType
    kind: ArtifactKind
    produced_by: RunContext
    created_at: datetime
    name: str | None = None
    logical_key: LogicalKey | None = None
    contract_ref: ContractRef | None = None
    inherited_policy_refs: list[str] = Field(default_factory=list)
    inputs: list[Sha256Hex] = Field(default_factory=list)
    derived_from: list[Sha256Hex] = Field(default_factory=list)
    related: list[RelatedRef] = Field(default_factory=list)


def new_run_id() -> str:
    """Issue a platform run id (UUID4 hex, no dashes)."""
    return uuid.uuid4().hex


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
