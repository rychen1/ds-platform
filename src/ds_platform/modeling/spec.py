"""Declarative experiment specifications and configuration hashing."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ds_platform.hashing import SHA256_HEX_PATTERN, sha256_hex
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.types import LOGICAL_KEY_PATTERN


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class DatasetRef(_FrozenModel):
    payload_id: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    logical_key: str | None = Field(default=None, pattern=LOGICAL_KEY_PATTERN)

    @model_validator(mode="after")
    def _require_reference(self) -> DatasetRef:
        if self.payload_id is None and self.logical_key is None:
            raise ValueError("payload_id or logical_key is required")
        return self


class FeatureSpec(_FrozenModel):
    views: tuple[str, ...]
    columns: tuple[str, ...] | None = None
    as_of: date | None = None

    @field_validator("views")
    @classmethod
    def _views_non_empty(cls, views: tuple[str, ...]) -> tuple[str, ...]:
        if not views:
            raise ValueError("views must contain at least one entry")
        return views


class TargetSpec(_FrozenModel):
    column: str = Field(min_length=1)
    task: Literal["classification", "regression"]


class ModelSpec(_FrozenModel):
    family: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)


class SplitSpec(_FrozenModel):
    method: Literal["holdout", "kfold"]
    seed: int
    test_size: float | None = None
    n_splits: int | None = None
    stratify: bool = False
    as_of: date | None = None


class MetricSpec(_FrozenModel):
    name: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)


class ExperimentSpec(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    dataset: DatasetRef
    features: FeatureSpec
    target: TargetSpec
    model: ModelSpec
    split: SplitSpec
    metrics: tuple[MetricSpec, ...]
    seed: int

    @field_validator("metrics")
    @classmethod
    def _metrics_non_empty(
        cls,
        metrics: tuple[MetricSpec, ...],
    ) -> tuple[MetricSpec, ...]:
        if not metrics:
            raise ValueError("metrics must contain at least one entry")
        return metrics

    @model_validator(mode="after")
    def _seed_matches_split(self) -> ExperimentSpec:
        if self.seed != self.split.seed:
            raise ValueError("seed must match split.seed")
        return self


class EncodingSpec(_FrozenModel):
    family: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    dim: int = Field(gt=0)
    seed: int


class ConditioningSpec(_FrozenModel):
    family: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    seed: int


class PerspectiveSpec(_FrozenModel):
    subject_field: str = Field(min_length=1)
    perspective_field: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)


class SequenceSpec(_FrozenModel):
    family: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    seed: int
    max_events: int | None = None


class GeometrySpec(_FrozenModel):
    metric: Literal["cosine", "l2"]
    k: int = Field(ge=1)
    family: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)


def spec_config_hash(spec: BaseModel) -> str:
    """Return the SHA-256 hex digest of any frozen modeling specification."""
    return sha256_hex(spec_canonical_json_bytes(spec))


def experiment_config_hash(spec: ExperimentSpec) -> str:
    """Return the SHA-256 hex digest of the canonical experiment configuration."""
    return spec_config_hash(spec)
