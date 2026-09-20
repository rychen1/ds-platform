"""Declarative experiment specifications and configuration hashing."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ds_platform.hashing import SHA256_HEX_PATTERN, sha256_hex
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.types import LOGICAL_KEY_PATTERN, freeze_json_value


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
    params: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


type FeatureKind = Literal["numeric", "categorical", "text", "passthrough"]


class FeatureSchema(_FrozenModel):
    """Hashable column-kind metadata. Not attached to ``FeatureTable`` cells."""

    columns: tuple[str, ...]
    kinds: tuple[FeatureKind, ...]

    @model_validator(mode="after")
    def _aligned_unique_columns(self) -> FeatureSchema:
        if len(self.columns) != len(self.kinds):
            raise ValueError("kinds length must match columns length")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")
        return self


class SplitSpec(_FrozenModel):
    """Train/validation/test assignment configuration.

    ``as_of`` drops entities with ``timestamp > as_of`` before splitting.
    It is not a temporal cut: use ``method="temporal"`` to put later
    remaining entities in test.

    ``test_size`` and ``validation_size`` are fractions of the kept entity
    set. When ``validation_size`` is set, holdout and temporal splits
    populate ``validation_ids``. K-fold does not support
    ``validation_size``.
    """

    method: Literal["holdout", "kfold", "temporal"]
    seed: int
    test_size: float | None = None
    validation_size: float | None = None
    n_splits: int | None = None
    stratify: bool = False
    as_of: date | None = None

    @model_validator(mode="after")
    def _validate_sizes(self) -> SplitSpec:
        if self.validation_size is None:
            return self
        if self.method == "kfold":
            raise ValueError("kfold split does not support validation_size")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if self.test_size is not None and self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")
        return self


class MetricSpec(_FrozenModel):
    name: str = Field(min_length=1)
    params: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


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
    """Encoder configuration. ``seed`` is hashed; adapters must apply it."""

    family: str = Field(min_length=1)
    params: Mapping[str, JsonValue] = Field(default_factory=dict)
    dim: int = Field(gt=0)
    seed: int

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


class ConditioningSpec(_FrozenModel):
    family: str = Field(min_length=1)
    params: Mapping[str, JsonValue] = Field(default_factory=dict)
    seed: int

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


class PerspectiveSpec(_FrozenModel):
    subject_field: str = Field(min_length=1)
    perspective_field: str = Field(min_length=1)
    params: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


class SequenceSpec(_FrozenModel):
    family: str = Field(min_length=1)
    params: Mapping[str, JsonValue] = Field(default_factory=dict)
    seed: int
    max_events: int | None = None

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


class GeometrySpec(_FrozenModel):
    metric: Literal["cosine", "l2"]
    k: int = Field(ge=1)
    family: str = Field(min_length=1)
    params: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _freeze_params(cls, params: Mapping[str, JsonValue]) -> Any:
        return freeze_json_value(dict(params))


def spec_config_hash(spec: BaseModel) -> str:
    """Return the SHA-256 hex digest of any frozen modeling specification."""
    return sha256_hex(spec_canonical_json_bytes(spec))


def experiment_config_hash(spec: ExperimentSpec) -> str:
    """Return the SHA-256 hex digest of the scientific experiment configuration.

    ``experiment_id`` is a human label and is excluded. Code, environment,
    and lockfiles belong on ``RunContext``, not in this hash.
    """
    payload = spec.model_dump(mode="python", exclude_none=True)
    payload.pop("experiment_id", None)
    return sha256_hex(spec_canonical_json_bytes(payload))
