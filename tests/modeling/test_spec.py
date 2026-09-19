"""ExperimentSpec models and experiment_config_hash (modeling Part 3)."""

from __future__ import annotations

import sys
from datetime import date

import pytest
from pydantic import ValidationError

from ds_platform.hashing import sha256_hex
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.modeling.spec import (
    DatasetRef,
    EncodingSpec,
    ExperimentSpec,
    FeatureSpec,
    MetricSpec,
    ModelSpec,
    SplitSpec,
    TargetSpec,
    experiment_config_hash,
    spec_config_hash,
)

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_DATASET_ID = "a" * 64
_LOGICAL_KEY = "proj:dataset:v0"


def _dataset_ref(**overrides: str | None) -> DatasetRef:
    payload: dict[str, str | None] = {"logical_key": _LOGICAL_KEY}
    payload.update(overrides)
    return DatasetRef(**payload)


def _experiment(**overrides: object) -> ExperimentSpec:
    defaults = {
        "experiment_id": "exp-001",
        "dataset": _dataset_ref(),
        "features": FeatureSpec(views=("structured",), as_of=date(2026, 3, 14)),
        "target": TargetSpec(column="label", task="classification"),
        "model": ModelSpec(family="baseline", params={"weight": 1.25}),
        "split": SplitSpec(method="holdout", seed=42, test_size=0.2),
        "metrics": (MetricSpec(name="accuracy"),),
        "seed": 42,
    }
    defaults.update(overrides)
    return ExperimentSpec(**defaults)  # type: ignore[arg-type]


def test_experiment_spec_valid_construction() -> None:
    spec = _experiment()
    assert spec.experiment_id == "exp-001"
    assert spec.dataset.logical_key == _LOGICAL_KEY
    assert spec.features.views == ("structured",)
    assert spec.target.task == "classification"


def test_dataset_ref_requires_payload_or_logical_key() -> None:
    with pytest.raises(ValidationError, match="payload_id or logical_key is required"):
        DatasetRef()


def test_dataset_ref_accepts_payload_id() -> None:
    ref = DatasetRef(payload_id=_DATASET_ID)
    assert ref.payload_id == _DATASET_ID


def test_dataset_ref_rejects_invalid_payload_id() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(payload_id="not-a-hash")


def test_dataset_ref_rejects_invalid_logical_key() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(logical_key="INVALID")


def test_feature_spec_rejects_empty_views() -> None:
    with pytest.raises(ValidationError, match="views must contain at least one"):
        FeatureSpec(views=())


def test_experiment_spec_rejects_empty_metrics() -> None:
    with pytest.raises(ValidationError, match="metrics must contain at least one"):
        _experiment(metrics=())


def test_experiment_spec_requires_matching_seed() -> None:
    with pytest.raises(ValidationError, match="seed must match split.seed"):
        _experiment(seed=1)


def test_experiment_spec_rejects_extra_fields() -> None:
    payload = _experiment().model_dump(mode="python")
    payload["extra_field"] = "not-allowed"
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_model_spec_rejects_non_json_params() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({"family": "baseline", "params": {"bad": object()}})


def test_experiment_spec_has_no_retrieval_field() -> None:
    assert "retrieval" not in ExperimentSpec.model_fields


def test_spec_config_hash_matches_experiment_config_hash() -> None:
    spec = _experiment()
    assert spec_config_hash(spec) == experiment_config_hash(spec)


def test_encoding_spec_hash_is_stable() -> None:
    spec = EncodingSpec(family="mean", dim=4, seed=1)
    assert spec_config_hash(spec) == spec_config_hash(
        EncodingSpec(family="mean", dim=4, seed=1)
    )


def test_experiment_config_hash_is_stable() -> None:
    first = _experiment()
    second = _experiment()
    assert experiment_config_hash(first) == experiment_config_hash(second)
    assert len(experiment_config_hash(first)) == 64


def test_experiment_config_hash_changes_when_spec_changes() -> None:
    base = _experiment()
    changed = _experiment(model=ModelSpec(family="other", params={"weight": 1.25}))
    assert experiment_config_hash(base) != experiment_config_hash(changed)


def test_experiment_config_hash_uses_spec_canonical_json_bytes() -> None:
    spec = _experiment()
    expected = sha256_hex(spec_canonical_json_bytes(spec))
    assert experiment_config_hash(spec) == expected


def test_experiment_config_hash_encodes_float_params_deterministically() -> None:
    spec = _experiment(model=ModelSpec(family="baseline", params={"weight": 1.25}))
    encoded = spec_canonical_json_bytes(spec).decode("utf-8")
    assert '"$spec_float"' in encoded
    assert experiment_config_hash(spec) == sha256_hex(spec_canonical_json_bytes(spec))


def test_experiment_config_hash_encodes_dates_as_iso() -> None:
    spec = _experiment(
        features=FeatureSpec(views=("structured",), as_of=date(2026, 3, 14))
    )
    encoded = spec_canonical_json_bytes(spec).decode("utf-8")
    assert "2026-03-14" in encoded


def test_spec_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.spec  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
