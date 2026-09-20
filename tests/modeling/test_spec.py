"""ExperimentSpec models and experiment_config_hash (modeling Part 3)."""

from __future__ import annotations

import warnings
from datetime import date

import pytest
from pydantic import ValidationError

from ds_platform.hashing import sha256_hex
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.modeling.spec import (
    DatasetRef,
    EncodingSpec,
    ExperimentSpec,
    FeatureSchema,
    FeatureSpec,
    MetricSpec,
    ModelSpec,
    SplitSpec,
    TargetSpec,
    experiment_config_hash,
    spec_config_hash,
)
from import_boundary_util import assert_import_does_not_pull

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


def test_spec_config_hash_includes_experiment_id() -> None:
    spec = _experiment()
    assert spec_config_hash(spec) != experiment_config_hash(spec)


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
    payload = spec.model_dump(mode="python", exclude_none=True)
    payload.pop("experiment_id")
    expected = sha256_hex(spec_canonical_json_bytes(payload))
    assert experiment_config_hash(spec) == expected


def test_experiment_config_hash_ignores_experiment_id() -> None:
    left = _experiment(experiment_id="exp-a")
    right = _experiment(experiment_id="exp-b")
    assert experiment_config_hash(left) == experiment_config_hash(right)


def test_experiment_config_hash_encodes_float_params_deterministically() -> None:
    spec = _experiment(model=ModelSpec(family="baseline", params={"weight": 1.25}))
    encoded = spec_canonical_json_bytes(spec).decode("utf-8")
    assert '"$spec_float"' in encoded
    payload = spec.model_dump(mode="python", exclude_none=True)
    payload.pop("experiment_id")
    digest = sha256_hex(spec_canonical_json_bytes(payload))
    assert experiment_config_hash(spec) == digest


def test_model_spec_params_lists_dump_without_warning() -> None:
    spec = ModelSpec(family="baseline", params={"ks": [1, 3]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dumped = spec.model_dump(mode="python")
    assert dumped["params"]["ks"] == [1, 3]
    assert not [
        item for item in caught if "Pydantic serializer warnings" in str(item.message)
    ]


def test_experiment_config_hash_encodes_dates_as_iso() -> None:
    spec = _experiment(
        features=FeatureSpec(views=("structured",), as_of=date(2026, 3, 14))
    )
    encoded = spec_canonical_json_bytes(spec).decode("utf-8")
    assert "2026-03-14" in encoded


def test_spec_import_does_not_load_forbidden_modules() -> None:
    assert_import_does_not_pull("ds_platform.modeling.spec", _FORBIDDEN)


def test_feature_schema_hash_changes_when_kind_changes() -> None:
    first = FeatureSchema(columns=("borough", "lat"), kinds=("categorical", "numeric"))
    second = FeatureSchema(columns=("borough", "lat"), kinds=("passthrough", "numeric"))
    assert spec_config_hash(first) != spec_config_hash(second)
    assert spec_config_hash(first) == spec_config_hash(
        FeatureSchema(columns=("borough", "lat"), kinds=("categorical", "numeric"))
    )


def test_feature_schema_rejects_length_mismatch_and_duplicates() -> None:
    with pytest.raises(ValidationError, match="kinds length"):
        FeatureSchema(columns=("a", "b"), kinds=("numeric",))
    with pytest.raises(ValidationError, match="columns must be unique"):
        FeatureSchema(columns=("a", "a"), kinds=("numeric", "categorical"))


def test_split_spec_rejects_validation_size_out_of_range() -> None:
    with pytest.raises(ValidationError, match="validation_size must be between"):
        SplitSpec(method="holdout", seed=1, test_size=0.2, validation_size=1.0)


def test_split_spec_rejects_sizes_that_leave_no_train() -> None:
    with pytest.raises(ValidationError, match="must be less than 1"):
        SplitSpec(method="holdout", seed=1, test_size=0.5, validation_size=0.5)


def test_experiment_config_hash_includes_validation_size() -> None:
    base = _experiment()
    with_validation = _experiment(
        split=SplitSpec(method="holdout", seed=42, test_size=0.2, validation_size=0.2)
    )
    assert experiment_config_hash(base) != experiment_config_hash(with_validation)
