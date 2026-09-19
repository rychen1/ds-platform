"""Import boundary and private spec canonicalization (modeling Part 1)."""

from __future__ import annotations

import struct
import sys
from datetime import date

import pytest

from ds_platform.hashing import canonical_json_bytes
from ds_platform.modeling._spec_json import (
    canonical_spec_float,
    spec_canonical_json_bytes,
    spec_canonical_value,
)

_FORBIDDEN_VENDORS = {
    "boto3",
    "catboost",
    "dagster",
    "dask",
    "duckdb",
    "lightgbm",
    "mlflow",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "pyspark",
    "ray",
    "sklearn",
    "sqlmesh",
    "streamlit",
    "torch",
    "xgboost",
}

_FORBIDDEN_CONSUMERS = {
    "restaurant_intelligence",
    "board_game_analysis",
}


_FORBIDDEN_EXPORTS = {
    "Action",
    "BeliefState",
    "Board",
    "Deck",
    "GameState",
    "GameTree",
    "Hand",
    "HasPredictProba",
    "InformationSet",
    "Mechanic",
    "Predictor",
    "RetrievalSpec",
    "RetrievedItem",
    "Retriever",
    "Trajectory",
    "evidence_payload_ids",
    "filter_temporal",
    "task_capability",
}


def test_import_ds_platform_modeling_succeeds() -> None:
    import ds_platform.modeling  # noqa: F401


def test_modeling_public_exports_are_importable() -> None:
    import ds_platform.modeling as modeling

    for name in modeling.__all__:
        assert hasattr(modeling, name), f"missing export: {name!r}"


def test_modeling_public_exports_exclude_deferred_names() -> None:
    import ds_platform.modeling as modeling

    assert _FORBIDDEN_EXPORTS.isdisjoint(set(modeling.__all__))


def test_modeling_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling  # noqa: F401

    loaded = _FORBIDDEN_VENDORS | _FORBIDDEN_CONSUMERS
    assert loaded.intersection(sys.modules) == set()


def test_spec_canonical_json_is_deterministic() -> None:
    payload = {"a": 1, "b": {"c": 2}, "d": date(2026, 3, 14)}
    first = spec_canonical_json_bytes(payload)
    second = spec_canonical_json_bytes({"b": {"c": 2}, "d": date(2026, 3, 14), "a": 1})
    assert first == second


def test_spec_canonical_date_uses_iso_format() -> None:
    encoded = spec_canonical_value({"cutoff": date(2026, 3, 14)})
    assert encoded == {"cutoff": "2026-03-14"}


@pytest.mark.parametrize(
    ("value", "expected_text"),
    [
        (1.25, struct.pack(">d", 1.25).hex()),
        (-2.0, struct.pack(">d", -2.0).hex()),
        (0.0, struct.pack(">d", 0.0).hex()),
    ],
)
def test_spec_canonical_float_positive_and_negative(
    value: float,
    expected_text: str,
) -> None:
    tagged = canonical_spec_float(value)
    assert tagged == {"$spec_float": expected_text}
    assert spec_canonical_value({"weight": value}) == {"weight": tagged}


def test_spec_canonical_float_special_values() -> None:
    assert canonical_spec_float(float("nan")) == {"$spec_float": "NaN"}
    assert canonical_spec_float(float("inf")) == {"$spec_float": "Infinity"}
    assert canonical_spec_float(float("-inf")) == {"$spec_float": "-Infinity"}
    assert canonical_spec_float(-0.0) == {"$spec_float": "-0"}


def test_artifact_canonical_json_still_rejects_floats() -> None:
    with pytest.raises(TypeError, match="floats are not allowed"):
        canonical_json_bytes({"weight": 1.25})


def test_artifact_canonical_json_unchanged_for_integers() -> None:
    assert canonical_json_bytes({"a": 1, "b": 2}) == b'{"a":1,"b":2}'
