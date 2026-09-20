"""End-to-end lifecycle tests for vendor adapters with run_experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from helpers import _FAST_PARAMS, _SOURCE

from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.integrations.catboost import (
    MEDIA_TYPE as CATBOOST_MEDIA_TYPE,
)
from ds_platform.integrations.catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)
from ds_platform.integrations.lightgbm import (
    MEDIA_TYPE as LIGHTGBM_MEDIA_TYPE,
)
from ds_platform.integrations.lightgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
)
from ds_platform.integrations.xgboost import (
    MEDIA_TYPE as XGBOOST_MEDIA_TYPE,
)
from ds_platform.integrations.xgboost import (
    XGBoostClassifier,
    XGBoostRegressor,
)
from ds_platform.modeling import (
    DatasetRef,
    ExperimentSpec,
    FeatureSpec,
    MetricSpec,
    ModelSpec,
    Scalar,
    SplitSpec,
    TargetSpec,
    experiment_config_hash,
    run_experiment,
)
from ds_platform.modeling.features import FeatureTable, select_columns

_CREATED = datetime(2026, 9, 20, 1, 0, tzinfo=UTC)


class _TabularView:
    name = "tabular"

    def __init__(self, *, task: str) -> None:
        self._task = task

    def transform(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        as_of: object = None,
    ) -> FeatureTable:
        del as_of
        entity_ids = tuple(str(row["id"]) for row in rows)
        if self._task == "classification":
            values = cast(
                tuple[tuple[Scalar, ...], ...],
                tuple(
                    (row["x1"], row["x2"], row["color"], row["label"]) for row in rows
                ),
            )
            columns = ("x1", "x2", "color", "label")
        else:
            values = cast(
                tuple[tuple[Scalar, ...], ...],
                tuple((row["x1"], row["x2"], row["target"]) for row in rows),
            )
            columns = ("x1", "x2", "target")
        return FeatureTable(
            entity_ids=entity_ids,
            columns=columns,
            values=values,
            source_payload_ids=(_SOURCE,) * len(entity_ids),
        )


def _classification_rows() -> list[dict[str, object]]:
    return [
        {"id": "e1", "x1": 1.0, "x2": 0.0, "color": "red", "label": "A"},
        {"id": "e2", "x1": 2.0, "x2": 1.0, "color": "blue", "label": "A"},
        {"id": "e3", "x1": 3.0, "x2": 0.5, "color": "red", "label": "B"},
        {"id": "e4", "x1": 4.0, "x2": 1.5, "color": "blue", "label": "B"},
        {"id": "e5", "x1": 5.0, "x2": 0.2, "color": "red", "label": "A"},
        {"id": "e6", "x1": 6.0, "x2": 1.1, "color": "blue", "label": "B"},
    ]


def _regression_rows() -> list[dict[str, object]]:
    return [
        {"id": "e1", "x1": 1.0, "x2": 0.0, "target": 1.0},
        {"id": "e2", "x1": 2.0, "x2": 1.0, "target": 2.5},
        {"id": "e3", "x1": 3.0, "x2": 0.5, "target": 3.0},
        {"id": "e4", "x1": 4.0, "x2": 1.5, "target": 4.5},
        {"id": "e5", "x1": 5.0, "x2": 0.2, "target": 5.0},
        {"id": "e6", "x1": 6.0, "x2": 1.1, "target": 6.5},
    ]


def _run() -> RunContext:
    return RunContext(
        run_id="integration-run",
        project="proj",
        started_at=_CREATED,
        environment=Environment.LOCAL,
    )


def _classification_spec(family: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=f"{family}-clf",
        dataset=DatasetRef(payload_id=_SOURCE),
        features=FeatureSpec(views=("tabular",)),
        target=TargetSpec(column="label", task="classification"),
        model=ModelSpec(family=family, params=_FAST_PARAMS),
        split=SplitSpec(method="holdout", seed=42, test_size=1 / 3),
        metrics=(MetricSpec(name="accuracy"), MetricSpec(name="macro_f1")),
        seed=42,
    )


def _regression_spec(family: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=f"{family}-reg",
        dataset=DatasetRef(payload_id=_SOURCE),
        features=FeatureSpec(views=("tabular",)),
        target=TargetSpec(column="target", task="regression"),
        model=ModelSpec(family=family, params=_FAST_PARAMS),
        split=SplitSpec(method="holdout", seed=42, test_size=1 / 3),
        metrics=(MetricSpec(name="rmse"), MetricSpec(name="mae")),
        seed=42,
    )


def _assert_lifecycle(
    tmp_path,
    *,
    spec: ExperimentSpec,
    rows: list[dict[str, object]],
    view: _TabularView,
    adapter_factory: Callable[[], object],
    serialize: Callable[[object], bytes],
    deserialize: Callable[[bytes], object],
    media_type: str,
    reload_smoke: Callable[[object, _TabularView, list[dict[str, object]], str], None],
) -> None:
    store = LocalStore(tmp_path)
    adapter = adapter_factory()
    result = run_experiment(
        spec,
        rows=rows,
        feature_views={"tabular": view},
        adapter=cast(Any, adapter),
        store=store,
        run=_run(),
        serialize_model=serialize,
        model_media_type=media_type,
    )
    assert result.config_hash == experiment_config_hash(spec)
    assert store.exists(result.model_payload_id)
    assert store.exists(result.prediction_payload_id)
    assert store.exists(result.evaluation_payload_id)

    model_bytes = store.get(result.model_payload_id)
    reloaded = deserialize(model_bytes)
    reload_smoke(reloaded, view, rows, spec.target.column)

    prediction_lines = (
        store.get(result.prediction_payload_id).decode("utf-8").strip().split("\n")
    )
    assert prediction_lines
    evaluation_payload = json.loads(
        store.get(result.evaluation_payload_id).decode("utf-8")
    )
    assert evaluation_payload["metrics"]
    assert set(evaluation_payload["metrics"]) == set(result.report.metrics)


@pytest.mark.parametrize(
    ("family", "adapter_factory", "serialize", "deserialize", "media_type"),
    [
        (
            "xgboost",
            lambda: XGBoostClassifier(
                params=_FAST_PARAMS,
                random_state=42,
                categorical_columns=("color",),
            ),
            lambda adapter: adapter.serialize(),
            XGBoostClassifier.deserialize,
            XGBOOST_MEDIA_TYPE,
        ),
        (
            "lightgbm",
            lambda: LightGBMClassifier(
                params=_FAST_PARAMS,
                random_state=42,
                categorical_columns=("color",),
            ),
            lambda adapter: adapter.serialize(),
            LightGBMClassifier.deserialize,
            LIGHTGBM_MEDIA_TYPE,
        ),
        (
            "catboost",
            lambda: CatBoostClassifier(
                params=_FAST_PARAMS,
                random_state=42,
                cat_features=("color",),
            ),
            lambda adapter: adapter.serialize(),
            CatBoostClassifier.deserialize,
            CATBOOST_MEDIA_TYPE,
        ),
    ],
)
def test_classification_lifecycle(
    tmp_path,
    family: str,
    adapter_factory: Callable[[], object],
    serialize: Callable[[object], bytes],
    deserialize: Callable[[bytes], object],
    media_type: str,
) -> None:
    def reload_smoke(
        reloaded,
        view: _TabularView,
        rows: list[dict[str, object]],
        target_column: str,
    ) -> None:
        table = view.transform(rows)
        feature_columns = tuple(
            column for column in table.columns if column != target_column
        )
        features = select_columns(table, feature_columns)
        predictions = reloaded.predict(features)
        assert len(predictions) == len(features.entity_ids)

    _assert_lifecycle(
        tmp_path,
        spec=_classification_spec(family),
        rows=_classification_rows(),
        view=_TabularView(task="classification"),
        adapter_factory=adapter_factory,
        serialize=serialize,
        deserialize=deserialize,
        media_type=media_type,
        reload_smoke=reload_smoke,
    )


@pytest.mark.parametrize(
    ("family", "adapter_factory", "serialize", "deserialize", "media_type"),
    [
        (
            "xgboost",
            lambda: XGBoostRegressor(params=_FAST_PARAMS, random_state=42),
            lambda adapter: adapter.serialize(),
            XGBoostRegressor.deserialize,
            XGBOOST_MEDIA_TYPE,
        ),
        (
            "lightgbm",
            lambda: LightGBMRegressor(params=_FAST_PARAMS, random_state=42),
            lambda adapter: adapter.serialize(),
            LightGBMRegressor.deserialize,
            LIGHTGBM_MEDIA_TYPE,
        ),
        (
            "catboost",
            lambda: CatBoostRegressor(params=_FAST_PARAMS, random_state=42),
            lambda adapter: adapter.serialize(),
            CatBoostRegressor.deserialize,
            CATBOOST_MEDIA_TYPE,
        ),
    ],
)
def test_regression_lifecycle(
    tmp_path,
    family: str,
    adapter_factory: Callable[[], object],
    serialize: Callable[[object], bytes],
    deserialize: Callable[[bytes], object],
    media_type: str,
) -> None:
    def reload_smoke(
        reloaded,
        view: _TabularView,
        rows: list[dict[str, object]],
        target_column: str,
    ) -> None:
        table = view.transform(rows)
        feature_columns = tuple(
            column for column in table.columns if column != target_column
        )
        features = select_columns(table, feature_columns)
        predictions = reloaded.predict(features)
        assert len(predictions) == len(features.entity_ids)

    _assert_lifecycle(
        tmp_path,
        spec=_regression_spec(family),
        rows=_regression_rows(),
        view=_TabularView(task="regression"),
        adapter_factory=adapter_factory,
        serialize=serialize,
        deserialize=deserialize,
        media_type=media_type,
        reload_smoke=reload_smoke,
    )


def test_model_artifact_kind_is_model(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = _classification_spec("xgboost")
    result = run_experiment(
        spec,
        rows=_classification_rows(),
        feature_views={"tabular": _TabularView(task="classification")},
        adapter=XGBoostClassifier(
            params=_FAST_PARAMS,
            random_state=42,
            categorical_columns=("color",),
        ),
        store=store,
        run=_run(),
        serialize_model=lambda adapter: cast(Any, adapter).serialize(),
        model_media_type=XGBOOST_MEDIA_TYPE,
    )
    for path in tmp_path.rglob("*"):
        if not path.is_file() or len(path.name) != 64:
            continue
        try:
            envelope = json.loads(path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if envelope.get("payload_id") == result.model_payload_id:
            assert envelope["kind"] == ArtifactKind.MODEL
            return
    raise AssertionError("model envelope not found")
