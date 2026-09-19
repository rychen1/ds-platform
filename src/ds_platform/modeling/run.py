"""Thin holdout experiment orchestration for modeling v1."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict

from ds_platform.modeling.capabilities import Classifier, Regressor
from ds_platform.modeling.evaluate import EvaluationReport, evaluate
from ds_platform.modeling.features import (
    FeatureTable,
    FeatureView,
    Scalar,
    align_feature_tables,
    extract_column,
    select_columns,
)
from ds_platform.modeling.records import (
    PredictionRow,
    prediction_payload_bytes,
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    put_prediction_artifact,
)
from ds_platform.modeling.spec import ExperimentSpec, experiment_config_hash
from ds_platform.modeling.split import apply_split, split_entities
from ds_platform.store import Store
from ds_platform.types import RunContext


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentResult(_FrozenModel):
    run_id: str
    config_hash: str
    feature_payload_id: str
    model_payload_id: str
    prediction_payload_id: str
    evaluation_payload_id: str
    report: EvaluationReport


def run_experiment(
    spec: ExperimentSpec,
    *,
    rows: Sequence[Mapping[str, object]],
    feature_views: Mapping[str, FeatureView],
    adapter: Classifier | Regressor,
    store: Store,
    run: RunContext,
    serialize_model: Callable[[object], bytes],
    model_media_type: str = "application/octet-stream",
) -> ExperimentResult:
    """Run a v1 holdout supervised experiment and persist modeling artifacts.

    This is optional v1 holdout orchestration for scalar supervised tasks.
    Callers with other workloads may use the underlying feature, split,
    evaluate, and record helpers directly.
    """
    if spec.split.method != "holdout":
        raise ValueError("run_experiment v1 requires split.method == 'holdout'")

    config_hash = experiment_config_hash(spec)
    run_with_hash = run.model_copy(update={"config_hash": config_hash})

    aligned = _materialize_features(spec, rows, feature_views)
    labels = extract_column(aligned, spec.target.column)
    features = _select_feature_columns(aligned, spec)
    labels_by_entity = dict(zip(aligned.entity_ids, labels, strict=True))

    split_labels: Sequence[object] | None = labels if spec.split.stratify else None
    assignment = split_entities(
        aligned.entity_ids,
        spec.split,
        labels=split_labels,
    )[0]
    train_features, _, test_features = apply_split(features, assignment)

    _fit_adapter(adapter, spec, train_features, labels_by_entity)
    test_predictions = adapter.predict(test_features)

    fitted_model = serialize_model(adapter)
    feature_inputs = _dataset_inputs(spec)
    feature_payload_id, _feature_record_id = put_feature_dataset(
        store,
        _feature_table_bytes(features),
        run=run_with_hash,
        inputs=feature_inputs,
        media_type="application/json",
    )
    model_payload_id, _model_record_id = put_model_artifact(
        store,
        fitted_model,
        run=run_with_hash,
        inputs=[feature_payload_id, *feature_inputs],
        media_type=model_media_type,
    )

    test_labels = _labels_for_entities(test_features.entity_ids, labels_by_entity)
    prediction_rows = _prediction_rows(
        test_features.entity_ids,
        y_true=test_labels,
        y_pred=test_predictions,
        y_proba=_maybe_predict_proba(adapter, test_features),
    )
    prediction_bytes = prediction_payload_bytes(prediction_rows)
    prediction_payload_id, _prediction_record_id = put_prediction_artifact(
        store,
        prediction_bytes,
        run=run_with_hash,
        inputs=[model_payload_id, feature_payload_id],
        media_type="application/jsonl",
    )

    report = evaluate(
        test_labels,
        list(test_predictions),
        spec.metrics,
        y_proba=_maybe_predict_proba(adapter, test_features),
    )
    evaluation_payload_id, _evaluation_record_id = put_evaluation_artifact(
        store,
        report,
        run=run_with_hash,
        subject_payload_id=prediction_payload_id,
        inputs=[prediction_payload_id, feature_payload_id],
    )

    return ExperimentResult(
        run_id=run_with_hash.run_id,
        config_hash=config_hash,
        feature_payload_id=feature_payload_id,
        model_payload_id=model_payload_id,
        prediction_payload_id=prediction_payload_id,
        evaluation_payload_id=evaluation_payload_id,
        report=report,
    )


def _materialize_features(
    spec: ExperimentSpec,
    rows: Sequence[Mapping[str, object]],
    feature_views: Mapping[str, FeatureView],
) -> FeatureTable:
    tables: list[FeatureTable] = []
    for view_name in spec.features.views:
        try:
            view = feature_views[view_name]
        except KeyError as exc:
            raise KeyError(f"feature view not found: {view_name!r}") from exc
        if view.name != view_name:
            raise ValueError(
                "feature view name mismatch: "
                f"key {view_name!r} != view.name {view.name!r}"
            )
        tables.append(view.transform(rows, as_of=spec.features.as_of))
    return align_feature_tables(tables)


def _select_feature_columns(
    aligned: FeatureTable,
    spec: ExperimentSpec,
) -> FeatureTable:
    target_column = spec.target.column
    if spec.features.columns is not None:
        if target_column in spec.features.columns:
            raise ValueError("feature allowlist must not include target column")
        return select_columns(aligned, spec.features.columns)
    feature_columns = tuple(
        column for column in aligned.columns if column != target_column
    )
    return select_columns(aligned, feature_columns)


def _fit_adapter(
    adapter: Classifier | Regressor,
    spec: ExperimentSpec,
    train_features: FeatureTable,
    labels_by_entity: Mapping[str, Scalar],
) -> None:
    train_labels = _labels_for_entities(train_features.entity_ids, labels_by_entity)
    if spec.target.task == "classification":
        _require_adapter_task(adapter, task="classification")
        cast(Classifier, adapter).fit(
            train_features,
            [_classification_label(value) for value in train_labels],
        )
        return
    _require_adapter_task(adapter, task="regression")
    cast(Regressor, adapter).fit(
        train_features,
        [_regression_label(value) for value in train_labels],
    )


def _require_adapter_task(adapter: Classifier | Regressor, *, task: str) -> None:
    y_hint = get_type_hints(adapter.fit).get("y")
    if task == "classification" and _hint_is_float_sequence(y_hint):
        raise TypeError("classification task requires a Classifier adapter")
    if task == "regression" and _hint_is_label_sequence(y_hint):
        raise TypeError("regression task requires a Regressor adapter")


def _hint_is_float_sequence(hint: object) -> bool:
    origin = get_origin(hint)
    if origin not in (Sequence, list):
        return False
    return get_args(hint) == (float,)


def _hint_is_label_sequence(hint: object) -> bool:
    origin = get_origin(hint)
    if origin not in (Sequence, list):
        return False
    return get_args(hint) == (str | int,)


def _labels_for_entities(
    entity_ids: Sequence[str],
    labels_by_entity: Mapping[str, Scalar],
) -> list[Scalar]:
    return [labels_by_entity[entity_id] for entity_id in entity_ids]


def _classification_label(value: Scalar) -> str | int:
    if isinstance(value, str | int):
        return value
    raise TypeError(f"classification label must be str or int, got {value!r}")


def _regression_label(value: Scalar) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"regression label must be numeric, got {value!r}")


def _prediction_rows(
    entity_ids: Sequence[str],
    *,
    y_true: Sequence[Scalar],
    y_pred: Sequence[Scalar],
    y_proba: Sequence[Sequence[float]] | None,
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for index, entity_id in enumerate(entity_ids):
        proba: tuple[float, ...] | None = None
        if y_proba is not None:
            proba = tuple(y_proba[index])
        rows.append(
            PredictionRow(
                entity_id=entity_id,
                y_pred=y_pred[index],
                y_true=y_true[index],
                y_proba=proba,
            )
        )
    return rows


def _maybe_predict_proba(
    adapter: Classifier | Regressor,
    features: FeatureTable,
) -> list[list[float]] | None:
    predict_proba = getattr(adapter, "predict_proba", None)
    if predict_proba is None:
        return None
    raw = predict_proba(features)
    return [list(item) for item in raw]


def _dataset_inputs(spec: ExperimentSpec) -> list[str]:
    if spec.dataset.payload_id is None:
        return []
    return [spec.dataset.payload_id]


def _feature_table_bytes(table: FeatureTable) -> bytes:
    payload = {
        "entity_ids": list(table.entity_ids),
        "columns": list(table.columns),
        "values": [list(row) for row in table.values],
        "source_payload_ids": list(table.source_payload_ids),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
