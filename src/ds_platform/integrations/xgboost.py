"""XGBoost adapters for ds-platform modeling protocols."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ds_platform.integrations._artifact import (
    pack_integration_model,
    unpack_integration_model,
)
from ds_platform.integrations._conversion import (
    apply_categorical_columns,
    feature_table_to_dataframe,
    resolve_column_names,
)
from ds_platform.integrations._labels import decode_labels, encode_labels
from ds_platform.integrations._params import jsonable_params, merge_estimator_params
from ds_platform.modeling.features import FeatureTable

MEDIA_TYPE = "application/x-xgboost+ubj"


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost integration requires the optional dependency. "
            "Install with: pip install 'ds-platform[xgboost]'"
        ) from exc
    return xgb


class XGBoostClassifier:
    """Platform adapter wrapping ``xgboost.XGBClassifier``."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> None:
        _require_xgboost()
        self._params = dict(params or {})
        self._random_state = random_state
        self._categorical_columns = tuple(categorical_columns or ())
        self._estimator: Any = None
        self._classes: tuple[str | int, ...] = ()

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None:
        xgb = _require_xgboost()
        resolve_column_names(features, self._categorical_columns)
        frame = feature_table_to_dataframe(features)
        if self._categorical_columns:
            frame = apply_categorical_columns(frame, self._categorical_columns)
        encoded_labels, self._classes = encode_labels(y)
        estimator_params = merge_estimator_params(
            self._params,
            random_state=self._random_state,
            random_param="random_state",
            reserved={
                "tree_method": self._params.get("tree_method", "hist"),
                "enable_categorical": bool(self._categorical_columns)
                or bool(self._params.get("enable_categorical")),
            },
        )
        self._estimator = xgb.XGBClassifier(**estimator_params)
        self._estimator.fit(frame, encoded_labels)

    def predict(self, features: FeatureTable) -> list[str | int]:
        estimator = _require_fitted(self._estimator)
        frame = _predict_frame(features, self._categorical_columns)
        raw = estimator.predict(frame)
        return decode_labels(raw, self._classes)

    def predict_proba(self, features: FeatureTable) -> list[list[float]]:
        estimator = _require_fitted(self._estimator)
        frame = _predict_frame(features, self._categorical_columns)
        return [list(row) for row in estimator.predict_proba(frame)]

    def serialize(self) -> bytes:
        estimator = _require_fitted(self._estimator)
        return pack_integration_model(
            library="xgboost",
            task="classification",
            metadata={
                "params": jsonable_params(self._params),
                "random_state": self._random_state,
                "categorical_columns": list(self._categorical_columns),
                "classes": list(self._classes),
            },
            model_bytes=_save_xgboost_model(estimator),
        )

    @classmethod
    def deserialize(cls, data: bytes) -> XGBoostClassifier:
        xgb = _require_xgboost()
        header, model_bytes = unpack_integration_model(data)
        adapter = cls(
            params=header.get("params"),
            random_state=header.get("random_state"),
            categorical_columns=header.get("categorical_columns"),
        )
        adapter._classes = tuple(header.get("classes") or ())
        adapter._estimator = xgb.XGBClassifier()
        _load_xgboost_model(adapter._estimator, model_bytes)
        return adapter


class XGBoostRegressor:
    """Platform adapter wrapping ``xgboost.XGBRegressor``."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> None:
        _require_xgboost()
        self._params = dict(params or {})
        self._random_state = random_state
        self._categorical_columns = tuple(categorical_columns or ())
        self._estimator: Any = None

    def fit(self, features: FeatureTable, y: Sequence[float]) -> None:
        xgb = _require_xgboost()
        resolve_column_names(features, self._categorical_columns)
        frame = feature_table_to_dataframe(features)
        if self._categorical_columns:
            frame = apply_categorical_columns(frame, self._categorical_columns)
        estimator_params = merge_estimator_params(
            self._params,
            random_state=self._random_state,
            random_param="random_state",
            reserved={
                "tree_method": self._params.get("tree_method", "hist"),
                "enable_categorical": bool(self._categorical_columns)
                or bool(self._params.get("enable_categorical")),
            },
        )
        self._estimator = xgb.XGBRegressor(**estimator_params)
        self._estimator.fit(frame, list(y))

    def predict(self, features: FeatureTable) -> list[float]:
        estimator = _require_fitted(self._estimator)
        frame = _predict_frame(features, self._categorical_columns)
        return [float(value) for value in estimator.predict(frame)]

    def serialize(self) -> bytes:
        estimator = _require_fitted(self._estimator)
        return pack_integration_model(
            library="xgboost",
            task="regression",
            metadata={
                "params": jsonable_params(self._params),
                "random_state": self._random_state,
                "categorical_columns": list(self._categorical_columns),
            },
            model_bytes=_save_xgboost_model(estimator),
        )

    @classmethod
    def deserialize(cls, data: bytes) -> XGBoostRegressor:
        xgb = _require_xgboost()
        header, model_bytes = unpack_integration_model(data)
        adapter = cls(
            params=header.get("params"),
            random_state=header.get("random_state"),
            categorical_columns=header.get("categorical_columns"),
        )
        adapter._estimator = xgb.XGBRegressor()
        _load_xgboost_model(adapter._estimator, model_bytes)
        return adapter


def _predict_frame(features: FeatureTable, categorical_columns: Sequence[str]):
    frame = feature_table_to_dataframe(features)
    if categorical_columns:
        frame = apply_categorical_columns(frame, categorical_columns)
    return frame


def _require_fitted(estimator: Any) -> Any:
    if estimator is None:
        raise RuntimeError("adapter must be fitted before predict or serialize")
    return estimator


def _save_xgboost_model(estimator) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".ubj")
    os.close(fd)
    try:
        estimator.save_model(path)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def _load_xgboost_model(estimator, data: bytes) -> None:
    fd, path = tempfile.mkstemp(suffix=".ubj")
    os.close(fd)
    try:
        Path(path).write_bytes(data)
        estimator.load_model(path)
    finally:
        os.unlink(path)


__all__ = [
    "MEDIA_TYPE",
    "XGBoostClassifier",
    "XGBoostRegressor",
]
