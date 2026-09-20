"""LightGBM adapters for ds-platform modeling protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from ds_platform.integrations._labels import decode_labels
from ds_platform.integrations._params import jsonable_params, merge_estimator_params
from ds_platform.modeling.features import FeatureTable

MEDIA_TYPE = "application/x-lightgbm+txt"


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM integration requires the optional dependency. "
            "Install with: pip install 'ds-platform[lightgbm]'"
        ) from exc
    return lgb


class LightGBMClassifier:
    """Platform adapter wrapping ``lightgbm.LGBMClassifier``."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> None:
        _require_lightgbm()
        self._params = dict(params or {})
        self._random_state = random_state
        self._categorical_columns = tuple(categorical_columns or ())
        self._estimator: Any = None
        self._booster: Any = None
        self._classes: tuple[str | int, ...] = ()

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None:
        lgb = _require_lightgbm()
        resolve_column_names(features, self._categorical_columns)
        frame = feature_table_to_dataframe(features)
        if self._categorical_columns:
            frame = apply_categorical_columns(frame, self._categorical_columns)
        estimator_params = merge_estimator_params(
            self._params,
            random_state=self._random_state,
            random_param="random_state",
            reserved={"verbosity": self._params.get("verbosity", -1)},
        )
        self._estimator = lgb.LGBMClassifier(**estimator_params)
        if self._categorical_columns:
            self._estimator.fit(
                frame,
                list(y),
                categorical_feature=list(self._categorical_columns),
            )
        else:
            self._estimator.fit(frame, list(y))
        self._booster = self._estimator.booster_
        self._classes = tuple(self._estimator.classes_)

    def predict(self, features: FeatureTable) -> list[str | int]:
        frame = _predict_frame(features, self._categorical_columns)
        return _predict_class_labels(self._booster, frame, self._classes)

    def predict_proba(self, features: FeatureTable) -> list[list[float]]:
        frame = _predict_frame(features, self._categorical_columns)
        booster = _require_booster(self._booster)
        matrix = booster.predict(frame)
        if len(self._classes) <= 2 and matrix.ndim == 1:
            matrix = _binary_scores_to_proba(matrix)
        return [list(row) for row in matrix]

    def serialize(self) -> bytes:
        booster = _require_booster(self._booster)
        return pack_integration_model(
            library="lightgbm",
            task="classification",
            metadata={
                "params": jsonable_params(self._params),
                "random_state": self._random_state,
                "categorical_columns": list(self._categorical_columns),
                "classes": list(self._classes),
            },
            model_bytes=booster.model_to_string().encode("utf-8"),
        )

    @classmethod
    def deserialize(cls, data: bytes) -> LightGBMClassifier:
        lgb = _require_lightgbm()
        header, model_bytes = unpack_integration_model(data)
        adapter = cls(
            params=header.get("params"),
            random_state=header.get("random_state"),
            categorical_columns=header.get("categorical_columns"),
        )
        adapter._booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
        adapter._classes = tuple(header.get("classes") or ())
        return adapter


class LightGBMRegressor:
    """Platform adapter wrapping ``lightgbm.LGBMRegressor``."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> None:
        _require_lightgbm()
        self._params = dict(params or {})
        self._random_state = random_state
        self._categorical_columns = tuple(categorical_columns or ())
        self._estimator: Any = None
        self._booster: Any = None

    def fit(self, features: FeatureTable, y: Sequence[float]) -> None:
        lgb = _require_lightgbm()
        resolve_column_names(features, self._categorical_columns)
        frame = feature_table_to_dataframe(features)
        if self._categorical_columns:
            frame = apply_categorical_columns(frame, self._categorical_columns)
        estimator_params = merge_estimator_params(
            self._params,
            random_state=self._random_state,
            random_param="random_state",
            reserved={"verbosity": self._params.get("verbosity", -1)},
        )
        self._estimator = lgb.LGBMRegressor(**estimator_params)
        if self._categorical_columns:
            self._estimator.fit(
                frame,
                list(y),
                categorical_feature=list(self._categorical_columns),
            )
        else:
            self._estimator.fit(frame, list(y))
        self._booster = self._estimator.booster_

    def predict(self, features: FeatureTable) -> list[float]:
        frame = _predict_frame(features, self._categorical_columns)
        raw = _predict_values(self._booster, frame)
        return [float(value) for value in raw]

    def serialize(self) -> bytes:
        booster = _require_booster(self._booster)
        return pack_integration_model(
            library="lightgbm",
            task="regression",
            metadata={
                "params": jsonable_params(self._params),
                "random_state": self._random_state,
                "categorical_columns": list(self._categorical_columns),
            },
            model_bytes=booster.model_to_string().encode("utf-8"),
        )

    @classmethod
    def deserialize(cls, data: bytes) -> LightGBMRegressor:
        lgb = _require_lightgbm()
        header, model_bytes = unpack_integration_model(data)
        adapter = cls(
            params=header.get("params"),
            random_state=header.get("random_state"),
            categorical_columns=header.get("categorical_columns"),
        )
        adapter._booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
        return adapter


def _predict_frame(features: FeatureTable, categorical_columns: Sequence[str]):
    frame = feature_table_to_dataframe(features)
    if categorical_columns:
        frame = apply_categorical_columns(frame, categorical_columns)
    return frame


def _require_booster(booster: Any) -> Any:
    if booster is None:
        raise RuntimeError("adapter must be fitted before predict or serialize")
    return booster


def _predict_values(booster: Any, frame: Any) -> Any:
    fitted = _require_booster(booster)
    return fitted.predict(frame)


def _predict_class_labels(
    booster: Any,
    frame: Any,
    classes: Sequence[str | int],
) -> list[str | int]:
    import numpy as np

    fitted = _require_booster(booster)
    raw = fitted.predict(frame)
    array = np.asarray(raw)
    if len(classes) == 2 and array.ndim == 1 and np.all((array >= 0) & (array <= 1)):
        indices = (array >= 0.5).astype(int)
        return decode_labels(indices, classes)
    if array.ndim == 2:
        indices = array.argmax(axis=1)
        return decode_labels(indices, classes)
    return decode_labels(array, classes)


def _binary_scores_to_proba(scores):
    import numpy as np

    positive = 1.0 / (1.0 + np.exp(-scores))
    return np.column_stack([1.0 - positive, positive])


__all__ = [
    "MEDIA_TYPE",
    "LightGBMClassifier",
    "LightGBMRegressor",
]
