"""CatBoost adapters for ds-platform modeling protocols."""

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
    feature_table_to_dataframe,
    resolve_column_names,
)
from ds_platform.integrations._params import jsonable_params, merge_estimator_params
from ds_platform.modeling.features import FeatureTable

MEDIA_TYPE = "application/x-catboost+cbm"


def _require_catboost():
    try:
        import catboost as cb
    except ImportError as exc:
        raise ImportError(
            "CatBoost integration requires the optional dependency. "
            "Install with: pip install 'ds-platform[catboost]'"
        ) from exc
    return cb


class CatBoostClassifier:
    """Platform adapter wrapping ``catboost.CatBoostClassifier``."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        cat_features: Sequence[str] | None = None,
    ) -> None:
        _require_catboost()
        self._params = dict(params or {})
        self._random_state = random_state
        self._cat_features = tuple(cat_features or ())
        self._estimator: Any = None

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None:
        cb = _require_catboost()
        cat_features = resolve_column_names(features, self._cat_features)
        frame = feature_table_to_dataframe(features)
        estimator_params = merge_estimator_params(
            self._params,
            random_state=self._random_state,
            random_param="random_seed",
            reserved={
                "allow_writing_files": self._params.get("allow_writing_files", False),
                "verbose": self._params.get("verbose", False),
            },
        )
        self._estimator = cb.CatBoostClassifier(**estimator_params)
        if cat_features:
            self._estimator.fit(frame, list(y), cat_features=list(cat_features))
        else:
            self._estimator.fit(frame, list(y))

    def predict(self, features: FeatureTable) -> list[str | int]:
        estimator = _require_fitted(self._estimator)
        frame = feature_table_to_dataframe(features)
        raw = estimator.predict(frame)
        flattened = raw.ravel().tolist()
        return [_coerce_label(value) for value in flattened]

    def predict_proba(self, features: FeatureTable) -> list[list[float]]:
        estimator = _require_fitted(self._estimator)
        frame = feature_table_to_dataframe(features)
        return [list(row) for row in estimator.predict_proba(frame)]

    def serialize(self) -> bytes:
        estimator = _require_fitted(self._estimator)
        return pack_integration_model(
            library="catboost",
            task="classification",
            metadata={
                "params": jsonable_params(self._params),
                "random_state": self._random_state,
                "cat_features": list(self._cat_features),
            },
            model_bytes=_save_catboost_model(estimator),
        )

    @classmethod
    def deserialize(cls, data: bytes) -> CatBoostClassifier:
        cb = _require_catboost()
        header, model_bytes = unpack_integration_model(data)
        adapter = cls(
            params=header.get("params"),
            random_state=header.get("random_state"),
            cat_features=header.get("cat_features"),
        )
        adapter._estimator = cb.CatBoostClassifier()
        _load_catboost_model(adapter._estimator, model_bytes)
        return adapter


class CatBoostRegressor:
    """Platform adapter wrapping ``catboost.CatBoostRegressor``."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        cat_features: Sequence[str] | None = None,
    ) -> None:
        _require_catboost()
        self._params = dict(params or {})
        self._random_state = random_state
        self._cat_features = tuple(cat_features or ())
        self._estimator: Any = None

    def fit(self, features: FeatureTable, y: Sequence[float]) -> None:
        cb = _require_catboost()
        cat_features = resolve_column_names(features, self._cat_features)
        frame = feature_table_to_dataframe(features)
        estimator_params = merge_estimator_params(
            self._params,
            random_state=self._random_state,
            random_param="random_seed",
            reserved={
                "allow_writing_files": self._params.get("allow_writing_files", False),
                "verbose": self._params.get("verbose", False),
            },
        )
        self._estimator = cb.CatBoostRegressor(**estimator_params)
        if cat_features:
            self._estimator.fit(frame, list(y), cat_features=list(cat_features))
        else:
            self._estimator.fit(frame, list(y))

    def predict(self, features: FeatureTable) -> list[float]:
        estimator = _require_fitted(self._estimator)
        frame = feature_table_to_dataframe(features)
        raw = estimator.predict(frame)
        return [float(value) for value in raw.ravel().tolist()]

    def serialize(self) -> bytes:
        estimator = _require_fitted(self._estimator)
        return pack_integration_model(
            library="catboost",
            task="regression",
            metadata={
                "params": jsonable_params(self._params),
                "random_state": self._random_state,
                "cat_features": list(self._cat_features),
            },
            model_bytes=_save_catboost_model(estimator),
        )

    @classmethod
    def deserialize(cls, data: bytes) -> CatBoostRegressor:
        cb = _require_catboost()
        header, model_bytes = unpack_integration_model(data)
        adapter = cls(
            params=header.get("params"),
            random_state=header.get("random_state"),
            cat_features=header.get("cat_features"),
        )
        adapter._estimator = cb.CatBoostRegressor()
        _load_catboost_model(adapter._estimator, model_bytes)
        return adapter


def _require_fitted(estimator: Any) -> Any:
    if estimator is None:
        raise RuntimeError("adapter must be fitted before predict or serialize")
    return estimator


def _coerce_label(value: object) -> str | int:
    if isinstance(value, str | int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"unexpected classification label type: {type(value)!r}")


def _save_catboost_model(estimator) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".cbm")
    os.close(fd)
    try:
        estimator.save_model(path)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def _load_catboost_model(estimator, data: bytes) -> None:
    fd, path = tempfile.mkstemp(suffix=".cbm")
    os.close(fd)
    try:
        Path(path).write_bytes(data)
        estimator.load_model(path)
    finally:
        os.unlink(path)


__all__ = [
    "MEDIA_TYPE",
    "CatBoostClassifier",
    "CatBoostRegressor",
]
