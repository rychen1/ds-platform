"""LightGBM integration tests."""

from __future__ import annotations

import pytest
from helpers import (
    _FAST_PARAMS,
    classification_labels,
    classification_table,
    regression_labels,
    regression_table,
)

from ds_platform.integrations.lightgbm import (
    MEDIA_TYPE,
    LightGBMClassifier,
    LightGBMRegressor,
)
from ds_platform.modeling.capabilities import Classifier, Regressor


def test_lightgbm_classifier_satisfies_protocol() -> None:
    assert isinstance(LightGBMClassifier(), Classifier)


def test_lightgbm_regressor_satisfies_protocol() -> None:
    assert isinstance(LightGBMRegressor(), Regressor)


def test_lightgbm_classifier_fit_predict_serialize_round_trip() -> None:
    table = classification_table()
    labels = classification_labels()
    adapter = LightGBMClassifier(
        params=_FAST_PARAMS,
        random_state=7,
        categorical_columns=("color",),
    )
    adapter.fit(table, labels)
    predictions = adapter.predict(table)
    assert len(predictions) == len(labels)
    proba = adapter.predict_proba(table)
    assert len(proba) == len(labels)

    payload = adapter.serialize()
    reloaded = LightGBMClassifier.deserialize(payload)
    reloaded_predictions = reloaded.predict(table)
    assert reloaded_predictions == predictions


def test_lightgbm_regressor_fit_predict_serialize_round_trip() -> None:
    table = regression_table()
    labels = regression_labels()
    adapter = LightGBMRegressor(params=_FAST_PARAMS, random_state=11)
    adapter.fit(table, labels)
    predictions = adapter.predict(table)
    assert len(predictions) == len(labels)

    payload = adapter.serialize()
    reloaded = LightGBMRegressor.deserialize(payload)
    reloaded_predictions = reloaded.predict(table)
    assert reloaded_predictions == pytest.approx(predictions)


def test_lightgbm_media_type_constant() -> None:
    assert MEDIA_TYPE == "application/x-lightgbm+txt"
