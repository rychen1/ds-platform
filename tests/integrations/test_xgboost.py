"""XGBoost integration tests."""

from __future__ import annotations

import pytest
from helpers import (
    _FAST_PARAMS,
    classification_labels,
    classification_table,
    regression_labels,
    regression_table,
)

from ds_platform.integrations.xgboost import (
    MEDIA_TYPE,
    XGBoostClassifier,
    XGBoostRegressor,
)
from ds_platform.modeling.capabilities import Classifier, Regressor


def test_xgboost_classifier_satisfies_protocol() -> None:
    assert isinstance(XGBoostClassifier(), Classifier)


def test_xgboost_regressor_satisfies_protocol() -> None:
    assert isinstance(XGBoostRegressor(), Regressor)


def test_xgboost_classifier_fit_predict_serialize_round_trip() -> None:
    table = classification_table()
    labels = classification_labels()
    adapter = XGBoostClassifier(
        params=_FAST_PARAMS,
        random_state=7,
        categorical_columns=("color",),
    )
    adapter.fit(table, labels)
    predictions = adapter.predict(table)
    assert len(predictions) == len(labels)
    proba = adapter.predict_proba(table)
    assert len(proba) == len(labels)
    assert len(proba[0]) >= 2

    payload = adapter.serialize()
    reloaded = XGBoostClassifier.deserialize(payload)
    reloaded_predictions = reloaded.predict(table)
    assert reloaded_predictions == predictions


def test_xgboost_regressor_fit_predict_serialize_round_trip() -> None:
    table = regression_table()
    labels = regression_labels()
    adapter = XGBoostRegressor(params=_FAST_PARAMS, random_state=11)
    adapter.fit(table, labels)
    predictions = adapter.predict(table)
    assert len(predictions) == len(labels)

    payload = adapter.serialize()
    reloaded = XGBoostRegressor.deserialize(payload)
    reloaded_predictions = reloaded.predict(table)
    assert reloaded_predictions == pytest.approx(predictions)


def test_xgboost_classifier_uses_feature_names() -> None:
    table = regression_table()
    adapter = XGBoostRegressor(params=_FAST_PARAMS, random_state=3)
    adapter.fit(table, regression_labels())
    estimator = adapter._estimator
    assert estimator is not None
    assert list(estimator.feature_names_in_) == list(table.columns)


def test_xgboost_media_type_constant() -> None:
    assert MEDIA_TYPE == "application/x-xgboost+ubj"
