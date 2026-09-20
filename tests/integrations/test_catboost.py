"""CatBoost integration tests."""

from __future__ import annotations

import pytest
from helpers import (
    _FAST_PARAMS,
    classification_labels,
    classification_table,
    regression_labels,
    regression_table,
)

from ds_platform.integrations.catboost import (
    MEDIA_TYPE,
    CatBoostClassifier,
    CatBoostRegressor,
)
from ds_platform.modeling.capabilities import Classifier, Regressor


def test_catboost_classifier_satisfies_protocol() -> None:
    assert isinstance(CatBoostClassifier(), Classifier)


def test_catboost_regressor_satisfies_protocol() -> None:
    assert isinstance(CatBoostRegressor(), Regressor)


def test_catboost_classifier_with_categorical_features() -> None:
    table = classification_table(include_missing=True)
    labels = classification_labels()
    adapter = CatBoostClassifier(
        params=_FAST_PARAMS,
        random_state=7,
        cat_features=("color",),
    )
    adapter.fit(table, labels)
    predictions = adapter.predict(table)
    assert len(predictions) == len(labels)
    proba = adapter.predict_proba(table)
    assert len(proba) == len(labels)

    payload = adapter.serialize()
    reloaded = CatBoostClassifier.deserialize(payload)
    assert reloaded.predict(table) == predictions


def test_catboost_regressor_fit_predict_serialize_round_trip() -> None:
    table = regression_table()
    labels = regression_labels()
    adapter = CatBoostRegressor(params=_FAST_PARAMS, random_state=11)
    adapter.fit(table, labels)
    predictions = adapter.predict(table)
    assert len(predictions) == len(labels)

    payload = adapter.serialize()
    reloaded = CatBoostRegressor.deserialize(payload)
    assert reloaded.predict(table) == pytest.approx(predictions)


def test_catboost_rejects_unknown_categorical_column() -> None:
    table = classification_table()
    adapter = CatBoostClassifier(cat_features=("missing",))
    with pytest.raises(KeyError, match="column not found"):
        adapter.fit(table, classification_labels())


def test_catboost_media_type_constant() -> None:
    assert MEDIA_TYPE == "application/x-catboost+cbm"
