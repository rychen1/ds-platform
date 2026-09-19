"""Capability protocols (modeling Part 6)."""

from __future__ import annotations

import sys

from ds_platform.modeling.capabilities import Classifier, Regressor
from ds_platform.modeling.features import FeatureTable

_FORBIDDEN = {
    "board_game_analysis",
    "numpy",
    "pandas",
    "restaurant_intelligence",
    "sklearn",
}

_SOURCE = "a" * 64


class _DummyClassifier:
    def fit(self, features: FeatureTable, y: list[str | int]) -> None:
        self.feature_count = len(features.entity_ids)
        self.labels = list(y)

    def predict(self, features: FeatureTable) -> list[str | int]:
        default = self.labels[0] if self.labels else "A"
        return [default] * len(features.entity_ids)


class _DummyRegressor:
    def fit(self, features: FeatureTable, y: list[float]) -> None:
        self.labels = list(y)

    def predict(self, features: FeatureTable) -> list[float]:
        default = self.labels[0] if self.labels else 0.0
        return [default] * len(features.entity_ids)


def test_dummy_classifier_satisfies_protocol() -> None:
    assert isinstance(_DummyClassifier(), Classifier)


def test_dummy_regressor_satisfies_protocol() -> None:
    assert isinstance(_DummyRegressor(), Regressor)


def test_capabilities_import_does_not_load_forbidden_modules() -> None:
    import ds_platform.modeling.capabilities  # noqa: F401

    assert _FORBIDDEN.intersection(sys.modules) == set()
