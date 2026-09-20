"""Shared fixtures for vendor integration tests."""

from __future__ import annotations

from ds_platform.modeling.features import FeatureTable, Scalar

_SOURCE = "a" * 64

_FAST_PARAMS = {
    "n_estimators": 12,
    "max_depth": 3,
}


def classification_table(*, include_missing: bool = False) -> FeatureTable:
    color: tuple[Scalar, ...] = ("red", "blue", "red", "blue", "red", "blue")
    x2: tuple[Scalar, ...] = (0.0, 1.0, 0.5, 1.5, 0.2, 1.1)
    if include_missing:
        x2 = (0.0, None, 0.5, 1.5, 0.2, 1.1)
    return FeatureTable(
        entity_ids=("e1", "e2", "e3", "e4", "e5", "e6"),
        columns=("x1", "x2", "color"),
        values=(
            (1.0, x2[0], color[0]),
            (2.0, x2[1], color[1]),
            (3.0, x2[2], color[2]),
            (4.0, x2[3], color[3]),
            (5.0, x2[4], color[4]),
            (6.0, x2[5], color[5]),
        ),
        source_payload_ids=(_SOURCE,) * 6,
    )


def classification_labels() -> list[str]:
    return ["A", "A", "B", "B", "A", "B"]


def regression_table() -> FeatureTable:
    return FeatureTable(
        entity_ids=("e1", "e2", "e3", "e4", "e5", "e6"),
        columns=("x1", "x2"),
        values=(
            (1.0, 0.0),
            (2.0, 1.0),
            (3.0, 0.5),
            (4.0, 1.5),
            (5.0, 0.2),
            (6.0, 1.1),
        ),
        source_payload_ids=(_SOURCE,) * 6,
    )


def regression_labels() -> list[float]:
    return [1.0, 2.5, 3.0, 4.5, 5.0, 6.5]
