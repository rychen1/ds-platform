"""Deterministic parameter normalization for vendor integrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_RANDOM_STATE_KEYS = frozenset({"random_state", "random_seed", "seed"})


def merge_estimator_params(
    params: Mapping[str, Any] | None,
    *,
    random_state: int | None,
    random_param: str,
    reserved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge caller params with explicit random-state and reserved keys.

    Explicit ``random_state`` wins over duplicate keys in ``params``.
    Reserved keys (for example integration-controlled defaults) win over
    ``params`` but not over explicit ``random_state``.
    """
    merged: dict[str, Any] = dict(params or {})
    for key in _RANDOM_STATE_KEYS:
        merged.pop(key, None)
    if reserved:
        merged.update(reserved)
    if random_state is not None:
        merged[random_param] = random_state
    return merged


def jsonable_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-serializable copy of estimator params for artifacts."""
    if params is None:
        return {}
    return {str(key): value for key, value in params.items()}
