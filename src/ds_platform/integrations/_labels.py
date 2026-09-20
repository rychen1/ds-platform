"""Classification label encoding for vendor integrations."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence


def encode_labels(
    labels: Sequence[str | int],
) -> tuple[list[int], tuple[str | int, ...]]:
    """Return integer-encoded labels and stable class order."""
    classes: list[str | int] = []
    for label in labels:
        if label not in classes:
            classes.append(label)
    mapping = {label: index for index, label in enumerate(classes)}
    return [mapping[label] for label in labels], tuple(classes)


def decode_labels(
    encoded: Iterable[object],
    classes: Sequence[str | int],
) -> list[str | int]:
    """Map vendor numeric predictions back to platform labels."""
    decoded: list[str | int] = []
    for value in encoded:
        if isinstance(value, str):
            decoded.append(value)
            continue
        if isinstance(value, bool):
            raise TypeError(f"unexpected encoded label type: {type(value)!r}")
        if isinstance(value, numbers.Integral):
            decoded.append(classes[int(value)])
            continue
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            decoded.append(classes[math.trunc(float(value))])
            continue
        raise TypeError(f"unexpected encoded label type: {type(value)!r}")
    return decoded
