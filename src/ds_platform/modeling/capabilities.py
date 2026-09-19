"""Structural typing protocols for modeling adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.sequences import SequenceTable


@runtime_checkable
class Classifier(Protocol):
    """Supervised classification adapter."""

    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None: ...

    def predict(self, features: FeatureTable) -> Sequence[str | int]: ...


@runtime_checkable
class Regressor(Protocol):
    """Supervised regression adapter."""

    def fit(self, features: FeatureTable, y: Sequence[float]) -> None: ...

    def predict(self, features: FeatureTable) -> Sequence[float]: ...


@runtime_checkable
class Embedder(Protocol):
    """Encode opaque records into a :class:`RepresentationTable`.

    ``fit`` may be a no-op for frozen or pretrained wrappers.
    """

    def fit(
        self,
        entity_ids: Sequence[str],
        records: Sequence[Mapping[str, object]],
    ) -> None: ...

    def encode(
        self,
        entity_ids: Sequence[str],
        records: Sequence[Mapping[str, object]],
    ) -> RepresentationTable: ...


@runtime_checkable
class ConditionedEncoder(Protocol):
    """Map aligned context and condition representations to an output table.

    ``contexts`` and ``conditions`` are expected to be keyed by ``pair_ids``.
    """

    def fit(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> None: ...

    def encode(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> RepresentationTable: ...


@runtime_checkable
class PerspectiveComposer(Protocol):
    """Compose a same-subject chain of perspective vectors."""

    def compose(self, chain: Sequence[tuple[float, ...]]) -> tuple[float, ...]: ...


@runtime_checkable
class SequenceEncoder(Protocol):
    """Contextualize event representations along a :class:`SequenceTable`.

    ``encode`` returns event-level vectors keyed by the same ``event_ids``.
    """

    def fit(
        self,
        sequences: SequenceTable,
        events: RepresentationTable,
    ) -> None: ...

    def encode(
        self,
        sequences: SequenceTable,
        events: RepresentationTable,
    ) -> RepresentationTable: ...


@runtime_checkable
class Clusterer(Protocol):
    """Assign cluster labels to a :class:`RepresentationTable`."""

    def fit(self, table: RepresentationTable) -> None: ...

    def predict(self, table: RepresentationTable) -> Sequence[str | int]: ...
