"""Regressions for the v0 adversarial-audit fixes.

These used to document broken behavior. They now assert the repaired
contracts. Keep them even if the original audit report is historical.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from ds_platform.hashing import canonical_json_bytes, payload_id, record_id
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.modeling.encode import run_encoding
from ds_platform.modeling.evaluate import EvaluationReport, evaluate, rmse
from ds_platform.modeling.features import (
    FeatureTable,
    align_feature_tables,
    require_source_payload_ids,
)
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.run import run_experiment
from ds_platform.modeling.spec import (
    DatasetRef,
    EncodingSpec,
    ExperimentSpec,
    FeatureSpec,
    MetricSpec,
    ModelSpec,
    SplitSpec,
    TargetSpec,
    experiment_config_hash,
)
from ds_platform.modeling.split import split_entities, split_groups
from ds_platform.store import HashMismatchError, LocalStore
from ds_platform.types import (
    ArtifactKind,
    ArtifactRecord,
    ContractRef,
    Environment,
    ExternalRunRef,
    RunContext,
)

_PID = payload_id(b"audit-subject")
_CREATED = datetime(2026, 9, 19, 19, 40, 0, tzinfo=UTC)


def _run(**overrides: object) -> RunContext:
    fields: dict[str, object] = {
        "run_id": "audit-run",
        "project": "audit",
        "started_at": _CREATED,
        "environment": Environment.LOCAL,
    }
    fields.update(overrides)
    return RunContext.model_validate(fields)


def _record(**overrides: object) -> ArtifactRecord:
    fields: dict[str, object] = {
        "payload_id": _PID,
        "media_type": "application/json",
        "kind": ArtifactKind.DOCUMENT,
        "produced_by": _run(),
        "created_at": _CREATED,
    }
    fields.update(overrides)
    return ArtifactRecord.model_validate(fields)


def test_microsecond_timestamps_change_record_id() -> None:
    first = _record(created_at=datetime(2026, 9, 19, 19, 40, 0, 1, tzinfo=UTC))
    second = _record(created_at=datetime(2026, 9, 19, 19, 40, 0, 999999, tzinfo=UTC))
    assert record_id(first) != record_id(second)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _record(created_at=datetime(2026, 9, 19, 19, 40, 0))


def test_offset_datetimes_normalize_to_utc() -> None:
    eastern = timezone(timedelta(hours=-4))
    left = _record(created_at=datetime(2026, 9, 19, 15, 40, 0, tzinfo=eastern))
    right = _record(created_at=datetime(2026, 9, 19, 19, 40, 0, tzinfo=UTC))
    assert record_id(left) == record_id(right)


def test_empty_lists_are_omitted_from_canonical_record() -> None:
    encoded = canonical_json_bytes(_record()).decode()
    assert '"inputs"' not in encoded
    assert '"schema_version"' not in encoded


def test_adding_empty_notes_list_does_not_change_rehashed_identity() -> None:
    current = json.loads(canonical_json_bytes(_record()).decode())
    evolved = dict(current)
    evolved["notes"] = []
    assert canonical_json_bytes(current) == canonical_json_bytes(evolved)


def test_artifact_record_is_frozen() -> None:
    record = _record(inputs=[_PID])
    with pytest.raises(ValidationError):
        record.inputs = (*record.inputs, "b" * 64)  # type: ignore[misc]


def test_get_detects_corrupted_payload(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    data = b"intact-bytes"
    pid = payload_id(data)
    store.put(pid, data, media_type="text/plain")
    path = tmp_path / pid[:2] / pid[2:4] / pid
    path.write_bytes(b"corrupted")
    with pytest.raises(HashMismatchError):
        store.get(pid)
    store.put(pid, data, media_type="text/plain")
    assert store.get(pid) == data


def test_run_encoding_uses_representation_kind(tmp_path: Path) -> None:
    class _Embedder:
        def fit(self, entity_ids: object, records: object) -> None:
            del entity_ids, records

        def encode(self, entity_ids: object, records: object) -> RepresentationTable:
            del records
            return RepresentationTable(
                entity_ids=tuple(entity_ids),  # type: ignore[arg-type]
                vectors=((1.0, 0.0),),
                dim=2,
                source_payload_ids=(_PID,),
            )

    store = LocalStore(tmp_path)
    result = run_encoding(
        EncodingSpec(family="toy", dim=2, seed=1),
        encoder=_Embedder(),
        entity_ids=["e1"],
        records=[{"x": 1}],
        store=store,
        run=_run(),
        serialize_model=lambda _model: b"encoder",
        inputs=[_PID],
    )
    sidecar = _sidecar_for_payload(tmp_path, result.representation_payload_id)
    record = json.loads(sidecar)
    assert record["kind"] == ArtifactKind.REPRESENTATION


def test_evaluate_rejects_metric_params_and_probabilities() -> None:
    with pytest.raises(ValueError, match="does not accept params"):
        evaluate(
            ["A"],
            ["A"],
            [MetricSpec(name="accuracy", params={"average": "weighted"})],
        )
    with pytest.raises(ValueError, match="y_proba is not used"):
        evaluate(["A"], ["A"], [MetricSpec(name="accuracy")], y_proba=[[1.0]])


def test_evaluation_report_rejects_nan() -> None:
    with pytest.raises(ValidationError, match="finite"):
        EvaluationReport(metrics={"rmse": float("nan")}, n=1, n_missing=0)


def test_model_spec_params_are_immutable() -> None:
    spec = ExperimentSpec(
        experiment_id="exp",
        dataset=DatasetRef(logical_key="audit:data:v0"),
        features=FeatureSpec(views=("v",)),
        target=TargetSpec(column="y", task="classification"),
        model=ModelSpec(family="x", params={"n": 1}),
        split=SplitSpec(method="holdout", seed=1, test_size=0.2),
        metrics=(MetricSpec(name="accuracy"),),
        seed=1,
    )
    before = experiment_config_hash(spec)
    with pytest.raises(TypeError):
        spec.model.params["n"] = 2  # type: ignore[index]
    assert experiment_config_hash(spec) == before


def test_experiment_id_is_not_in_config_hash() -> None:
    left = ExperimentSpec(
        experiment_id="exp-a",
        dataset=DatasetRef(payload_id=_PID),
        features=FeatureSpec(views=("v",)),
        target=TargetSpec(column="y", task="classification"),
        model=ModelSpec(family="x"),
        split=SplitSpec(method="holdout", seed=1, test_size=0.2),
        metrics=(MetricSpec(name="accuracy"),),
        seed=1,
    )
    right = left.model_copy(update={"experiment_id": "exp-b"})
    assert experiment_config_hash(left) == experiment_config_hash(right)


def test_feature_source_ids_must_be_hashes() -> None:
    with pytest.raises(ValueError, match="sha256"):
        require_source_payload_ids(("not-a-payload-id",))
    FeatureTable(
        entity_ids=("e1",),
        columns=("x",),
        values=((1,),),
        source_payload_ids=("not-a-payload-id",),
    )


def test_representation_rejects_nan() -> None:
    with pytest.raises(ValidationError, match="finite"):
        RepresentationTable(
            entity_ids=("e1",),
            vectors=((float("nan"), 0.0),),
            dim=2,
            source_payload_ids=(_PID,),
        )


def test_align_can_error_on_missing_entities() -> None:
    left = FeatureTable(
        entity_ids=("e1", "e2"),
        columns=("x",),
        values=((1,), (2,)),
        source_payload_ids=(_PID, _PID),
    )
    right = FeatureTable(
        entity_ids=("e2",),
        columns=("z",),
        values=((9,),),
        source_payload_ids=(_PID,),
    )
    with pytest.raises(ValueError, match="missing entities"):
        align_feature_tables([left, right], on_missing="error")


def test_rmse_rejects_bool() -> None:
    with pytest.raises(TypeError, match="numeric"):
        rmse([True], [False])


def test_spec_canonical_datetime_matches_artifact_form() -> None:
    instant = datetime(2026, 3, 14, 15, 9, 26, 123456, tzinfo=UTC)
    assert spec_canonical_json_bytes({"when": instant}) == canonical_json_bytes(
        {"when": instant}
    )


def test_external_run_refs_are_open_ended() -> None:
    ref = ExternalRunRef(system="prefect", run_id="flow-1")
    run = _run(external_run_ids=(ref,))
    assert run.external_run_ids[0].system == "prefect"


def test_contract_change_does_not_change_payload_id() -> None:
    first = _record(
        contract_ref=ContractRef(uri="https://example.test/a", schema_hash=_PID)
    )
    second = _record(
        contract_ref=ContractRef(uri="https://example.test/b", schema_hash="c" * 64)
    )
    assert first.payload_id == second.payload_id
    assert record_id(first) != record_id(second)


def test_temporal_split_is_ordered() -> None:
    assignment = split_entities(
        ["late", "early"],
        SplitSpec(method="temporal", seed=0, test_size=0.5),
        timestamps=[
            datetime(2026, 6, 1).date(),
            datetime(2026, 1, 1).date(),
        ],
    )[0]
    assert assignment.train_ids == ("early",)
    assert assignment.test_ids == ("late",)


def test_schema_version_field_exists() -> None:
    assert ArtifactRecord.model_fields["schema_version"].default == 0
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate({**_record().model_dump(), "schema_version": 1})


# --- Pass 2 — regressions for post-audit fixes ---


def test_exists_false_on_corrupt_payload(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    data = b"intact-bytes"
    pid = payload_id(data)
    store.put(pid, data, media_type="text/plain")
    path = tmp_path / pid[:2] / pid[2:4] / pid
    path.write_bytes(b"corrupted")
    assert store.exists(pid) is False
    with pytest.raises(HashMismatchError):
        store.get(pid)


def test_locate_returns_none_for_corrupt_payload(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    data = b"intact-bytes"
    pid = payload_id(data)
    store.put(pid, data, media_type="text/plain")
    path = tmp_path / pid[:2] / pid[2:4] / pid
    path.write_bytes(b"corrupted")
    assert store.locate(pid) is None


def test_evaluation_report_omits_empty_notes_from_spec_canonical() -> None:
    encoded = spec_canonical_json_bytes(
        EvaluationReport(metrics={"accuracy": 1.0}, n=1, n_missing=0)
    ).decode()
    assert '"notes"' not in encoded


def test_inherited_policy_refs_require_sha256() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        _record(inherited_policy_refs=("not-a-payload-id",))


def test_holdout_single_entity_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two entities"):
        split_entities(
            ["only"],
            SplitSpec(method="holdout", seed=1, test_size=0.2),
        )


def test_as_of_filter_dropping_all_entities_is_rejected() -> None:
    with pytest.raises(ValueError, match="no entities remain"):
        split_entities(
            ["e1"],
            SplitSpec(
                method="holdout",
                seed=1,
                test_size=0.2,
                as_of=datetime(2020, 1, 1, tzinfo=UTC).date(),
            ),
            timestamps=[datetime(2021, 1, 1, tzinfo=UTC).date()],
        )


def test_split_groups_rejects_conflicting_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="conflicting labels"):
        split_groups(
            ["g1", "g1"],
            SplitSpec(method="holdout", seed=1, test_size=0.5),
            labels=["A", "B"],
        )


def test_run_experiment_rejects_single_entity(tmp_path: Path) -> None:
    class _SingleView:
        name = "v"

        def transform(
            self,
            rows: object,
            *,
            as_of: object = None,
        ) -> FeatureTable:
            del rows, as_of
            return FeatureTable(
                entity_ids=("only",),
                columns=("y",),
                values=(("A",),),
                source_payload_ids=(_PID,),
            )

    class _Classifier:
        def fit(self, features: FeatureTable, y: object) -> None:
            del features, y

        def predict(self, features: FeatureTable) -> list[str]:
            return ["A"] * len(features.entity_ids)

    store = LocalStore(tmp_path)
    spec = ExperimentSpec(
        experiment_id="single-entity",
        dataset=DatasetRef(payload_id=_PID),
        features=FeatureSpec(views=("v",)),
        target=TargetSpec(column="y", task="classification"),
        model=ModelSpec(family="x"),
        split=SplitSpec(method="holdout", seed=1, test_size=0.2),
        metrics=(MetricSpec(name="accuracy"),),
        seed=1,
    )
    with pytest.raises(ValueError, match="at least two entities"):
        run_experiment(
            spec,
            rows=[{"id": "only", "y": "A"}],
            feature_views={"v": _SingleView()},
            adapter=_Classifier(),
            store=store,
            run=_run(),
            serialize_model=lambda _model: b"model",
        )


def test_run_experiment_rejects_split_as_of(tmp_path: Path) -> None:
    class _View:
        name = "v"

        def transform(
            self,
            rows: object,
            *,
            as_of: object = None,
        ) -> FeatureTable:
            del as_of
            return FeatureTable(
                entity_ids=("e1", "e2"),
                columns=("y",),
                values=(("A",), ("B",)),
                source_payload_ids=(_PID, _PID),
            )

    class _Classifier:
        def fit(self, features: FeatureTable, y: object) -> None:
            del features, y

        def predict(self, features: FeatureTable) -> list[str]:
            return ["A"] * len(features.entity_ids)

    store = LocalStore(tmp_path)
    spec = ExperimentSpec(
        experiment_id="as-of-footgun",
        dataset=DatasetRef(payload_id=_PID),
        features=FeatureSpec(views=("v",)),
        target=TargetSpec(column="y", task="classification"),
        model=ModelSpec(family="x"),
        split=SplitSpec(
            method="holdout",
            seed=1,
            test_size=0.2,
            as_of=datetime(2026, 1, 1, tzinfo=UTC).date(),
        ),
        metrics=(MetricSpec(name="accuracy"),),
        seed=1,
    )
    with pytest.raises(ValueError, match="does not support split.as_of"):
        run_experiment(
            spec,
            rows=[{"id": "e1", "y": "A"}, {"id": "e2", "y": "B"}],
            feature_views={"v": _View()},
            adapter=_Classifier(),
            store=store,
            run=_run(),
            serialize_model=lambda _model: b"model",
        )


def test_accuracy_rejects_bool_labels() -> None:
    from ds_platform.modeling.evaluate import accuracy

    with pytest.raises(TypeError, match="str or int labels"):
        accuracy([True], [1])


def test_duplicate_inputs_rejected_on_artifact_record() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _record(inputs=(_PID, _PID))


def test_self_referential_related_ref_rejected() -> None:
    from ds_platform.types import RelatedRef, RelationType

    with pytest.raises(ValidationError, match="payload_id"):
        _record(
            related=(RelatedRef(rel=RelationType.CLAIMS_ABOUT, payload_id=_PID),)
        )


def test_kfold_assignments_have_empty_validation_ids() -> None:
    assignments = split_entities(
        ["a", "b", "c", "d"],
        SplitSpec(method="kfold", seed=1, n_splits=2),
    )
    assert all(not assignment.validation_ids for assignment in assignments)


def test_spec_canonical_omits_empty_evidence_on_prediction_row() -> None:
    from ds_platform.modeling.records import PredictionRow

    encoded = spec_canonical_json_bytes(
        PredictionRow(entity_id="e1", y_pred="A")
    ).decode()
    assert '"evidence"' not in encoded


def test_run_experiment_rejects_zero_entities(tmp_path: Path) -> None:
    class _EmptyView:
        name = "v"

        def transform(
            self,
            rows: object,
            *,
            as_of: object = None,
        ) -> FeatureTable:
            del rows, as_of
            return FeatureTable(
                entity_ids=(),
                columns=("y",),
                values=(),
                source_payload_ids=(),
            )

    class _Classifier:
        def fit(self, features: FeatureTable, y: object) -> None:
            del features, y

        def predict(self, features: FeatureTable) -> list[str]:
            return []

    store = LocalStore(tmp_path)
    spec = ExperimentSpec(
        experiment_id="zero-entities",
        dataset=DatasetRef(payload_id=_PID),
        features=FeatureSpec(views=("v",)),
        target=TargetSpec(column="y", task="classification"),
        model=ModelSpec(family="x"),
        split=SplitSpec(method="holdout", seed=1, test_size=0.2),
        metrics=(MetricSpec(name="accuracy"),),
        seed=1,
    )
    with pytest.raises(ValueError, match="at least one entity"):
        run_experiment(
            spec,
            rows=[],
            feature_views={"v": _EmptyView()},
            adapter=_Classifier(),
            store=store,
            run=_run(),
            serialize_model=lambda _model: b"model",
        )


def _sidecar_for_payload(root: Path, target_payload_id: str) -> str:
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith(".tmp-"):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("payload_id") == target_payload_id:
            return path.read_text()
    raise AssertionError(f"sidecar not found for {target_payload_id}")
