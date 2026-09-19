# ds-platform adversarial audit

**Date:** 2026-09-19  
**Scope:** `ds-platform` as implemented (`0.0.1`, Python 3.13, Pydantic-only core).  
**Method:** run the existing suite, then construct cases the suite can stay green on while the architecture is still wrong.  
**Remediation:** the CRITICAL and HIGH findings below were fixed in the subsequent implementation pass. `tests/audit/test_adversarial_audit.py` now asserts the repaired contracts. This document remains the historical investigation.

Existing production suite: **175 tests, all green.**  
Audit-only tests: **50, all green** (they assert current behavior, including bugs).  
Combined: **225 passed.**  
Ruff: **clean.**  
Pyright: **0 errors.**  
Adversarial cases attempted: **92** (50 encoded as tests; 42 investigated in code and rejected, confirmed intentional, or classified without a dedicated test).

---

## Executive Summary

The current architecture is **conditionally sound, and not ready to be treated as a durable v0 foundation.**

The expensive boundaries are mostly the right ones. Artifact is the root noun. `payload_id`, `record_id`, and `logical_key` are separated. Location is not identity. Quality, evaluation, and claims are other artifacts. The core package does not import consumer projects or vendor SDKs. Modeling is a separate import with no sklearn/Arrow/MLflow dependency. That design still matches the stated principle: the platform owns reusable semantic contracts; projects own domain semantics.

The implementation is not yet safe to freeze under millions of artifacts. Three problems are structural, not stylistic:

1. **`record_id` is not a faithful function of the logical record.** Canonical datetime encoding drops microseconds, so two annotations that differ only in `created_at` by up to 999 ms share an identity. Naive datetimes are silently treated as UTC.
2. **Content addressing is write-time only.** `LocalStore.get()` returns corrupted bytes without checking `sha256(bytes) == payload_id`. Existing production tests lock this in.
3. **`ArtifactRecord` has no schema version, forbids unknown fields, includes empty lists in the hash, and recomputes `record_id` from a parsed model.** Adding one default-list field later makes every historical `record_id` change on parse-then-rehash. That is an unrecoverable migration if consumers already persisted citations.

Around those sit several HIGH issues that current unit tests cannot see: multi-file artifacts are unspecified; `kind=dataset` is being used for representation vectors; evaluation artifacts are not self-describing; `SplitSpec.as_of` is a random split after a cutoff, not a temporal split; stratified splits depend on first-seen label order; and `board-game-analysis` already imports the private `ds_platform.modeling._spec_json` module.

Do not add a catalog, S3 backend, workflow engine, or model registry to “fix” this. The missing work is to freeze identity, canonicalization, the record schema, and a few modeling artifact kinds before more projects write more bytes.

---

## Findings

### CRITICAL

#### C1. Distinct `created_at` values can share a `record_id`

1. **Finding.** Canonical datetime encoding uses `%Y-%m-%dT%H:%M:%SZ`. Sub-second precision is discarded. Two `ArtifactRecord`s that differ only by microseconds hash identically. Naive datetimes are assigned UTC without conversion.
2. **Evidence.** `ds_platform/hashing.py` `_format_datetime`. Production test `test_canonical_json_formats_datetime_as_utc_seconds_z` documents the format and never checks collisions.
3. **Minimal reproduction.**

   ```python
   first = record(created_at=datetime(2026, 9, 19, 19, 40, 0, 1, tzinfo=UTC))
   second = record(created_at=datetime(2026, 9, 19, 19, 40, 0, 999999, tzinfo=UTC))
   assert first.created_at != second.created_at
   assert record_id(first) == record_id(second)  # True today
   ```

   Also: `datetime(2026, 9, 19, 19, 40, 0)` (naive) hashes equal to the UTC-aware equivalent.
4. **Why existing tests missed it.** They lock the string format, not injectivity of `created_at → record_id`.
5. **Affected components.** `hashing.py`, every `ArtifactRecord`, `put_record`, modeling `put_*` helpers (default `created_at = run.started_at`).
6. **Proposed fix.** Pick one and freeze it: (a) include milliseconds or microseconds in the canonical form, or (b) remove `created_at` from the hashed envelope and treat it as non-identifying metadata. Do not keep second precision *and* keep `created_at` inside `record_id`. Require timezone-aware datetimes; reject naive values.
7. **Must fix before v0?** **yes**

#### C2. `LocalStore.get()` does not verify content identity

1. **Finding.** After a payload is overwritten on disk, `get(payload_id)` returns the corrupted bytes. The caller is told those bytes *are* `payload_id`. `put()` of the original bytes then raises `PayloadConflictError`, so the store cannot self-heal.
2. **Evidence.** `store.py` `get()` is `path.read_bytes()` with no digest check. Production `test_put_does_not_silently_mutate_existing_payload` asserts `store.get(_PID) == b"tampered"`.
3. **Minimal reproduction.** Put `b"intact-bytes"`, overwrite the object file with `b"corrupted"`, `get()` returns `b"corrupted"` and `payload_id(get()) != payload_id`.
4. **Why existing tests missed it.** They test write-once on `put()`, and treat a successful read of tampered bytes as the setup for a conflict, not as a failed integrity check.
5. **Affected components.** `LocalStore`, every later `Store` implementation that copies this pattern, all consumers that trust `get()`.
6. **Proposed fix.** `get()` (and ideally `exists`/`locate` users who then read) must rehash and raise `HashMismatchError` on mismatch. Optionally allow an explicit `overwrite_repair` path; do not make conflict-on-correct-bytes the only outcome.
7. **Must fix before v0?** **yes**

#### C3. `ArtifactRecord` schema is not evolvable without rewriting history

1. **Finding.** There is no `schema_version`. `extra="forbid"` rejects unknown fields. `canonical_json_bytes` omits `None` but **includes empty lists**. `record_id()` hashes a re-serialized model, not the stored bytes. Adding `notes: list[str] = []` later means: old stored JSON omits `notes`; parse fills `[]`; rehash includes `"notes":[]`; every citation of the old `record_id` becomes unreproducible.
2. **Evidence.** `types.py` `ArtifactRecord` / `_CoreModel`. Audit test `test_audit_adding_default_list_field_would_change_rehashed_record_id`. Current round-trip tests only prove stability while the field set is unchanged.
3. **Minimal reproduction.** Canonicalize a record; add `"notes": []` to the dict; hashes differ. That is exactly what a default-list field addition does after parse.
4. **Why existing tests missed it.** `test_put_record_round_trips_canonical_bytes` uses the same library version for write and read.
5. **Affected components.** `ArtifactRecord`, committed JSON Schema, every persisted sidecar, every consumer citation of `record_id`.
6. **Proposed fix.** Before any more records are written as “v0”: add `schema_version: Literal[0] = 0` now; publish a frozen field list; define `record_id` of a stored document as `sha256(stored_bytes)` (`record_id_from_bytes`); treat `record_id(model)` as “hash of this version’s canonical form” only. Adding a field is a new schema version with an explicit migration, not a silent default.
7. **Must fix before v0?** **yes**

---

### HIGH

#### H1. Multi-file artifacts are unspecified

1. **Finding.** There is no manifest, directory artifact, part list, or ordered-file identity. A dataset of `part-000.parquet` + `schema.json` has no platform representation. Consumers will invent incompatible ones (zip the directory, hash a file list, hash only the first part, treat a folder path as identity).
2. **Evidence.** No `Manifest` / `put_directory` in `ds_platform` or `ds_platform.modeling`. Architecture freeze still says “tables as Parquet.” Implementation stores one byte blob per `payload_id`.
3. **Minimal reproduction.** `hasattr(ds_platform, "Manifest") is False`.
4. **Why existing tests missed it.** No multi-file tests exist.
5. **Affected components.** Artifact identity, future S3 layout, BGA/RI corpora if they grow past single JSONL files.
6. **Proposed fix.** For v0, **freeze “one artifact = one byte sequence.”** If a project has many files, it must produce one payload (a zip, a single Parquet, or a JSON manifest *it* hashes as bytes). Do not add a platform manifest until the identity rule is written: sorted relative paths, file bytes only (not mtime/uid), no empty-dir entries, UTF-8 names, relocation-invariant.
7. **Must fix before v0?** **yes** (the freeze/documentation; not an implementation)

#### H2. `SplitSpec.as_of` is a filter plus a random split, not a temporal split

1. **Finding.** Entities with `timestamp <= as_of` are kept, then shuffled with `seed`. Later-kept rows can land in train while earlier rows land in test. `run_experiment` never passes timestamps, so any `split.as_of` raises instead of doing a temporal holdout.
2. **Evidence.** `split.py` `_prepare_entities` + `_random_holdout_test_ids`. Audit: `seed=0`, dates 2026-01-01 / 2026-06-01, `as_of=2026-06-01` → train=`late`, test=`early`. `run_experiment` + `as_of` → `ValueError: timestamps are required`.
3. **Minimal reproduction.** See `tests/audit/test_adversarial_audit.py::test_audit_as_of_is_a_filter_not_a_temporal_split`.
4. **Why existing tests missed it.** `test_split_entities_temporal_cutoff_drops_later_entities` only checks that a *post-cutoff* id is dropped, not that remaining order is temporal.
5. **Affected components.** `SplitSpec`, `split_entities`, `run_experiment`, any consumer that believes `as_of` means “train on the past.”
6. **Proposed fix.** Rename or document `as_of` as `drop_after`. If a temporal method is wanted later, add `method="temporal"` that sorts by timestamp and cuts once. Do not silently keep the current name.
7. **Must fix before v0?** **yes** (semantics/docs; the leakage is already real if anyone uses `as_of` + `split_entities` directly)

#### H3. Stratified splits are not stable under entity reorder

1. **Finding.** Stratification iterates `dict` insertion order of labels and uses `seed + label_index`. The same entity set, same labels, same seed, different first-seen label order → different test sets. `seed=42` does not identify a split without the encounter order.
2. **Evidence.** Audit test with 6 `A` + 6 `B` in opposite block order. `split.py` `_stratified_holdout_test_ids`.
3. **Minimal reproduction.** See `test_audit_stratified_split_depends_on_first_seen_label_order`.
4. **Why existing tests missed it.** Determinism tests reuse the same list order.
5. **Affected components.** `split_entities`, `split_groups`, `run_experiment` (entity order = first feature view order).
6. **Proposed fix.** Sort labels by a stable key (repr, or caller-supplied class order) before seeding. Persist `SplitAssignment` if the split is part of an experiment’s scientific record.
7. **Must fix before v0?** **yes** if stratified splits are part of the v0 contract; otherwise document “order-dependent” and persist assignments

#### H4. Representation vectors are persisted as `kind=dataset`

1. **Finding.** `run_encoding` calls `put_feature_dataset` for representation JSON. The artifact kind is `dataset`. There is no representation kind. `RepresentationTable` itself carries no `family`, encoder version, or `EncodingSpec` hash. Two tables are “compatible” only if `dim` matches and entity ids overlap.
2. **Evidence.** `encode.py` lines that call `put_feature_dataset`. `RepresentationTable` fields. Audit test `test_audit_run_encoding_persists_representations_as_dataset_kind`. BGA already builds on this helper.
3. **Minimal reproduction.** Run `run_encoding`; sidecar `kind == "dataset"`.
4. **Why existing tests missed it.** They check that a payload was written, not that the kind is the right noun.
5. **Affected components.** `run_encoding`, `put_feature_dataset`, artifact-kind vocabulary, every downstream query of “datasets.”
6. **Proposed fix.** Either add `kind=representation` now, or persist encodings as `document`/`index` with an explicit contract, or require callers to pass a kind. Do not keep “vector table = dataset.” Compatibility of two tables should cite `spec_config_hash(EncodingSpec)` plus `dim`.
7. **Must fix before v0?** **yes** (kind choice is expensive after volume)

#### H5. Evaluation artifacts are not self-describing

1. **Finding.** The evaluation payload is `{metrics, n, n_missing, notes}`. The record has `evaluation_of` → one subject payload (usually predictions) and `inputs`. It does not record model id, dataset id, split, metric definitions/versions, or contract. Built-in metrics ignore `MetricSpec.params` and `y_proba` (notes only). `evaluation_report_bytes` emits Python `NaN` (`{"rmse":NaN}`), which is not RFC 8259 JSON.
2. **Evidence.** `records.py` `put_evaluation_artifact` / `evaluation_report_bytes`. `evaluate.py`. Audit tests for payload keys, ignored params, and `NaN`.
3. **Minimal reproduction.** `put_evaluation_artifact(...)`; inspect payload and sidecar.
4. **Why existing tests missed it.** They check `evaluation_of` is set and `accuracy` is present.
5. **Affected components.** `EvaluationReport`, experiment runner, any later eval comparison across projects.
6. **Proposed fix.** Keep the slim payload, but require record-side citations: `evaluation_of` prediction, `inputs` must include model and feature/dataset payload ids when known. Hash metric identity as `name` + canonical params (and then honor those params, or reject unknown params). Use spec-float encoding for report bytes, or reject non-finite metrics.
7. **Must fix before v0?** **yes** for citation completeness and finite JSON; metric-param behavior can wait if params stay rejected

#### H6. `DatasetRef` by logical key contributes no lineage

1. **Finding.** `run_experiment` sets feature/model `inputs` from `spec.dataset.payload_id` only. A spec that names `logical_key="proj:dataset:v0"` persists an experiment whose artifacts do not cite the dataset. Logical keys are not resolved (there is no catalog).
2. **Evidence.** `run.py` `_dataset_inputs`. Audit test `test_audit_logical_key_only_dataset_ref_omits_dataset_from_inputs`.
3. **Minimal reproduction.** `DatasetRef(logical_key="audit:data:v0")` → `_dataset_inputs(spec) == []`.
4. **Why existing tests missed it.** The happy-path experiment uses a payload id.
5. **Affected components.** `ExperimentSpec`, `run_experiment`, provenance walks.
6. **Proposed fix.** Require `payload_id` when using `run_experiment`, or refuse to persist lineage-incomplete experiments. Do not add an alias resolver yet.
7. **Must fix before v0?** **yes** for the runner; `DatasetRef` itself may keep both fields

#### H7. Modeling payloads use a second, leaky canonicalizer

1. **Finding.** Artifact envelopes reject floats. Spec/representation hashing uses IEEE hex tagged `{"$spec_float": "..."}`. Feature tables and evaluation reports use `json.dumps` of raw floats. Prediction JSONL uses `json.dumps` and sorts **encoded lines**, not `entity_id`. A datetime accidentally passed into spec canonicalization is handled as `date` (`datetime` is a `date` subclass) and emits ISO with microseconds and `+00:00`, not `Z`.
2. **Evidence.** `_spec_json.py`, `run.py` `_feature_table_bytes`, `records.py` `evaluation_report_bytes` / `prediction_payload_bytes`. Audit tests.
3. **Minimal reproduction.** Feature table with `1.25` → `b"1.25"` and no `$spec_float`. `spec_canonical_json_bytes({"when": aware_datetime})` contains `.123456+00:00`.
4. **Why existing tests missed it.** Spec tests cover spec floats; record tests cover integer envelopes; nobody hashes a feature table across the two encoders.
5. **Affected components.** Experiment reproducibility, representation payload identity, evaluation payload identity, BGA’s import of `_spec_json`.
6. **Proposed fix.** One rule: envelopes stay float-free; any float-bearing **payload** uses `spec_canonical_json_bytes` (or Parquet, later). Promote `_spec_json` to a public module or stop using it from consumers. Reject `datetime` in the spec encoder or format it like artifact datetimes.
7. **Must fix before v0?** **yes** for the rule; promoting the module is cheap

#### H8. Core records are mutable; frozen specs still share mutable dicts

1. **Finding.** `ArtifactRecord` is not frozen. After `record_id(record)`, `record.inputs.append(...)` changes the next hash. `EvaluationReport.metrics` and `ModelSpec.params` are ordinary dicts on frozen models; in-place mutation changes `experiment_config_hash`.
2. **Evidence.** `_CoreModel` has `extra="forbid"` only. Audit tests for record append and `spec.model.params["n"] = 2`.
3. **Minimal reproduction.** See those two audit tests.
4. **Why existing tests missed it.** They never mutate after hash.
5. **Affected components.** All core types, experiment identity, any cached hash.
6. **Proposed fix.** Freeze `_CoreModel`. Use tuples / `MappingProxyType` for lists and params, or copy-on-validate deep freeze. Nested mutation should fail.
7. **Must fix before v0?** **yes** for `ArtifactRecord`; **yes** for spec params if config hashes are cited

#### H9. `experiment_id` is inside configuration identity

1. **Finding.** Renaming `exp-a` → `exp-b` with identical dataset/model/split/metrics changes `experiment_config_hash`. Code version, environment, and lockfile are *not* in that hash (they belong in `RunContext` if at all). The hash therefore mixes a human label with scientific configuration and still omits the things that actually make a run unreproducible.
2. **Evidence.** `ExperimentSpec` fields; audit rename test. `CodeRef` is optional and unused by `run_experiment` except via the caller-supplied `RunContext`.
3. **Minimal reproduction.** `model_copy(update={"experiment_id": "exp-b"})` changes the hash.
4. **Why existing tests missed it.** They change `ModelSpec.family`, not the id.
5. **Affected components.** `experiment_config_hash`, `RunContext.config_hash`.
6. **Proposed fix.** Hash a dedicated config object that excludes `experiment_id`. Record `experiment_id` on the run or as a logical key. Keep code/env out of config hash; keep them in provenance.
7. **Must fix before v0?** **yes** if consumers will cite `config_hash` as “same experiment”

#### H10. Holdout experiments do not persist the split, and drop labels from the feature artifact

1. **Finding.** `run_experiment` persists features *without* the target column. Stratified splits need labels that are then only present on the test prediction file. Train labels live only inside `serialize_model` bytes. `SplitAssignment` is not written. Reproducing a stratified split from artifacts alone is impossible.
2. **Evidence.** `run.py` `_select_feature_columns` then `put_feature_dataset(_feature_table_bytes(features))`. No `put_*` for assignments.
3. **Minimal reproduction.** Inspect the stored feature JSON after `run_experiment`; no target column; no split artifact.
4. **Why existing tests missed it.** They recompute the split from the original in-memory rows.
5. **Affected components.** Experiment reproducibility, evaluation interpretation.
6. **Proposed fix.** Persist `SplitAssignment` as a `document` (or include it in the prediction payload header). Keep labels out of the feature matrix if desired, but then persist a label column artifact or the assignment.
7. **Must fix before v0?** **no** if `run_experiment` stays “optional glue”; **yes** if it is the supported experiment path

#### H11. Logical keys have no uniqueness, versioning, or resolution

1. **Finding.** The same `logical_key` can label distinct payloads and distinct records in one store. The store has no `resolve` / `latest`. Architecture says uniqueness is per-project and aliases move; implementation has neither uniqueness nor alias history. That is coherent *only if* nobody treats the key as identity. Tests and docs use keys as if they were stable names (`bga:corpus:v0`).
2. **Evidence.** Audit: two `put_feature_dataset` calls with `logical_key="audit:corpus:v0"` and different bytes both succeed.
3. **Minimal reproduction.** That test.
4. **Why existing tests missed it.** They never reuse a key.
5. **Affected components.** `logical_key`, future catalog, champion aliases (`RelationType.CHAMPION_OF` exists; no writer).
6. **Proposed fix.** Do **not** add a resolver now. Document: keys are non-unique labels; collisions are allowed; resolution is a project concern until a catalog exists. Freeze that sentence.
7. **Must fix before v0?** **yes** (documentation / freeze). No catalog.

#### H12. `board-game-analysis` already depends on a private module

1. **Finding.** `board_game_analysis.modeling.perspectives` and `interventions` import `ds_platform.modeling._spec_json`. The leading underscore is not a real boundary. Shrinking or moving that module later is a cross-repo break. The modeling `__all__` has 70+ names; several are convenience functions, not contracts.
2. **Evidence.** Consumer source imports; `modeling/__init__.py` `__all__`.
3. **Minimal reproduction.** Open those BGA files.
4. **Why existing tests missed it.** Platform tests only check that *platform* does not import BGA.
5. **Affected components.** Public API, spec canonicalization, consumer adoption already in progress.
6. **Proposed fix.** Export `spec_canonical_json_bytes` from `ds_platform.modeling` or `ds_platform.hashing`. Treat current `__all__` as larger than the freeze set; mark a subset as v0.
7. **Must fix before v0?** **yes** (export the one function consumers already need)

---

### MEDIUM

#### M1. `align_feature_tables` / `align_representation_tables` silently drop entities

Inner join, first-table order, first-table `source_payload_ids` only. Missing entities and secondary sources disappear. Correct if documented; easy to misread as a union. Existing tests check shape, not dropped ids.

**Must fix before v0?** no — document fail-closed vs inner-join.

#### M2. `seed=42` is not a reproducibility contract

The only `random` use in the package is `split.py` (`random.Random(seed)`). Encoders, models, clustering, and `run_encoding` do not seed adapters. `EncodingSpec.seed` is hashed and otherwise unused by the platform. Stochastic `predict_proba` is invoked twice in `run_experiment`.

**Must fix before v0?** no — document that seed applies to platform splits only.

#### M3. Bool is an `int`, so labels and metrics coerce

`_classification_label(True) is True`. `_regression_label(True) == 1.0`. `rmse([True], [False]) == 1.0`. Feature tables allow `True` and `1` as distinct cells that compare equal.

**Must fix before v0?** no — reject `bool` in label helpers if this shows up in a consumer.

#### M4. `source_payload_ids` are unconstrained strings

`FeatureTable` / `RepresentationTable` accept `"not-a-payload-id"`. Provenance on in-memory tables can be fiction. Envelope `inputs` *are* hash-pattern checked.

**Must fix before v0?** no.

#### M5. Relation vocabulary cannot express generation vs support

`RelationType` is `{quality_for, evaluation_of, claims_about, reviews, champion_of}`. “A generated B” is overloaded onto `inputs` (and unused `derived_from`). “A supports claim B” vs “claims_about” is one-directional. Cycles and self-references are accepted. Duplicate inputs are kept. Modeling helpers never set `derived_from`.

This is a gap, not a reason to grow an ontology. Freeze `inputs` = “consumed by this run,” `derived_from` = unused until defined, typed `related` = the five values only.

**Must fix before v0?** no — document the two list fields or delete `derived_from` before it acquires two meanings.

#### M6. `ExternalRunIds` freezes vendor field names

`dagster` / `mlflow` / `sqlmesh` / `otel_trace` are core schema fields. Adding Prefect later is a schema change (see C3). A small `dict[str, str]` or a list of `{system, id}` would have been cheaper. The *absence* of those SDKs in the import graph is still correct.

**Must fix before v0?** no if you accept a closed peer set; yes if you want open-ended systems.

#### M7. Unicode is not normalized

NFC `café` and NFD `café` are different payloads. Correct for byte identity; surprising for `name` / `logical_key`. Logical keys already restrict to a conservative ASCII pattern, which limits the damage.

**Must fix before v0?** no.

#### M8. kNN is dense O(n²) with NaN-unsafe distances

`pairwise_distances` builds a full matrix. Ties break by `(distance, entity_id)` (deterministic). NaN distances sort in CPython-dependent ways. `k >= n` is rejected. Zero vectors have cosine distance `1.0` (similarity 0). Fine for hundreds of entities; not for 10⁵.

**Must fix before v0?** no. Do not add ANN.

#### M9. `GeometrySpec` is not consumed by geometry functions

`knn(table, k=..., metric=...)` ignores `GeometrySpec`. The spec exists for hashing only. Harmless if callers treat it as config, not as an engine.

**Must fix before v0?** no.

#### M10. Architecture document header is stale

`docs/architecture.md` still says “Implementation has not started.” The code has a store, types, and a modeling package. Future readers will audit the wrong source of truth.

**Must fix before v0?** no (docs hygiene).

---

### LOW

#### L1. `LocalStore` ignores `media_type`

Persisted only on the record. Two puts of the same bytes with different types are idempotent. Intentional, but easy to assume the store remembers type.

#### L2. Prediction JSONL docstring disagrees with the sort key

Docstring: sorted by `entity_id`. Code: sorted JSON text. Usually the same; not a theorem.

#### L3. Crash between `mkstemp` and `replace` leaves `.tmp-*` files

No recovery job. Harmless at laptop scale.

#### L4. Test fixtures use `bga:` / `ri:` / `example_cafe`

Not production contamination. Slightly trains readers to treat those prefixes as platform vocabulary.

#### L5. `Clusterer` has no seed or persistence story

Thin protocol. Fine.

#### L6. `validation_ids` is always empty

Holdout and k-fold never fill it. The field suggests a three-way split the v1 API does not implement.

---

### INTENTIONAL

#### I1. Three-name identity (payload / record / logical key)

1. **What was investigated.** Whether location, filename, or logical key leaked into `payload_id`.
2. **Why it looked suspicious.** Many platforms hash paths. Tests use `bga:corpus:v0` as if it were an id.
3. **Why it is intentional.** `payload_id` is `sha256` of exact bytes. Relocating `local/foo.parquet` → `s3://bucket/foo.parquet` does not change it. `Location` cannot be placed on `ArtifactRecord`. Logical key changes `record_id` only.
4. **What documentation should clarify.** Logical keys are labels, not lookups. See H11.

#### I2. Artifact canonical JSON rejects floats

1. **Investigated.** Whether numeric records could drift across machines.
2. **Suspicious.** Modeling needs floats; rejecting them looks incomplete.
3. **Intentional.** Envelopes stay deterministic. Spec hashing uses bit-stable hex. The bug is using `json.dumps` for some *payloads* (H7), not the envelope rule.
4. **Clarify.** Two encoders, two jobs. Do not “fix” envelopes by allowing JSON numbers.

#### I3. No catalog / no alias resolver

1. **Investigated.** Same logical key, different payloads.
2. **Suspicious.** Looks like a missing unique index.
3. **Intentional.** Architecture deferred the catalog. Implementation is consistent with “no catalog.”
4. **Clarify.** Do not resolve keys in `ds-platform` until a project actually cannot live without it.

#### I4. Write-once objects, shared keyspace for records and payloads

1. **Investigated.** `put_record` stores canonical JSON under `record_id` in the same `Store` as payloads.
2. **Suspicious.** A JSON payload could occupy the same key as a record.
3. **Intentional.** Same bytes, same object. Records are payloads of kind implied by the document, not a second namespace.
4. **Clarify.** Do not introduce `records/` vs `objects/` prefixes as identity. Layout prefixes are an implementation detail.

#### I5. Path traversal is rejected by the id grammar

1. **Investigated.** `../etc/passwd`, uppercase hex, non-hex ids.
2. **Suspicious.** Filesystem stores usually have traversal bugs.
3. **Intentional.** `_SHA256_HEX` is the only path component.
4. **Clarify.** Keep this rule on every backend.

#### I6. Concurrent `put` of the same bytes is last-writer-wins

1. **Investigated.** Two processes writing the same new `payload_id`.
2. **Suspicious.** Race around `is_file()` + `os.replace`.
3. **Intentional / acceptable.** Hash check implies identical bytes (absent SHA-256 collision). `os.replace` is atomic on POSIX. The leftover risk is C2 (read after external corruption), not concurrent identical writes.
4. **Clarify.** Idempotent put is the contract. Do not add locking for this case.

#### I7. No vendor imports; no `Predictor`; no workflow engine

1. **Investigated.** Import graph, `Classifier`/`Regressor` protocols, `run_experiment`.
2. **Suspicious.** Protocols look sklearn-shaped; `ExternalRunIds` names vendors; `run_experiment` looks like an orchestrator.
3. **Intentional.** Protocols are structural and tiny. `run_experiment` is optional holdout glue. Core `dependencies = ["pydantic>=2.10"]`. Forbidden-import tests include `board_game_analysis` and `restaurant_intelligence`.
4. **Clarify.** `run_experiment` is not the modeling architecture. `fit`/`predict` is a consumer adapter shape, not a framework.

#### I8. Quality / evaluation / claims are separate artifacts

1. **Investigated.** Whether reports were inlined on the envelope.
2. **Suspicious.** `related` plus `inputs` plus `derived_from` looks like a proto-god-object.
3. **Intentional.** Envelope stays slim. The unused `derived_from` is the weak part (M5), not the separation.
4. **Clarify.** Keep the slim envelope. Do not start stuffing metrics onto `ArtifactRecord`.

---

## Freeze Candidates

Freeze these *meanings* before more artifacts are written. Implementation may stay small.

| Decision | Freeze as |
| --- | --- |
| Root noun | Artifact = bytes + kind + `payload_id` |
| `payload_id` | `sha256` hex of exact payload bytes; lowercase; no algorithm prefix |
| `record_id` | `sha256` of a versioned canonical envelope; define injectivity of timestamps (C1) |
| `logical_key` | Optional project-namespaced label; **not unique; not resolved** |
| Location | `Location.uri` only; never hashed into identity |
| Write-once | Same id + different bytes is an error; same bytes is idempotent |
| Envelope field set | Versioned, slim, no inlined reports (C3) |
| Artifact kinds | Current twelve **plus an explicit ruling on representations** (H4). Do not reuse `dataset` for vectors. |
| `inputs` | Consumed payload ids for this record’s producing run |
| `related` | Closed set of five `RelationType`s until a second consumer needs a sixth |
| `ContractRef` | `uri` + `schema_hash`; platform does not validate payload bytes |
| Store protocol | `put` / `get` / `exists` / `locate`; `get` verifies hash (C2) |
| Canonical envelope JSON | UTF-8, sorted keys, no whitespace, no floats, no bytes/sets, enums as values, **timezone-aware datetimes with a frozen precision** |
| Spec/float canonicalizer | IEEE binary64 hex tagged object; public |
| Modeling interchange (v0) | In-memory tables are tuples; on-disk table convention remains Parquet *when a project writes a table*, but the library does not implement it yet |
| Split seed | `random.Random(seed)` on **platform splits only**; not adapters |
| Model protocols | Narrow `Classifier` / `Regressor` / `Embedder` / …; no universal `Predictor` |
| Package boundary | Core stays Pydantic-only; modeling stays in `ds_platform.modeling`; no consumer imports |
| Peers | Dagster, MLflow, OpenLineage, OTel, S3 remain external |

---

## Deliberately Swappable

| Choice | Why it must stay replaceable |
| --- | --- |
| `LocalStore` | First backend. S3 (or anything content-addressed) should implement `Store` only. |
| Pydantic | Implementation of types, not the JSON Schema / hash contract. |
| `json.dumps` as the encoder | The **rules** are the contract; the function is not. |
| Python tuples for `FeatureTable` | Convenient now; Arrow later must not change `payload_id` of persisted files. Persist bytes, not Python objects. |
| `run_experiment` / `run_encoding` | Optional glue. Consumers may call helpers directly (BGA already does). |
| `serialize_model` callback | Pickle vs joblib vs safetensors is a project choice; record `media_type` (and later a contract). |
| Built-in metric set | Names are a small vocabulary; implementations can be replaced if identity includes version/params. |
| Exact extra artifact kinds | `trace`, `card`, etc. remain unfrozen. |
| `ExternalRunIds` values | The *systems* are optional; see M6 on field names. |
| Compute engine | Polars / DuckDB / Ray never belong in core. |

---

## Deferred Infrastructure

Do not introduce these until the listed pressure exists.

| Infrastructure | Pressure that would justify it |
| --- | --- |
| Alias / logical-key index (Postgres or a Parquet manifest) | A project cannot find “current champion” without scanning all records, **and** that scan is actually painful (likely >10⁵ records or multi-machine). |
| S3 `Store` | A second machine must read the same objects, or local disk is too small. |
| Hash-verified repair / scrub job | After C2 exists, operational evidence of bitrot. |
| OpenLineage emitter | A project already runs Marquez/DataHub and wants export, not a platform lineage product. |
| OTel SDK | A project already traces jobs; today only store ids. |
| Arrow/Parquet helpers | A table is too large for JSON tuples **and** two projects would otherwise invent incompatible conventions. |
| Catalog UI / metadata DB | Humans cannot navigate artifacts from filenames and notebooks. |
| Approximate nearest neighbors | Geometry on >~5–10k entities, after exact kNN is measured and found wanting. |
| Distributed execution | A job does not fit one machine; still write artifacts through `Store`. |
| Kafka, Kubernetes, feature store, model registry, warehouse | Not justified by this codebase. MLflow/Dagster remain project peers. |

---

## Audit by requested area

### 1. Artifact identity

`payload_id` is byte-faithful: empty, binary, CRLF vs LF, NFC vs NFD, JSON key order, and relocation do not leak into it. Metadata and logical keys change `record_id` only. Floats/sets/bytes are rejected in envelope canonicalization.

Failures: C1 (datetime injectivity), C3 (evolution), naive-as-UTC. Same payload does **not** acquire two payload ids from path or media type. Same payload **can** acquire many `record_id`s (intentional). Same logical record **can** acquire two `record_id`s after a schema add (C3) or if one side uses naive datetimes.

### 2. ArtifactRecord boundary

**In the record (and hashed):** payload id, media type, kind, `produced_by`, `created_at`, optional name/logical key/contract, policy refs, `inputs`, `derived_from`, `related`.

**Not in the record:** location, `record_id`, quality/eval bodies, model bytes.

**Problems:** not frozen (H8); `created_at` and `produced_by.started_at` can diverge; `derived_from` unused; lists can grow without bound (no cycle/size guard); self-inputs allowed; no schema version.

Field-by-field identity (current behavior):

| Field | Changing it changes `record_id`? | Intentional? |
| --- | --- | --- |
| `payload_id` | yes | yes |
| `media_type` | yes | yes |
| `kind` | yes | yes |
| `produced_by.*` | yes | yes (run is part of the annotation) |
| `created_at` | only at whole-second resolution | **no** (C1) |
| `name` / `logical_key` | yes if set | yes |
| `contract_ref` | yes | yes |
| `inherited_policy_refs` | yes (empty list included) | yes, but evolution-hostile |
| `inputs` / `derived_from` / `related` | yes (order-sensitive, duplicates kept) | mostly yes |
| URI / machine / env not on the model | no | yes |

### 3. Logical keys

Pattern: `{project}:{name}[:version]`, lowercase. Uniqueness: none. Reuse: allowed. Overwrite: no (new payload, new record). Confused with identity: only by humans/docs. Version token is syntactic, not a pointer history. No-catalog design is coherent if H11 is written down. Cross-project: `bga:x` and `ri:x` are different strings; the store does not know “project.” Same key + different kind: allowed.

### 4. Store protocol

`put` hashes, write-once, tmp + `fsync` + `replace`, id grammar blocks traversal. `exists`/`locate` do not read bytes. `get` does not verify (C2). Concurrent identical writes are safe. Concurrent *conflicting* bytes for one id are not possible without a hash collision. Partial writes of the final path are avoided; `.tmp-*` orphans are possible. `media_type` is unused by `LocalStore`. S3 boundary is clean: `Store` + `Location.uri`. Do not persist `file://` URIs as if they were portable.

### 5. Multi-file artifacts

Underspecified. **Migration-risk finding (H1).** Ordering, empty dirs, mtimes, nested trees, partial upload, and relocation are all undefined because the type does not exist.

### 6–7. Provenance and relation semantics

A normal `run_experiment` chain is: dataset payload id (sometimes) → feature dataset → model → predictions → evaluation. Representation encoding cites caller `inputs` and then the representation as a *dataset*. Upstream is lost when only a logical key is supplied (H6). Missing upstream artifacts are not checked (store is not a graph). Cycles are representable. Typed relations distinguish quality/eval/claims/reviews/champion. They do **not** distinguish generated-from vs cited-as-evidence vs “same payload, two keys.” `produced_by` is the run; `inputs` is the bag of parent payload ids. That is inspectable without a server **if** the records are at hand.

Ambiguous graphs that current types allow:

- Model M evaluated by E, but E only points at predictions P; M is two hops away and optional.
- Claim set C `claims_about` document D, while quality Q is `quality_for` dataset S derived from D — no typed “Q informs C.”
- Two records with the same `logical_key` both `champion_of` different models.

Do not grow a giant ontology. Define `inputs` and stop using `derived_from` until it has one sentence.

### 8. Contracts

`ContractRef` is `(uri, schema_hash)`. Contract change + same payload → new `record_id`, same `payload_id`. Payload change + same contract → new `payload_id`; platform does not detect schema violation. Cross-project contracts are just URIs. JSON Schema is generated from Pydantic; ODCS is not implemented (alignment is documentary). Contract identity is correctly *not* payload identity.

### 9. Canonical serialization

Envelope rules are explicit and mostly good. Cross-machine risk today: naive datetimes; second truncation; Python `int` unbounded (not a v0 issue); Unicode without NFC (byte-correct). Spec encoder is careful about NaN/Inf/−0. Cross-encoder datetime and payload-float drift are H7. Two supported Python versions are not in play (`requires-python = ">=3.13"` only).

### 10. Schema / version evolution

Nothing is frozen with a version number except the implicit current field set. `extra=forbid` is good for catching mistakes and bad for reading v1 records with a v0 library. Modeling specs have the same trap. Manifests do not exist. **The expensive freeze is C3 + the kind list + the two hash functions.**

### 11. Splits

Entity and group splits exist. Group split de-duplicates first-seen. Temporal is H2. K-fold has empty `validation_ids`. Duplicate entities rejected for entities, dropped for groups. Empty input → empty assignment. Ordering changes assignments (H3). Deterministic **given the same sequence**. Not persisted. Leakage-safe only if the caller’s entity order and labels are leakage-safe.

Another machine gets the same assignment iff it passes the same ordered ids, labels, timestamps, and `SplitSpec`. That is not implied by `seed=42` alone.

### 12. Features

Duplicate entities/columns rejected. Missing column → `KeyError`. Extra columns kept. Column order is data. Row order is data. Dtype `1` vs `1.0` vs `True` are different cells. Nulls allowed. No categorical type. No Arrow. Align is silent inner join (M1). Index is `entity_ids`, not a pandas index.

### 13. Representations

Compatibility = overlapping ids + equal `dim` for distance helpers. No family/version. NaN/Inf allowed. Empty `dim=0` allowed. Duplicate entities rejected. `run_encoding` checks `dim` and entity-id order. Same logical key + different encoder config is two payloads (good) stored as two datasets (H4).

### 14. Model protocol

`Classifier` / `Regressor` are `fit` / `predict` on `FeatureTable`. Not a universal hierarchy. Hidden sklearn assumptions: method names; optional `predict_proba`; `run_experiment` distinguishes tasks via **type hints** on `fit` (`Sequence[float]` vs `Sequence[str | int]`). Untyped adapters pass both checks. `fit` returning `self` vs `None` is ignored. Serialization is caller-supplied. This is coherent as a thin adapter, not as “framework-neutral modeling theory.”

### 15. Evaluation

Empty valid pairs → metrics return `0.0` with `n` still the original length. Length mismatch rejected. Missing labels counted, not imputed. Unseen prediction labels ignored by macro-F1 (labels from `y_true` only). Vector regression is not a first-class path (`rmse` is scalar). Self-description: H5.

### 16. Experiment reproducibility

| Change | Config hash | Run provenance | Artifact payload |
| --- | --- | --- | --- |
| dataset payload id | yes | inputs (if set) | features if rows change |
| dataset logical key only | yes | **no inputs** (H6) | depends on rows passed in |
| features / target / model / split / metrics / seed | yes | via `config_hash` on run | yes if behavior changes |
| `experiment_id` | **yes** (H9) | no | no |
| code version / env / lockfile | no | only if caller fills `CodeRef` / `environment` | model bytes if code changes |
| `serialize_model` implementation | no | no | **yes** (pickle protocol etc.) |

Ambient unrecorded dependencies: Python version, Pydantic version (for in-memory models, not hashes of bytes), adapter code, feature-view implementations, row order, `serialize_model`.

### 17. Randomness

Only `random.Random` in splits. No NumPy. Clustering adapters are consumer-side. Distributed execution would need to keep this RNG or persist assignments.

### 18. Geometry

Exact pairwise, lex tie-break, rejects `k >= n`, zero-vector cosine defined, NaN unordered, O(n²) memory. Do not add ANN.

### 19. Artifact kinds

| Kind | Distinct? | Risk |
| --- | --- | --- |
| `raw` / `document` / `dataset` | yes if dataset means tabular *data* | `dataset` overloaded by encodings (H4) |
| `model` / `prompt` | yes | prompt vs document blur |
| `prediction` / `evaluation` | yes | eval payload too thin (H5) |
| `quality` / `claim_set` / `review` | yes | unused by modeling |
| `run` | weakly defined; `RunContext` is on every record | likely redundant with `produced_by` |
| `index` | “bytes we do not query” | fine if kept dumb |
| missing `representation` | — | H4 |

`run` and `index` can stay if `run` means “a persisted run snapshot” and is not implied by every record. Freeze that sentence or drop `run`.

### 20. Adapter / external-system boundary

No Dagster/SQLMesh/MLflow/boto3/Arrow imports. Leakage points: `ExternalRunIds` field names; `Location` as `file://`; `file://` in consumer-persisted state; `run_experiment` pickle examples in tests; architecture text that promises Parquet while code speaks JSON tuples. Swapping LocalStore → S3 does not require semantic consumer changes **if** consumers never stored `locate()` URIs. Swapping Polars → other engines is already possible because Polars is not in the API. Swapping DuckDB → Athena is not a platform problem.

### 21. Dependencies / packaging

Runtime: Pydantic. Dev: pytest, ruff, pyright. Python 3.13 only. `ds_platform.modeling` import does not pull extra packages. No circular imports found. Public `ds_platform` exports are reasonably tight; `modeling` is not. Optional deps do not exist (good). `_spec_json` is implementation leaked as API (H12).

### 22. Cross-project contamination

**No reverse imports.** `src/ds_platform` has no BGA/RI modules, datasets, or schemas. Consumer names appear in docs, forbidden-import tests, and example logical keys. `example_cafe.schema.json` is a generic composition fixture, not RI code. Modeling abstractions (perspectives, sequences, interventions) are clearly shaped so BGA can use them; they are still domain-free. That is acceptable. The contamination risk is BGA locking private platform APIs (H12), not the platform importing BGA.

### 23. Public API

`ds_platform.__all__` is a plausible v0 surface. `ds_platform.modeling.__all__` is already a kitchen drawer (70+ names). Equivalent concepts: `spec_config_hash` vs `experiment_config_hash`; two align functions; `put_feature_dataset` used for non-datasets. Removable later only if BGA/RI have not imported them — too late for `_spec_json` and several table helpers. Do not grow `__all__`. Publish a freeze subset.

### 24. Immutability

Modeling tables use frozen Pydantic + tuples (good). Core `_CoreModel` is mutable (H8). Nested dicts on frozen specs are mutable (H8). Shared list aliasing on `ArtifactRecord.inputs` is possible.

### 25. Error semantics

| Input | Result |
| --- | --- |
| Wrong kind | accepted (no payload check) |
| Wrong contract | accepted (H, contract section) |
| Wrong representation dim in `run_encoding` | rejected |
| Wrong feature order | accepted; it is data |
| Wrong split / leakage | accepted (H2) |
| Wrong model output length | later zip/index errors, not a typed check |
| Missing provenance | accepted (H6) |
| Duplicate identity (same bytes) | idempotent put |
| Duplicate logical key | accepted (H11) |
| Incompatible schema extra field | rejected |
| Corrupted payload | **silently returned** (C2) |
| Path-traversal id | rejected |
| Unknown metric | rejected |
| Metric params | accepted, ignored |
| Bool labels | accepted, coerced (M3) |

Silent wrong: C1, C2, H2, H3, H5 (`NaN` JSON / ignored params), H6, M1, M3.

### 26. Scale

| Operation | Complexity | 10K | 100K | 1M | 10M |
| --- | --- | --- | --- | --- | --- |
| `put`/`get` by id | O(size) disk | fine | fine | fine | fine if object storage |
| Fanout dirs `ab/cd/hash` | ~n/65536 files/dir | tiny | tiny | ~15/dir | ~150/dir |
| Resolve by logical key | full scan (no API) | annoying | painful | the catalog pressure | requires an index |
| `knn` / pairwise | O(n²) memory | ~0.8 GB floats | ~80 GB | impossible | impossible |
| `FeatureTable` in RAM | Python tuples | uncomfortable if wide | wrong tool | Arrow/Parquet | Arrow/Parquet |
| Lineage walk | follow lists in records | fine | fine if records are local | needs batch fetch | same |

**Next system, if any:** a **logical-key → payload_id index** when alias lookup is real and scanning is slow. Then S3 when disk/locality fails. Not Kafka. Not a metadata platform. Not ANN until kNN is measured.

### 27. Migration-cost decisions (the expensive twenty)

| # | Decision | Current | Why later is expensive | Sound? | Freeze | Swappable | Revisit if |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Payload identity | SHA-256 of bytes | All citations | yes | yes | hash function only under new prefix | SHA-256 is broken |
| 2 | Record identity | SHA-256 of canonical JSON | All audit citations | **not yet** (C1, C3) | after C1/C3 | encoder library | schema version bump |
| 3 | Canonical envelope | sorted JSON, no floats, seconds `Z` | Every sidecar | **partial** | after datetime rule | `json` module | need subsecond or exclude clock |
| 4 | Record schema | unversioned Pydantic model | Millions of sidecars | **no** | version 0 field list | Pydantic | new required concern |
| 5 | Multi-file identity | none | Two incompatible corpus layouts | **no** | “single blob” for v0 | zip/Parquet choice | second project has real parts |
| 6 | Logical keys | non-unique labels | If anyone assumed uniqueness | yes if documented | non-resolution | catalog impl | lookup pain (scale) |
| 7 | Artifact kinds | 12 enums | Kind is on every record | **almost** | after H4 | extra kinds | representation/run clarification |
| 8 | Contract identity | uri + schema hash | Cross-project refs | yes | yes | ODCS later | need compatibility ranges |
| 9 | Provenance | `produced_by` + `inputs` + `related` | Graphs already walk these | **thin** | meanings of the three | OpenLineage export | `derived_from` defined |
| 10 | Relation types | 5 values | Stored on records | yes-small | closed set | add one at a time | a sixth relation is unavoidable |
| 11 | RunContext | required on every record | Every sidecar | yes | fields + timezone | vendor ids (M6) | need completed_at in identity? |
| 12 | Representation identity | vector JSON, kind=dataset | Wrong noun at scale | **no** | kind + encoding hash | encoder impl | H4 fixed |
| 13 | Experiment config hash | whole `ExperimentSpec` | Cited in `config_hash` | **no** (H9) | exclude label | extra params | runner is actually used |
| 14 | Split semantics | order-dependent RNG | Silent leakage | **no** (H2/H3) | document or persist | RNG impl | temporal method added |
| 15 | Model protocol | fit/predict protocols | Consumer adapters | yes-narrow | no Predictor | adapters | two in-repo impls share code |
| 16 | Evaluation semantics | scores JSON + one relation | Comparisons across projects | **thin** (H5) | minimum citations | metric formulas | params become real |
| 17 | Public API | large modeling `__all__` | Import breakage | **too wide** | subset | helpers | H12 |
| 18 | Package/deps | Pydantic, 3.13 | Install surface | yes | no vendor deps | type checker | second runtime dep earned |
| 19 | Storage abstraction | `Store` + LocalStore | Path leakage | yes | protocol + verify | S3 impl | second backend exists |
| 20 | Interchange | architecture: Parquet/Arrow; code: JSON tuples | Consumers persist JSON features | **split-brain** | “v0 library does not write Parquet; projects may” | Parquet helpers | two projects need the same writer |

### 28. “Sounds sophisticated”

| Abstraction | Verdict |
| --- | --- |
| Artifact / three ids / slim record | Protects a real expensive boundary |
| `Store` protocol | Justified; one implementation |
| `Location` | Thin, but it exists to stop URI-as-id |
| `FeatureView` / table types | Real consumer boundary |
| Narrow model protocols | Real; do not grow |
| Dual canonicalizers | Real (floats); currently incomplete (H7) |
| `InterventionQuery` / perspectives / sequences | Used by BGA; domain-free; keep |
| `GeometrySpec` | Config object not wired to functions; harmless |
| `champion_of` without aliases | Slightly early; cheap enum |
| `run_experiment` | Glue, not architecture; keep optional |
| `ExternalRunIds` named fields | Fashionable peer list in the schema; see M6 |
| Catalog / backend family / Predictor / retrieval | Correctly absent |

No abstraction was flagged merely for having a small implementation.

---

## Final Assessment

| Question | Answer |
| --- | --- |
| Any identity bugs? | **Yes.** C1 (record_id collision on sub-second timestamps); naive UTC; C2 (read path ignores identity). Payload hashing of raw bytes is sound. |
| Any canonicalization bugs? | **Yes.** Datetime precision; naive tz; spec encoder treats `datetime` as `date`; feature/eval payloads use non-canonical floats (H7). Dict order, enums, bool-before-int, and envelope float rejection are sound. |
| Any provenance gaps? | **Yes.** H6, unused `derived_from`, eval does not cite model/dataset/split, `run_encoding` kind confusion, no cycle guard. The *shape* (slim record + typed relations) is right. |
| Any schema evolution traps? | **Yes. C3 is the expensive one.** |
| Any leakage? | **Yes, in splits (H2).** Feature/target column stripping in `run_experiment` is careful. Align inner-join can drop rows (M1). |
| Any nondeterminism? | Splits are deterministic **given order**. Stratified splits are order-dependent (H3). `seed` does not control adapters (M2). kNN ties are deterministic. |
| Any mutable-state bugs? | **Yes (H8).** Tables are fine. |
| Any API boundary problems? | Modeling `__all__` too large; `_spec_json` already imported by BGA (H12). Core export list is acceptable. |
| Any hidden framework coupling? | No runtime sklearn/Arrow/MLflow. Soft coupling: `fit`/`predict`/`predict_proba`, type-hint task dispatch, vendor-named `ExternalRunIds`. |
| Any cross-project contamination? | **No** in `src/`. Direction of imports is correct. |
| Any serious scale traps? | kNN O(n²); no logical-key index; in-memory tuple tables. None require new infrastructure now. Catalog is the first *specific* pressure. |
| What must be fixed before v0? | C1, C2, C3, H1 (freeze single-blob), H2 (name/docs), H3 or persist splits, H4 (kind), H5 (citations + finite JSON), H6 (runner inputs), H7 (payload canonical rule + public spec encoder), H8 (freeze records), H9 (config hash), H11 (document non-uniqueness), H12 (export what BGA already uses). |
| What can safely wait? | S3, catalog implementation, Arrow/Parquet writers, OpenLineage, OTel SDK, ANN, extra relation types, shrinking unused modeling helpers, NFC, `derived_from` enforcement, `run_experiment` split persistence if the runner stays optional. |

### Is `ds-platform` ready as a durable v0 foundation?

**No.** It is a **conditionally sound pre-v0**: the architecture should be kept, the dependency boundary is clean, and the existing 175 tests are not lying about the things they cover. They are lying by omission about identity injectivity, read-path integrity, schema evolution, split semantics, and a few kind/hash contracts.

Treat the current commit as a working prototype that two sibling projects may keep using **only if** the CRITICAL items and the freeze list above are closed before anyone calls the record schema “v0” and writes millions of artifacts against it.

### Verification appendix

| Check | Result |
| --- | --- |
| Existing production tests | 175 passed |
| Audit-only tests | 50 passed (`tests/audit/`; not a spec) |
| Full combined suite | 225 passed |
| Ruff | clean (`ruff check`, `ruff format --check`) |
| Pyright | 0 errors, 0 warnings |
| Production code modified | no |
| New infrastructure introduced | no |

Adversarial cases attempted (92): payload/record identity (empty, binary, CRLF, NFC/NFD, JSON order, floats/sets/bytes, microseconds, naive tz, offsets, relocation, logical key, metadata); record mutation, cycles, duplicate inputs, `derived_from`; logical-key reuse and missing resolver; store traversal, uppercase ids, corruption read/write, concurrent identical put, media type; multi-file absence; relation vocabulary; contract change vs payload change; schema extra fields and default-list evolution; spec vs envelope datetime; temporal and stratified splits; `run_experiment` + `as_of`; align drop; bool/int cells; unconstrained source ids; NaN representations; `run_encoding` kind; bool labels; ignored metric params/`y_proba`; mutable report/spec; `experiment_id` hash; logical-key lineage; NaN eval JSON; prediction sort key; feature-table JSON floats; kNN ties/NaN; public-API size; vendor field names; import graph / consumer direction; promised-vs-actual Parquet.
