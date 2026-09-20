# ds-platform adversarial audit — Pass 2 (post-remediation)

**Date:** 2026-09-19  
**Scope:** `ds-platform` as implemented after the Pass 1 CRITICAL/HIGH remediation.  
**Method:** run the production suite, re-probe Pass 1 findings, construct new cases the suite can stay green on while semantics are still wrong or incomplete.  
**Remediation:** this document is audit-only. No production fixes were applied in this pass.

Pass 1 report (historical, pre-fix findings): [`ds-platform-adversarial-audit.md`](./ds-platform-adversarial-audit.md)

---

## Executive Summary

The platform is **materially safer than Pass 1** and **conditionally sound for early adopters who stay inside the documented v0 contracts**. Identity hashing, store integrity on read, schema-version discipline, frozen models, temporal split, stratify stability, representation kind, split persistence, and eval hardening all landed as intended.

It is **not yet safe to freeze as a durable v0 foundation** without addressing several MEDIUM integrity and orchestration gaps. None of the new findings are as catastrophic as Pass 1’s `record_id` collisions or silent corruption reads, but together they allow **scientifically invalid experiments to complete successfully**, **existence checks to lie about corruption**, and **spec payload identity to drift** if empty-collection omit rules are tightened later.

Do not add a catalog, S3 backend, workflow engine, or model registry to fix this. The missing work is narrow: tighten store existence semantics, validate experiment preconditions, align spec canonical omit rules with artifact rules, and tighten a few envelope fields.

---

## Gate (Pass 2)

| Check | Result |
|-------|--------|
| Production suite | **206 passed** (before Pass 2 audit tests) |
| Pass 2 audit tests | **+15** (document remaining behavior) |
| Combined | **221 passed** |
| Ruff | clean |
| Pyright | 0 errors |
| Adversarial cases attempted | **35** (15 encoded as tests; 20 investigated and classified) |

---

## Pass 1 remediation status

| ID | Pass 1 severity | Status in Pass 2 |
|----|-----------------|------------------|
| C1 | CRITICAL — `created_at` non-injective | **Fixed** — microsecond UTC; naive rejected |
| C2 | CRITICAL — `get()` no verify | **Fixed** — rehash on read; `put()` repairs corrupt keys |
| C3 | CRITICAL — schema evolution trap | **Mostly fixed** — `schema_version=0`, omit empty lists, frozen models; `extra=forbid` still blocks unknown forward fields |
| H1 | HIGH — multi-file artifacts | **Documented** — one blob per `payload_id` (intentional v0) |
| H2 | HIGH — `as_of` was random, not temporal | **Fixed** — `method="temporal"`; `as_of` documented as filter-only |
| H3 | HIGH — stratify order-dependent | **Fixed** — labels sorted by `repr` |
| H4 | HIGH — representation as `dataset` | **Fixed** — `ArtifactKind.REPRESENTATION` |
| H5 | HIGH — eval not self-describing | **Partially fixed** — rejects params/y_proba/NaN; uses spec-float encoding |
| H6 | HIGH — logical_key-only lineage | **Fixed** — `run_experiment` requires `dataset.payload_id` |
| H7 | HIGH — dual canonicalizer drift | **Mostly fixed** — shared datetime form; public `spec_canonical_json_bytes` export |
| H8 | HIGH — mutability | **Fixed** — frozen models + `FrozenJSON`/`FrozenList` |
| H9 | HIGH — `experiment_id` in hash | **Fixed** — excluded from `experiment_config_hash` |
| H10 | HIGH — split not persisted | **Fixed** — `put_split_assignment` in `run_experiment` |
| H12 | HIGH — private `_spec_json` | **Partially fixed** — re-exported from `ds_platform.modeling`; BGA still imports private path |

---

## New findings

### MEDIUM

#### M1. `LocalStore.exists()` / `locate()` do not verify content integrity

1. **Finding.** After on-disk corruption, `exists(payload_id)` is `True` and `locate()` returns a URI. Only `get()` rehashes and raises `HashMismatchError`.
2. **Evidence.** `store.py` `_existing_path` checks file presence only. Pass 2 probe: corrupt file → `exists=True`, `get` → `HashMismatchError`.
3. **Why tests missed it.** Pass 1 regression covers `get()` only.
4. **Affected components.** `LocalStore`, any workflow that gates on `exists()` before lazy read.
5. **Proposed fix.** Optionally verify digest in `exists()`/`locate()`, or document that existence means “path present,” not “bytes match id.” Prefer verify-on-read contract everywhere callers assume integrity.
6. **Must fix before v0?** **yes** (contract clarity at minimum)

#### M2. Spec canonical JSON always serializes empty collections; artifact canonical omits them

1. **Finding.** `spec_canonical_json_bytes` includes `"notes":[]` on `EvaluationReport` and `"evidence":[]` on `PredictionRow`. Artifact `canonical_json_bytes` omits empty lists/tuples and `schema_version=0`.
2. **Evidence.** Pass 2 probe: eval bytes contain `"notes":[]`; artifact sidecar omits `"inputs"` when empty.
3. **Why tests missed it.** Spec and artifact encoders tested separately; no cross-rule parity test.
4. **Affected components.** All modeling payload ids (`evaluation`, `prediction`, `split`, feature tables).
5. **Proposed fix.** Apply the same omit rules in `_spec_json` as in `hashing._omit_canonical_item`, or document spec payloads as a separate identity regime forever.
6. **Must fix before v0?** **yes** if spec payload ids are citation targets

#### M3. `inherited_policy_refs` accepts arbitrary strings

1. **Finding.** `inputs` and `derived_from` use `Sha256Hex`. `inherited_policy_refs` is `Sequence[str]` with no pattern check. Garbage policy ids persist in sidecars.
2. **Evidence.** `ArtifactRecord(inherited_policy_refs=("not-a-hash",))` validates.
3. **Why tests missed it.** No adversarial test for policy ref shape.
4. **Affected components.** `ArtifactRecord`, governance/retention lineage.
5. **Proposed fix.** Either type as `Sha256Hex` (if policies are content-addressed) or add a dedicated `PolicyRef` type with its own pattern.
6. **Must fix before v0?** **yes** if `inherited_policy_refs` is part of the compliance story

#### M4. `run_experiment` completes with empty or unusable train/test sets

1. **Finding.** With zero entities, a single entity, or an `as_of` filter that drops every entity, `run_experiment` still persists the full artifact chain. `fit()` may receive zero rows; `evaluate()` returns `accuracy=0.0` on empty vectors.
2. **Evidence.** Pass 2 probes: `rows=[]` succeeds; one entity → `fit n=0`; `as_of` past all timestamps → empty assignment, no error.
3. **Why tests missed it.** `test_run.py` uses five entities and never asserts minimum train/test counts.
4. **Affected components.** `run_experiment`, `split_entities`, scientific validity of persisted experiments.
5. **Proposed fix.** Require `len(train_ids) >= 1` and `len(test_ids) >= 1` (or explicit opt-in) before fit/predict/persist.
6. **Must fix before v0?** **yes** if `run_experiment` is the v0 supervised contract

#### M5. `run_experiment` is holdout-only; `split.as_of` fails at runtime without timestamps

1. **Finding.** Runner rejects `kfold`/`temporal` upfront. For `split.as_of`, `split_entities` raises `ValueError: timestamps are required when split.as_of is set` because the runner never passes timestamps.
2. **Evidence.** `run.py` line 68–69; Pass 2 probe with `as_of` set.
3. **Why tests missed it.** `test_run_experiment_rejects_kfold_spec` exists; no test for `as_of` footgun.
4. **Affected components.** `run_experiment`, experiment specs that set `as_of` expecting temporal leakage control.
5. **Proposed fix.** Validate at spec construction or runner entry; either wire timestamps from caller rows or reject `as_of` in `run_experiment` with a clear message.
6. **Must fix before v0?** **yes** (fail fast at spec time)

#### M6. `split_groups` silently keeps the first label when group ids repeat

1. **Finding.** Duplicate `group_ids` dedupe by first occurrence; conflicting labels for the same group are dropped without error.
2. **Evidence.** `split.py` `split_groups`; probe `['g1','g1']` with labels `['A','B']` keeps `'A'`.
3. **Why tests missed it.** Dedup behavior tested for ids, not conflicting metadata.
4. **Affected components.** Group-level CV for sequence/event tables.
5. **Proposed fix.** Raise on label/timestamp mismatch for duplicate group ids, or document “first wins.”
6. **Must fix before v0?** **recommended**

#### M7. Built-in metrics use unchecked `==` at the evaluation boundary

1. **Finding.** `accuracy()` returns `1.0` for `y_true=[True], y_pred=[1]`. `run_experiment` rejects bool labels at fit time, but direct `evaluate()` / stored predictions do not.
2. **Evidence.** `evaluate.py` `accuracy`; probe output `1.0`.
3. **Why tests missed it.** `rmse` rejects bool; `accuracy` does not.
4. **Affected components.** `evaluate`, downstream eval artifact interpretation.
5. **Proposed fix.** Reject bool labels in classification metrics, or document Python equality semantics explicitly.
6. **Must fix before v0?** **recommended**

#### M8. Duplicate `inputs` entries and self-referential `related` refs are allowed

1. **Finding.** `ArtifactRecord(inputs=(pid, pid))` validates. `related` may point at the record’s own `payload_id`.
2. **Evidence.** Pass 2 probes.
3. **Why tests missed it.** Lineage tests use well-formed graphs only.
4. **Affected components.** Provenance graph integrity.
5. **Proposed fix.** Enforce uniqueness on `inputs`; optionally forbid self-edges in `related`.
6. **Must fix before v0?** **optional**

---

### LOW / intentional

| ID | Finding | Notes |
|----|---------|-------|
| L1 | K-fold assignments always have empty `validation_ids` | Holdout-style API; document or add val fold later |
| L2 | Multi-file artifacts unspecified | Frozen v0 rule: one byte sequence per artifact |
| L3 | In-memory `source_payload_ids` unconstrained | `require_source_payload_ids()` opt-in at persist |
| L4 | BGA imports `ds_platform.modeling._spec_json` directly | Consumer debt; public export exists |
| L5 | `extra=forbid` blocks forward-compatible unknown fields | Partial C3 fix; needs explicit schema_version bumps |
| L6 | `GeometrySpec` not wired to geometry functions | Config/hash only; harmless if documented |
| L7 | `record_id(model)` vs `record_id_from_bytes(stored)` split | Documented; changing canonical rules still migrates ids |
| L8 | Empty evaluation returns `accuracy=0.0` | Defined but misleading without `n` guard |

---

## Rejected / intentional (Pass 2)

- **`derived_from` non-hash strings** — already rejected via `Sha256Hex` (Pass 2 probe corrected Pass 1 note).
- **Temporal + `as_of` combination** — filter then temporal sort is coherent; empty/single-entity edge cases are the real bug (M4).
- **Private `_spec_json` in platform tests** — acceptable inside package; external consumers should migrate (L4).

---

## Recommended fix order (Pass 2)

1. **M4 + M5** — make `run_experiment` fail fast on invalid splits and unsupported split modes.
2. **M1** — align `exists()`/`locate()` with integrity contract or document explicitly.
3. **M2** — unify empty-collection omit rules across artifact and spec canonicalizers.
4. **M3** — tighten `inherited_policy_refs` typing.
5. **M6–M8** — lineage and metric hygiene.

---

## Audit tests added (Pass 2)

See `tests/audit/test_adversarial_audit.py` — section **Pass 2 — remaining gaps**. These assert current behavior (including bugs) so regressions are visible if fixes land later.
