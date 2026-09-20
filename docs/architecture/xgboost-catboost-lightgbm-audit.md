# Architecture Audit: XGBoost, LightGBM, and CatBoost Integration

**Date:** 2026-09-20  
**Scope:** Architecture audit only — no implementation, no dependency changes, no core modifications.  
**Repository state:** Pre-alpha `ds-platform` v0.0.1; modeling foundation implemented; 235 tests passing at audit time.

---

## Executive Summary

This audit investigated whether XGBoost, LightGBM, and CatBoost should become part of `ds-platform`, what the correct integration boundary is, and whether the current modeling architecture has semantic gaps that must be resolved first.

**Findings in brief:**

1. **These libraries should not enter `ds-platform` core.** The architecture freeze, adversarial audit, and `modeling-architecture-plan-v1.md` all place vendor ML libraries in consumer projects or, at most, future optional extras — never in the default install.

2. **Standard supervised classification and regression can be integrated today without changing core `ds-platform`.** Consumer adapters implementing `Classifier` / `Regressor` on `FeatureTable`, plus existing `run_experiment`, split helpers, evaluation, and artifact record helpers, are sufficient for holdout experiments.

3. **The integration boundary is consumer-owned adapters**, not platform-owned algorithm classes, registries, or `ModelSpec.family` dispatch. Adapters translate `FeatureTable` ↔ library-native inputs (typically NumPy/pandas/Pool) inside the consumer project.

4. **Real semantic gaps exist, but they are adapter-layer concerns, not blockers for core changes.** The largest gap is **feature-type semantics**: `FeatureTable` stores untyped `Scalar` cells with column names but no distinction between numeric, categorical, text, or embedding columns. CatBoost in particular loses exploitable information if naively flattened to a numeric matrix. A second gap is **training-time validation / early stopping**, which all three libraries support via `eval_set` and callbacks but `run_experiment` does not expose (holdout-only, empty `validation_ids`). A third gap is **ranking**, which all three libraries support but the current `Classifier`/`Regressor`/`run_experiment` path cannot express.

5. **No evidence yet justifies a core architectural change** for boosting-library support. Cross-project adapter sharing pressure (two consumers needing identical wrappers) would be the trigger for optional extras — not preemptive platform work.

---

## Current ds-platform Modeling Contract

### Package and dependency architecture

| Aspect | Current state |
| --- | --- |
| Runtime dependency | `pydantic>=2.10` only |
| Optional dependency groups | **None** (only `[dependency-groups] dev`) |
| Vendor imports in `src/` | **None** (no sklearn, numpy, xgboost, lightgbm, catboost) |
| Public modeling import | `import ds_platform.modeling` — 70+ exports, vendor-free |
| Forbidden-import tests | `tests/modeling/test_import_graph.py` blocks xgboost, lightgbm, catboost, sklearn, numpy, etc. |
| Lockfile | `uv.lock` exists; CI uses `--locked` |

### Experiment boundary (`run_experiment`)

`run_experiment()` is deliberately thin holdout orchestration for scalar supervised tasks:

```
ExperimentSpec + rows + feature_views + adapter + store + run + serialize_model
    → materialize FeatureViews → align → extract labels
    → split_entities (holdout only) → train/test slice
    → adapter.fit(train) → adapter.predict(test)
    → persist feature dataset, split assignment, model, predictions, evaluation
    → ExperimentResult
```

**Hard constraints enforced by the runner:**

- `split.method == "holdout"` only (k-fold, temporal rejected)
- `split.as_of` not supported (caller must filter rows)
- `dataset.payload_id` required for lineage
- Non-empty train and test assignments required
- `ExperimentSpec.seed` must equal `SplitSpec.seed` (split RNG only)
- Model serialization via caller-supplied `serialize_model(adapter) → bytes`
- Optional `predict_proba` via duck-typing (`getattr(adapter, "predict_proba")`)

**What the runner does not do:**

- Instantiate models from `ModelSpec.family` (family is descriptive JSON only)
- Pass validation data or `eval_set` to adapters
- Control adapter random seeds (`ModelSpec.params` is hashed but not applied)
- Import or know about any ML library
- Support ranking, survival, or clustering (separate paths exist for clustering via `run_clustering`)

**Assumptions relevant to gradient-boosted trees:**

- Adapters accept `FeatureTable` with `fit(features, y)` and `predict(features)`
- Labels are `str | int` (classification) or `float` (regression) extracted from a column
- Feature matrix is rectangular: entity_ids × named columns of `Scalar` cells
- Missing values (`None`) are valid cells in `FeatureTable` but evaluation excludes pairs where y_true or y_pred is None
- Task discrimination uses **type hints on `adapter.fit`'s `y` parameter** (sklearn-shaped convention, not sklearn dependency)

These assumptions are **generic enough for GBDT classifiers/regressors** when wrapped in consumer adapters. They are **not generic enough** for ranking, native CatBoost Pool workflows with text/embeddings, or early-stopping workflows that need a platform-managed validation split.

### Capability boundary

| Protocol | Semantic contract | sklearn-shaped? |
| --- | --- | --- |
| `Classifier` | `fit(FeatureTable, Sequence[str\|int])`, `predict(FeatureTable) → Sequence[str\|int]` | Method names yes; no sklearn import |
| `Regressor` | `fit(FeatureTable, Sequence[float])`, `predict(FeatureTable) → Sequence[float]` | Same |
| `Embedder` | `fit/encode` on opaque records → `RepresentationTable` | No |
| `Clusterer` | `fit/predict` on `RepresentationTable` | No |

**Protocol sufficiency for GBDT:**

- `fit` / `predict` on `FeatureTable` is sufficient for **point-prediction** classification and regression.
- `predict_proba` is **not** in the protocol but is optionally invoked by `run_experiment` via duck-typing.
- Feature importance, staged prediction, early-stopping state, evaluation history, native booster access, and ranking scores are **intentionally outside** the protocol.
- Model lifecycle (early stopping, eval sets, callbacks) belongs in the adapter, not the platform.

**Hidden constraints from signatures:**

- Classification labels must be `str | int` (not float-encoded categories unless cast)
- Regression labels must be numeric (`int | float`, not bool)
- Adapters with untyped `y` hints pass both classification and regression checks (documented audit finding)

### Feature boundary

`FeatureTable` is an in-memory rectangular matrix:

- **Rows:** opaque `entity_ids` (unique)
- **Columns:** unique names, fixed order
- **Cells:** `Scalar = str | int | float | bool | None`
- **Lineage:** per-row `source_payload_ids` (SHA-256 hex)
- **No column type metadata** (numeric vs categorical vs text vs embedding)
- **No sparse representation**
- **No feature store, registry, or serving**

`FeatureView` is a named, deterministic `rows → FeatureTable` transform. `RepresentationTable` is a separate dense float-vector plane for embeddings/clustering.

**Distinction:**

| Layer | What it represents |
| --- | --- |
| Semantic feature representation | `FeatureTable` — named columns, typed cells at Scalar granularity |
| Library-specific input | NumPy array, pandas DataFrame with dtypes, LightGBM Dataset, CatBoost Pool — **adapter responsibility** |

The platform deliberately stops at `FeatureTable`. Converting to library inputs is not a platform concern today.

### Split boundary

`split_entities`, `split_groups`, `apply_split`, `apply_representation_split` support holdout, k-fold, temporal, stratified, and group isolation. `SplitAssignment` includes optional `method`, `seed`, `fold_index` metadata and can be persisted via `put_split_assignment`.

**Gap for GBDT:** holdout assignments have **empty `validation_ids`**. Consumers needing early stopping must either (a) split validation inside the adapter, (b) use split helpers directly with a custom three-way assignment, or (c) hold out validation before calling `run_experiment`.

### Evaluation boundary

`evaluate()` is model-free. Built-in metrics: `accuracy`, `macro_f1`, `rmse`, `mae`. Consumer metrics via `extra_metrics` mapping. Produces `EvaluationReport` → `put_evaluation_artifact`.

Ranking metrics (NDCG, MAP), calibration metrics, and Poisson deviance are **not** in core — consumers add via `extra_metrics`.

### Artifact / provenance boundary

Lifecycle preserved by existing helpers:

```
FeatureView → FeatureTable (persisted as kind=dataset)
→ SplitAssignment (kind=document)
→ model bytes (kind=model, caller serialize_model)
→ predictions (kind=prediction)
→ EvaluationReport (kind=evaluation, evaluation_of prediction)
```

- **Payload identity:** SHA-256 of bytes (content-addressed)
- **Record identity:** SHA-256 of canonical envelope JSON
- **Config hash:** `experiment_config_hash(spec)` on `RunContext.config_hash`
- **External correlation:** `ExternalRunRef(system, run_id)` on `RunContext`
- **No model version/library metadata** on `ArtifactRecord` beyond `media_type` and lineage `inputs`

Serialization format, library version, and feature schema at training time are **caller/adapter responsibility**, recorded only indirectly via model bytes and feature dataset artifact.

---

## XGBoost Findings

**Sources:** [XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), [Python API reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [Categorical data tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html), [Model IO tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)

### API surface

| Area | Finding |
| --- | --- |
| Classifier | `XGBClassifier` — sklearn-compatible |
| Regressor | `XGBRegressor` — sklearn-compatible |
| Ranking | Native + sklearn path; survival sklearn interface in progress |
| Native API | `DMatrix`, `xgb.train`, `Booster` — richer than sklearn wrapper |

### Fit / predict semantics

- `fit(X, y, eval_set=[(X_val, y_val)], ...)`
- Early stopping via `early_stopping_rounds` param or `xgboost.callback.EarlyStopping`
- Requires **manual** train/validation split for early stopping (no internal CV)
- `predict` uses `best_iteration` automatically when early stopping enabled
- Multiclass: handled via objective; `predict_proba` available on sklearn wrapper

### Categorical features

- Sklearn path: pandas/cudf DataFrame with `category` dtype + `enable_categorical=True` + `tree_method="hist"` or `"approx"`
- Native path: `feature_types` with `"c"` for categorical; experimental re-coder for native interface
- Pre-encoded ordinal integers also supported via `feature_types` without DataFrame categories
- **Information loss if FeatureTable → plain float matrix without category metadata**

### Missing values

- Native missing value support in tree methods
- NaN typically accepted; behavior depends on `missing` parameter and tree method

### Feature names / ordering

- Sklearn wrapper tracks `feature_names_in_` when fit on DataFrame
- Column order matters; names preserved through sklearn API
- Adapter must preserve `FeatureTable.columns` order when building input matrix

### Serialization

- **Recommended:** `save_model("model.json")` or `.ubj` — portable across XGBoost versions (forward-compatible)
- Pickle supported but version/Python/environment dependent; security risk on untrusted input
- JSON/UBJ stores trees + objective; custom objective functions may need re-supply on load

### Semantic mismatches with ds-platform

| Mismatch | Severity |
| --- | --- |
| `FeatureTable` has no categorical dtype → must encode in adapter or use pandas category column | Adapter concern |
| Early stopping needs eval_set not provided by `run_experiment` | Adapter or custom split |
| `ModelSpec.params` not auto-applied to estimator | Adapter concern |
| Ranking needs query/group structure absent from protocols | Out of scope for current runner |
| Native Booster features (cached DMatrix predict) not in protocol | Adapter-specific |

---

## LightGBM Findings

**Sources:** [LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html), [LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html), [early_stopping callback](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.html)

### API surface

| Area | Finding |
| --- | --- |
| Classifier | `LGBMClassifier` |
| Regressor | `LGBMRegressor` |
| Ranking | `LGBMRanker` with `group` parameter in `fit` |
| Native API | `lgb.Dataset`, `lgb.train`, `Booster` |

### Fit / predict semantics

- `fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20)])`
- Early stopping via **callbacks** (not constructor param in current API)
- Post-fit attributes: `best_iteration_`, `best_score_`, `evals_result_`, `n_estimators_`
- `predict` respects best iteration when early stopping used

### Categorical features

- `categorical_feature` in `fit`: list of column indices or names, or `'auto'` for pandas unordered categorical columns
- Values cast to int32; large integer codes problematic; negative values treated as missing
- **Different from XGBoost:** expects integer-coded categoricals or pandas categorical, not necessarily `category` dtype alone
- Cannot monotonically constrain output w.r.t. categorical features

### Missing values

- Native missing value handling in tree splits
- NaN accepted for numeric features

### Feature names / ordering

- `feature_name_`, `feature_names_in_` after fit
- Column order preserved from input

### Serialization

- `booster_.save_model("model.txt")` — text format, portable
- Pickle of sklearn wrapper common but environment-dependent
- Native format preferred for cross-version stability

### Semantic mismatches with ds-platform

| Mismatch | Severity |
| --- | --- |
| Categorical encoding assumptions differ from XGBoost and CatBoost | Adapter must own encoding strategy |
| Early stopping requires callbacks + eval_set | Outside `run_experiment` |
| `LGBMRanker` needs `group` array | Cannot use `run_experiment`; needs Ranker protocol or custom runner |
| String categorical values in FeatureTable must be encoded before LightGBM | Adapter concern |

LightGBM is **not identical** to XGBoost despite both being GBDT: categorical handling, early stopping API, and ranking interfaces differ materially.

---

## CatBoost Findings

**Sources:** [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier), [Pool](https://catboost.ai/docs/en/concepts/python-reference_pool), [fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit), [save_model](https://catboost.ai/docs/en/concepts/python-reference_catboost_save_model)

### API surface

| Area | Finding |
| --- | --- |
| Classifier | `CatBoostClassifier` — sklearn-compatible |
| Regressor | `CatBoostRegressor` |
| Ranking | `CatBoostRanker` |
| Native container | `Pool` with `cat_features`, `text_features`, `embedding_features` |

### Richer input semantics (critical)

CatBoost accepts, via `Pool` or constructor/fit params:

- **Categorical features** — by index or name; native handling without manual one-hot
- **Text features** — raw text columns with built-in text processing
- **Embedding features** — fixed-dimension embedding columns
- **Group/query data** for ranking
- **Sample weights, baselines, timestamps**

Sklearn-style `fit(X, y, cat_features=[...])` works on 2D matrices/DataFrames but **loses text and embedding semantics** unless `Pool` is used.

### Fit / predict semantics

- `fit(X, y, eval_set=..., early_stopping_rounds=..., cat_features=..., text_features=...)`
- Early stopping via `early_stopping_rounds` and `use_best_model`
- `predict`, `predict_proba` with various prediction types
- Feature importance, object importance (SHAP-like) — post-hoc, not part of fit/predict contract

### Missing values

- Native missing value treatment; CatBoost handles NaN/None in numeric and categorical columns

### Serialization

- **Default:** `save_model("model.cbm")` — binary CatBoost format
- **JSON:** `save_model("model.json", format="json", pool=pool)` — pool required for categorical models to be applicable
- JSON without pool is "reviewable but not applicable" for categorical models
- Cross-language loading (Java, Rust) with limitations on categorical/text features

### FeatureView → numpy → CatBoost path

**Would lose meaningful information:**

| Feature kind in domain | Loss on naive numeric matrix |
| --- | --- |
| Raw categorical strings | CatBoost can handle via `cat_features`, but adapter must declare indices/names |
| Text columns | **Lost** unless adapter routes to `text_features` or pre-processes |
| Embedding columns | **Lost** unless adapter uses `embedding_features` or flattens to numeric |
| Ordered vs unordered category semantics | Lost without explicit adapter mapping |

**Conclusion:** CatBoost's advantage (ordered target statistics for categoricals, native text) requires **adapter knowledge of column semantics**, not a platform type change — but the platform provides no standard way to express those semantics in `FeatureTable` today.

### Semantic mismatches with ds-platform

| Mismatch | Severity |
| --- | --- |
| No column kind in FeatureTable | **Highest among the three libraries** |
| Pool-based training not representable in Classifier protocol | Adapter wraps Pool internally |
| Text/embedding features have no platform representation | Consumer FeatureView or RepresentationTable + adapter |
| JSON model export needs training Pool for categorical models | Serialization strategy is adapter-owned |

---

## Cross-Library Comparison

Investigative comparison — not a recommendation or ranking.

| Concern | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| Classifier | `XGBClassifier` | `LGBMClassifier` | `CatBoostClassifier` |
| Regressor | `XGBRegressor` | `LGBMRegressor` | `CatBoostRegressor` |
| Categorical features | pandas `category` + `enable_categorical`; or ordinal + `feature_types` | `categorical_feature` list; int32 codes; pandas auto | `cat_features` on Pool/fit; native ordered encoding |
| Missing values | Native NaN handling | Native NaN handling | Native handling |
| Feature names | `feature_names_in_` (sklearn) | `feature_names_in_` | Via DataFrame columns or `feature_names` on Pool |
| Feature ordering | Significant — splits reference column index | Significant | Significant |
| Probability prediction | `predict_proba` | `predict_proba` | `predict_proba` |
| Early stopping | `early_stopping_rounds` or callback | `lgb.early_stopping()` callback | `early_stopping_rounds` + `use_best_model` |
| Evaluation sets | `eval_set` in `fit` | `eval_set` in `fit` | `eval_set` in `fit` |
| Callbacks | `xgboost.callback.*` | `lightgbm.callback.*` | Limited compared to XGB/LGB |
| Serialization | JSON/UBJ (portable); pickle (fragile) | Text model file; pickle | CBM binary; JSON (+ Pool for categorical) |
| Feature importance | `feature_importances_`, native | `feature_importances_`, native | Multiple importance types |
| Ranking | XGBRanker / native | LGBMRanker + `group` | CatBoostRanker + group_id |
| Richer input types | Mostly numeric + categorical | Numeric + categorical | Numeric + categorical + **text** + **embeddings** |
| sklearn compatibility | Strong sklearn wrapper | Strong sklearn wrapper | Strong sklearn wrapper + richer native Pool API |
| Native API differences | DMatrix, xgb.train | lgb.Dataset, lgb.train | Pool-centric |
| Reproducibility concerns | `random_state`, `n_jobs`, GPU, thread count, version | `random_state`, `num_threads`, GPU, version | `random_seed`, `thread_count`, GPU, version |

---

## Feature Representation Findings

### What FeatureTable can represent today

| Input kind | Representable? | Notes |
| --- | --- | --- |
| Numeric columns | Yes | `int`, `float` cells |
| Categorical columns | Partially | `str` or `int` cells, but **no `kind=categorical` metadata** |
| Missing values | Yes | `None` cells |
| Strings | Yes | As opaque strings — indistinguishable from categorical or text |
| Encoded categories | Yes | Integer codes in cells |
| Boolean | Yes | Distinct from int/float in FeatureTable validation |
| Embeddings | No (in FeatureTable) | Use `RepresentationTable` (dense float vectors) |
| Text | Partially | String cells only; no text-feature semantics |
| Mixed types | Yes | Per-cell Scalar typing |
| Feature names | Yes | `columns` tuple |
| Feature ordering | Yes | Column order is data |

### Can all three libraries consume the same semantic representation?

**No — not without adapter-side decisions.**

| Translation path | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| FeatureTable → float matrix, strings ordinal-encoded | Works | Works | Works but wastes CatBoost native categorical |
| FeatureTable → pandas DataFrame, category dtype | Works with `enable_categorical` | Works with `categorical_feature='auto'` | Works with `cat_features` |
| FeatureTable → Pool with cat/text/embedding indices | N/A | N/A | **Best for CatBoost**; requires adapter column-kind map |
| RepresentationTable → numeric features only | Works (via `as_feature_table`) | Works | Works for numeric; embeddings need explicit handling |

**Where information loss occurs:**

1. **Platform layer:** FeatureTable treats all string columns identically — adapter must know which are categorical vs text.
2. **Adapter layer:** If adapter blindly one-hot encodes everything, all three libraries lose native categorical handling.
3. **Runner layer:** `run_experiment` persists feature dataset as JSON FeatureTable — good for audit, but does not record adapter encoding choices unless model bytes or separate metadata capture them.

### Recommendation boundary (audit only)

Column-kind metadata **could** eventually live in consumer FeatureViews (e.g., a view that tags columns in naming convention or companion metadata) without changing `FeatureTable`. A core change would only be justified if **multiple consumers** need a shared, hashable feature-schema artifact — that is not current pressure.

---

## Protocol Findings

### Sufficient for GBDT classification/regression (point prediction)

| Capability | In protocol? | Should be in protocol? |
| --- | --- | --- |
| `fit` / `predict` | Yes | Yes — current boundary is correct |
| `predict_proba` | Duck-typed optional | No — keep adapter-specific |
| Decision scores / margin | No | No — adapter-specific |
| Feature importance | No | No — post-hoc analysis artifact |
| Early stopping state | No | No — training internals |
| Native model access | No | No |
| Categorical metadata | No | **Maybe later** as feature-schema artifact, not protocol method |
| Ranking / group queries | No | **Separate Ranker protocol** if cross-project need emerges |
| Calibration | No | No — evaluation concern |

### Classifier/Regressor accidentally assume sklearn?

**Soft coupling, not hard dependency:**

- Method names `fit`, `predict`, optional `predict_proba` mirror sklearn conventions
- Task routing via type hints on `y` is sklearn-adjacent
- No requirement for `BaseEstimator`, `n_features_in_`, or numpy inputs
- Any adapter that accepts `FeatureTable` and returns sequences satisfies the protocol

GBDT sklearn wrappers fit this shape when wrapped once more to accept `FeatureTable`.

### Should a capability be added for boosting libraries?

**Not preemptively.** The audit found no behavior that must be in a generic protocol beyond what adapters already encapsulate:

1. **Genuinely shared across projects at protocol level:** `fit(FeatureTable, y)`, `predict(FeatureTable)` — already present.
2. **Adapter-specific:** eval_set wiring, callbacks, categorical encoding, Pool construction, native serialization.
3. **Library-specific:** CatBoost text features, XGBoost DMatrix caching, LightGBM Dataset construction.
4. **Does not belong in ds-platform:** hyperparameter search, GPU dispatch, feature importance computation.

---

## Experiment Runner Findings

### Is `run_experiment` appropriate for GBDT?

**Yes, for v1 holdout classification/regression experiments**, with consumer adapters handling:

- FeatureTable → library matrix conversion
- Mapping `ModelSpec.params` to estimator constructor params
- Mapping `ExperimentSpec.seed` / `ModelSpec.params` to library random_state
- Internal validation split if early stopping desired
- Native serialization in `serialize_model`

### Runner gaps that affect GBDT workflows

| Gap | Impact | Workaround |
| --- | --- | --- |
| Holdout only | Cannot k-fold via runner | Use `split_entities` + manual loop |
| Empty `validation_ids` | No platform validation fold for early stopping | Adapter-internal val split or custom three-way split before runner |
| No `eval_set` hook | Runner cannot pass val data to adapter | Adapter holds val split logic |
| Train on train only, predict on test | Correct for honest evaluation | Adapters must not leak test into fit |
| `predict_proba` optional | Fine for metrics needing probabilities | Implement on adapter |
| No sample weights | Cannot pass weights through runner | Custom experiment loop |

**Verdict:** `run_experiment` remains appropriate as optional glue. GBDT libraries do not require runner changes for basic experiments. Early stopping and validation are **adapter/training concerns**, consistent with architecture freeze ("platform does not own training loops").

---

## Artifact / Serialization Findings

### Current model artifact contract

```python
put_model_artifact(
    store, serialize_model(adapter), run=..., inputs=[...], media_type=...
)
```

- Payload identity = SHA-256 of bytes
- **Any bytes allowed** — pickle, JSON, CBM, native text model, etc.
- `media_type` is declarative (e.g., `application/octet-stream`, `application/json`)
- **No enforced schema** for model payload
- **No library name/version field** on ArtifactRecord
- Feature schema at training time recoverable from `inputs` lineage → feature dataset artifact (FeatureTable JSON)
- Loading not implemented in platform — consumers deserialize

### Can lifecycle represent GBDT models faithfully?

| Aspect | Representable? | Gap |
| --- | --- | --- |
| Model bytes as artifact | Yes | None |
| Hyperparameters in config hash | Yes | Via `ModelSpec.params` in ExperimentSpec hash |
| Feature names/order | Partial | Feature dataset artifact has columns; adapter encoding not recorded |
| Library/version | No | Not on RunContext or ArtifactRecord unless caller adds to `media_type` or separate document artifact |
| Early stopping config | No | Not in ExperimentSpec |
| Training/eval split used for early stopping | No | Unless persisted as separate split assignment |
| Provenance chain | Yes | inputs link model → features → split → dataset |
| Cross-environment load | Depends on serialization format | Native formats (JSON/UBJ/CBM/text) better than pickle |

### Serialization preferences by library

| Library | Portable format | Fragile format |
| --- | --- | --- |
| XGBoost | `.json`, `.ubj` via `save_model` | pickle |
| LightGBM | text model via `booster_.save_model` | pickle |
| CatBoost | `.cbm`; JSON with Pool for categorical | pickle |

**Security:** pickle of estimators is supported by tests (`test_lifecycle.py`) but carries untrusted-deserialization risk — consumers should prefer native model formats for durable artifacts.

---

## Reproducibility Findings

### What ds-platform guarantees

| Property | Guaranteed? | Mechanism |
| --- | --- | --- |
| Experiment config identity | Yes | `experiment_config_hash(spec)` — deterministic canonical JSON |
| Split reproducibility given same entity order | Yes | `random.Random(split.seed)` in `split.py` |
| Artifact payload identity | Yes | SHA-256 of bytes |
| Lineage | Yes | `ArtifactRecord.inputs`, `RunContext.config_hash` |
| Code version | Optional | `RunContext.code_ref` if caller sets |
| External run correlation | Optional | `ExternalRunRef` |

### What ds-platform does NOT guarantee

| Property | Notes |
| --- | --- |
| Same model bytes across XGBoost/LightGBM/CatBoost versions | Library-version dependent |
| Same predictions CPU vs GPU | Libraries may differ slightly |
| Same predictions across thread counts | Possible floating-point nondeterminism |
| Booster randomness from `ExperimentSpec.seed` | **Not applied by platform** — adapter must map seed to `random_state` |
| Same split across machines if entity order differs | Split seed alone insufficient |
| Categorical encoding reproducibility | Adapter responsibility |
| Identical pickle hashes across environments | Not guaranteed |

### Cross-library reproducibility notes

- All three accept `random_state` / `random_seed` — adapter must wire from `ModelSpec.params` or `ExperimentSpec.seed`
- GPU training may introduce nondeterminism unless library-specific deterministic flags set
- CatBoost JSON export for categorical models requires original Pool metadata for full applicability

**Verdict:** `RunContext` + artifact model are **sufficient** for recording what was intended and what was produced. They do **not** guarantee bitwise-identical retraining — that remains a consumer/MLflow concern.

---

## Dependency Findings

### Current packaging architecture permits

| Option | Permitted? | Precedent |
| --- | --- | --- |
| Keep libraries out of ds-platform entirely | Yes | Current state; architecture freeze |
| Consumer projects declare xgboost/lightgbm/catboost | Yes | Intended path per modeling plan |
| Optional extras e.g. `ds-platform[xgboost]` | Technically yes | **No extras exist yet**; plan says "after cross-project pressure" |
| Core dependency on any boosting library | **No** | Violates freeze and forbidden-import tests |

### Transitive dependency concerns

| Library | Notable transitive deps | Platform concern |
| --- | --- | --- |
| XGBoost | numpy, scipy | Would pull numeric stack into optional extra, not core |
| LightGBM | numpy | Same |
| CatBoost | numpy, plotting tools (optional) | Heavier; often larger install |

All three have **materially different** install sizes and platform binaries (CPU/GPU builds). They should remain **optional** if ever added to this repo.

### sklearn as hidden dependency

GBDT sklearn wrappers do **not** require sklearn as a pip dependency for XGBoost/LightGBM/CatBoost themselves — they implement sklearn-compatible interfaces. However, **consumer adapters often use sklearn** for metrics, preprocessing, or `train_test_split`. That sklearn usage stays in consumer projects.

---

## Integration Boundary Findings

### Where the libraries naturally belong

```
restaurant-intelligence / board-game-analysis
    modeling/
        views/           ← FeatureView implementations (column semantics)
        adapters/
            xgboost.py   ← XGBClassifierWrapper(Classifier)
            lightgbm.py
            catboost.py  ← may use Pool internally
        run_*.py         ← ExperimentSpec + run_experiment calls
```

**Not in ds-platform core:**

- Estimator instantiation
- FeatureTable → numpy/pandas/Pool conversion
- Early stopping / callback wiring
- Native save_model serialization
- Library version pinning

### Smallest legitimate integration boundary (derived from audit)

1. **Core remains unchanged** — protocols, FeatureTable, ExperimentSpec, artifact helpers, run_experiment.
2. **Consumer adapter** implements `Classifier` or `Regressor`:
   - Accepts `FeatureTable`
   - Converts to library input internally
   - Applies `ModelSpec.params` to estimator
   - Handles missing values and categorical encoding per library
   - Optionally implements `predict_proba`
3. **Consumer `serialize_model`** uses native format (e.g., `lambda m: m.save_model_to_buffer()`).
4. **Optional future extra** (`ds-platform[boosting]`) only if two consumers duplicate identical adapter code — shares adapters, not algorithms in core.

No registry, no `ModelSpec.family` dispatch, no platform `XGBoostModel` class.

### Existing extension points in codebase

| Extension point | Location | Ready for GBDT? |
| --- | --- | --- |
| `Classifier` / `Regressor` protocols | `capabilities.py` | Yes |
| `run_experiment(..., adapter=...)` | `run.py` | Yes (holdout) |
| `serialize_model` callback | `run.py` | Yes |
| `ModelSpec.params` | `spec.py` | Yes (JSON hyperparams) |
| `put_model_artifact` | `records.py` | Yes |
| `evaluate(..., extra_metrics=...)` | `evaluate.py` | Yes |
| `split_entities` / `put_split_assignment` | `split.py` | Yes (custom val splits) |
| Forbidden-import guard | `test_import_graph.py` | Enforces vendor-free core |

### Patterns from rest of ds-platform

- **No optional dependency groups** for vendors today
- **No adapter packages** in repo (`modeling/adapters/` explicitly omitted in plan)
- **External system references:** `ExternalRunRef` — correlation only, no clients
- **Store protocol:** vendor-neutral bytes in/out
- **Serialization:** caller-owned; platform hashes bytes

The established pattern is: **semantic contracts in core, vendor code in consumers.**

---

## Architectural Risks

| Risk | Description | Mitigation (when implementing) |
| --- | --- | --- |
| sklearn becomes hidden core dep | Adapters import sklearn for preprocessing | Keep adapters in consumer; optional extra only |
| NumPy becomes semantic requirement | FeatureTable → array conversion needs numpy | NumPy stays inside adapter, not in FeatureTable type |
| pandas becomes required abstraction | GBDT categorical paths prefer DataFrame | Adapter choice, not platform |
| CatBoost semantics leak into core | Adding `text_features` to FeatureTable | Keep in consumer view metadata |
| Vendor classes as public types | Exporting `XGBClassifier` from ds-platform | Forbidden by architecture |
| ModelSpec.family dispatch | `family="xgboost"` → import xgboost | Explicitly rejected in plan and run.py |
| Model registries | Central catalog of estimators | Non-goal |
| Serialization tied to vendor | Platform assumes `.cbm` format | Caller sets media_type; bytes opaque to platform |
| run_experiment as dispatcher | Runner imports libraries | Not present; must not add |
| Classifier becomes giant protocol | Adding importance, staged_predict, etc. | Keep narrow |
| FeatureView → feature store | Online serving, registry | Non-goal |
| Optional extras → plugin framework | `[xgboost]` triggers discovery | Only share adapters, not auto-dispatch |
| Generic ModelAdapter proliferation | Base adapter hierarchy in platform | YAGNI until second consumer needs it |
| Core becomes ML library | Training loops, metrics, algorithms | Architecture freeze rejects this |

---

## Open Questions

1. **Will Restaurant Intelligence and BGA both need boosting before either has sklearn adapters?** If RI starts with sklearn logistic/RF and BGA stays regression-only, boosting adapter sharing may never justify a platform extra.

2. **Should column-kind metadata become a hashable feature-schema artifact?** Useful for CatBoost and audit, but no consumer has requested it yet. Could live in consumer projects as a `document` artifact initially.

3. **Is ranking on the near-term roadmap for any consumer?** If yes, evidence for a `Ranker` protocol should come from a concrete experiment spec, not from library capability alone.

4. **Should `run_experiment` v2 support validation folds for early stopping?** Possible without vendor awareness (pass train/val/test slices to adapter), but not required for basic GBDT integration.

5. **Will MLflow record library versions?** Likely yes in consumer projects — may reduce need for platform model metadata fields.

---

## Audit Conclusion

### 1. Can all three be integrated without changing core `ds-platform`?

**Yes**, for standard supervised holdout classification and regression. Consumer adapters implementing `Classifier`/`Regressor` on `FeatureTable`, using existing split/evaluate/artifact helpers and `run_experiment`, are sufficient.

### 2. If not, what exact semantic boundary is missing?

No **core** change is **required**. Operational gaps that consumers must handle in adapters:

- Column kind (numeric / categorical / text / embedding) — not in `FeatureTable`
- Validation fold / early stopping — not wired through `run_experiment`
- Ranking query groups — no `Ranker` protocol
- Library/version metadata on model artifacts — not on `ArtifactRecord`

These are **documented gaps**, not blockers for consumer integration.

### 3. Which current abstractions are sufficient?

- `ExperimentSpec`, `ModelSpec`, `SplitSpec`, `MetricSpec`, `experiment_config_hash`
- `FeatureTable`, `FeatureView`, `align_feature_tables`, `select_columns`
- `Classifier`, `Regressor` protocols
- `run_experiment`, `ExperimentResult`
- `evaluate`, `EvaluationReport`, `extra_metrics`
- `put_model_artifact`, `put_prediction_artifact`, `put_evaluation_artifact`, `put_feature_dataset`, `put_split_assignment`
- `RunContext`, `ExternalRunRef`, `ArtifactRecord`, `ArtifactKind.MODEL|PREDICTION|EVALUATION`
- `split_entities`, `SplitAssignment`, `load_split_assignment`

### 4. Which, if any, need modification?

**None required for initial GBDT support.**

**Consider only after cross-project evidence:**

| Change | Trigger |
| --- | --- |
| Feature-schema / column-kind in spec or artifact | Two consumers need shared, hashable categorical/text metadata |
| `run_experiment` validation fold support | Repeated consumer boilerplate for early stopping |
| `Ranker` protocol + runner | Concrete ranking experiment in a consumer |
| Optional `ds-platform[xgboost]` extra | Identical adapter code duplicated in two projects |
| Model artifact `contract_ref` for schema | Need for validated model payload interchange |

### 5. Which capabilities should remain library-specific?

- Early stopping, callbacks, eval_set construction
- Categorical encoding strategy (native vs ordinal vs one-hot)
- CatBoost Pool, text_features, embedding_features
- Native serialization format choice
- GPU/CPU device selection, thread counts
- Feature importance, SHAP, object importance
- Ranking group/query tensors
- Hyperparameter tuning loops

### 6. What should remain outside the core?

- XGBoost, LightGBM, CatBoost packages and all transitive numeric stack usage
- Sklearn preprocessing pipelines (unless optional extra justified later)
- Adapter implementations
- Training orchestration beyond thin `run_experiment`
- Model loading/serving infrastructure
- Algorithm-specific metric implementations (NDCG, etc.) — via `extra_metrics` in consumers

### 7. What evidence would be needed before making a core architectural change?

1. **Two consumer projects** (e.g., RI and BGA) implementing **duplicate** boosting adapters with the same FeatureTable conversion and serialization conventions.
2. A **concrete ranking experiment** that cannot be expressed with split helpers + custom runner.
3. **Reproducibility audit failure** where feature-schema or validation-split metadata must be content-addressed platform artifacts, not consumer documents.
4. **Forbidden-import test failure** because core genuinely needs a shared non-vendor abstraction (e.g., column-kind enum) — not because a library is popular.

Until that evidence exists, **the zero-change hypothesis stands.**

---

## Appendix: Files Inspected

### Documentation
- `docs/architecture.md`
- `docs/architecture/ds-platform-adversarial-audit.md`
- `docs/architecture/ds-platform-adversarial-audit-pass2.md`
- `docs/modeling-architecture-plan-v1.md`
- `README.md`

### Source — modeling
- `src/ds_platform/modeling/__init__.py`
- `src/ds_platform/modeling/_spec_json.py`
- `src/ds_platform/modeling/spec.py`
- `src/ds_platform/modeling/capabilities.py`
- `src/ds_platform/modeling/features.py`
- `src/ds_platform/modeling/representations.py`
- `src/ds_platform/modeling/split.py`
- `src/ds_platform/modeling/evaluate.py`
- `src/ds_platform/modeling/records.py`
- `src/ds_platform/modeling/run.py`
- `src/ds_platform/modeling/cluster.py`
- `src/ds_platform/modeling/encode.py`
- `src/ds_platform/modeling/pairs.py`
- `src/ds_platform/modeling/perspectives.py`
- `src/ds_platform/modeling/sequences.py`
- `src/ds_platform/modeling/interventions.py`
- `src/ds_platform/modeling/geometry.py`

### Source — core
- `src/ds_platform/__init__.py`
- `src/ds_platform/types.py`
- `src/ds_platform/store.py`
- `src/ds_platform/hashing.py`

### Tests (all modeling tests reviewed)
- `tests/modeling/test_import_graph.py`
- `tests/modeling/test_lifecycle.py`
- `tests/modeling/test_run.py`
- `tests/modeling/test_capabilities.py`
- `tests/modeling/test_features.py`
- `tests/modeling/test_spec.py`
- `tests/modeling/test_split.py`
- `tests/modeling/test_evaluate.py`
- `tests/modeling/test_records.py`
- `tests/modeling/test_representations.py`
- `tests/modeling/test_modeling_hardening.py`
- `tests/modeling/test_geometry.py`
- `tests/modeling/test_pairs.py`
- `tests/modeling/test_perspectives.py`
- `tests/modeling/test_sequences.py`
- `tests/modeling/test_interventions.py`
- `tests/audit/test_adversarial_audit.py` (modeling-related sections)

### Configuration
- `pyproject.toml`
- `uv.lock` (presence confirmed)

## Appendix: External Documentation Consulted

- XGBoost: sklearn estimator interface, Python API reference, categorical data tutorial, model IO / serialization tutorial
- LightGBM: LGBMClassifier, LGBMRegressor API, early_stopping callback, sklearn module source
- CatBoost: CatBoostClassifier, Pool, fit, save_model/load_model, JSON export tutorial

## Appendix: Repository State at Audit Time

| Item | Status |
| --- | --- |
| Audit file created | `docs/architecture/xgboost-catboost-lightgbm-audit.md` (this file) |
| Other files modified for audit | **None** (pre-existing uncommitted modeling hardening changes from prior session remain in working tree) |
| `git status` | Modified: `__init__.py`, `evaluate.py`, `records.py`, `split.py`; untracked: `cluster.py`, `test_modeling_hardening.py` |
| Tests run | `pytest`: **235 passed** |
| ruff / pyright | Not re-run for this audit-only task |
