# Modeling architecture plan v1

**Status: v1 implemented; representation/sequence/geometry foundation
implemented; probability-aware evaluation implemented; leakage-safe
transforms and three-way splits implemented; cross-validation aggregation
and model comparison implemented; retrieval/RAG still proposed.**

This document describes the approved architecture for reusable DS/ML
experimentation contracts in `ds-platform` and records what is implemented
versus deferred.

**Implemented (v1):** `src/ds_platform/modeling/` — experiment specs,
`FeatureTable` / `FeatureView`, split and evaluate helpers, artifact record
helpers, `Classifier` / `Regressor` protocols, and optional holdout
`run_experiment`. Public exports live on `import ds_platform.modeling` only;
the top-level `ds_platform` package does not re-export modeling symbols.

**Implemented (ML foundation):** parallel representation plane —
`RepresentationTable`, `PairTable`, `PerspectiveTable`, `SequenceTable`,
conditioned / sequence / cluster protocols, grouped splits, intervention
queries, collection geometry, and `kind=index` bytes. `run_experiment` is
unchanged.

**Still proposed / not implemented:** `retrieval.py`, RAG types
(`Retriever`, `RetrievedItem`, `filter_temporal`, …), MCP/`Tool`, vendor
adapter extras, and consumer-project adoption.

Read [architecture.md](architecture.md) for the artifact freeze set. This
document extends that freeze with the modeling contract layer.

---

## 1. Purpose and scope

### What this adds (when implemented)

A small, project-agnostic layer for:

- declarative **experiment specifications** (`ExperimentSpec`)
- narrow **capability protocols** (`Classifier`, `Regressor`, …) — not a
  `BaseModel` hierarchy
- composable **feature representations** (`FeatureView`, `FeatureTable`)
- entity-level **splitting** with optional temporal cutoffs
- **evaluation** as plain functions, separate from model adapters
- **artifact helpers** that write existing kinds (`model`, `prediction`,
  `evaluation`, `dataset`) using existing provenance fields
- **retrieval contracts** for RAG-style workflows without a generic RAG
  framework

### What this does not add

- A training runtime, orchestrator, or experiment UI
- A model registry, experiment registry, or feature store
- A universal `predict()` abstraction
- Per-algorithm platform classes (XGBoost, LightGBM, …)
- New `ArtifactKind` values
- Domain types for restaurants, board games, or any consumer project
- Vendor imports on the default `import ds_platform` path

Consumers (`restaurant-intelligence`, `board-game-analysis`, future projects)
implement domain **feature views**, **targets**, and **library adapters**.
They call the platform runner and record artifacts.

---

## 2. Dependency-direction invariant (non-negotiable)

`ds-platform` is the reusable foundation. Other projects import
`ds-platform`. `ds-platform` does not import the projects that consume it.

```text
ds-platform
    ↑
    │ imported by
    │
    ├── restaurant-intelligence
    ├── board-game-analysis
    └── future projects
```

**Allowed**

- `restaurant-intelligence` MAY import `ds-platform`
- `board-game-analysis` MAY import `ds-platform`
- Future projects MAY import `ds-platform`

**Forbidden**

- `ds-platform` MUST NOT import `restaurant-intelligence`
- `ds-platform` MUST NOT import `board-game-analysis`
- `ds-platform` MUST NOT contain project-specific domain logic
- `ds-platform` MUST NOT depend on project-specific schemas, datasets,
  experiments, targets, or business concepts
- A project-specific implementation must never require adding that
  project's dependency back into `ds-platform`

When a capability appears reusable, first ask whether it can be expressed
entirely in generic DS / data / artifact terms. If yes, it may belong in
`ds-platform`. If it requires knowledge of a particular project's domain, it
belongs in that project.

This invariant applies to import graphs, `pyproject.toml` dependencies,
optional extras, tests, and schemas shipped in the wheel.

### Audit of current codebase (documentation pass)

- **No reverse import exists today.** Runtime code imports only stdlib,
  pydantic, and `ds_platform.*`. `pyproject.toml` depends only on pydantic.
- **Documentary mentions** of consumer repos in README and `architecture.md`
  are allowed (adoption targets, not imports).
- **Test gap:** forbidden-import tests cover vendor stacks (boto3, mlflow,
  …) but not consumer packages. Implementation must extend those tests so
  `import ds_platform.modeling` cannot load `restaurant_intelligence` or
  `board_game_analysis`.

---

## 3. Relationship to existing ds-platform architecture

This plan **preserves** the existing freeze:

| Existing decision | Modeling layer behavior |
| --- | --- |
| Root noun is **artifact** | Features, models, predictions, evaluations remain artifacts |
| Closed **ArtifactKind** vocabulary | Uses `dataset`, `model`, `prediction`, `evaluation`, `prompt`, `index`, `claim_set`, `review` — **no new kinds** |
| **No common predict API** | Task-specific protocols (`Classifier`, `Regressor`); no `Predictor` |
| **Quality ≠ evaluation** | `evaluate()` writes `evaluation` artifacts; quality stays separate |
| **Champion is an alias** | `logical_key` + `champion_of`; no registry in platform |
| **MLflow is external** | Cite via `RunContext.external_run_ids.mlflow` |
| **One Store protocol** | Record helpers call existing `Store` / `put_record` |
| **Citation / Evidence / Claim** | Retrieval and predictions reuse attribution types |
| **No feature store** | `FeatureTable` is an in-memory value object, not a store |

This plan **adds** (when implemented):

- `src/ds_platform/modeling/` — specs, protocols, functions, thin orchestration
- `RunContext.config_hash` populated from `experiment_config_hash(spec)`
- Optional `RetrievalSpec` hashed alongside the experiment

This plan **does not amend** artifact kinds or envelope shape.

---

## 4. Package tree

**v1 + ML foundation implemented** under `src/ds_platform/`
(except `retrieval.py`):

```text
src/ds_platform/
├── __init__.py                 # existing; modeling is a subpackage import
├── hashing.py                  # existing
├── store.py                    # existing
├── types.py                    # existing
├── schemas/                    # existing; optional experiment-spec schema later
└── modeling/                   # v1 + representation foundation
    ├── __init__.py             # public exports (not top-level re-export)
    ├── _spec_json.py           # private spec canonicalization
    ├── spec.py
    ├── capabilities.py
    ├── features.py
    ├── representations.py
    ├── pairs.py
    ├── perspectives.py
    ├── sequences.py
    ├── interventions.py
    ├── geometry.py
    ├── split.py
    ├── evaluate.py
    ├── records.py
    ├── run.py
    ├── encode.py
    ├── transforms.py
    └── retrieval.py            # NOT IMPLEMENTED — still proposed
```

**Intentionally omitted**

- Top-level repo folders (`modeling/`, `features/`, `evaluation/`) outside
  the Python package
- `src/ds_platform/artifacts/`, `lineage/`, `quality/` — already covered by
  `types.py` and `store.py`
- `modeling/adapters/` — until a second project needs a shared library
  wrapper (optional extra, not core)
- `modeling/nlp/`, `vision/`, `rag/`, `mcp/` — concepts, not boundaries

`import ds_platform` need not re-export all modeling symbols on day one.
Consumers may use `import ds_platform.modeling`.

---

## 5. API surface by module

Conventions for the proposed public API:

- Frozen Pydantic models, `extra="forbid"`, for specs and value objects
- Hash and logical-key fields are `str` (validated internally)
- No sklearn, numpy, or consumer domain types in signatures
- Plain functions preferred over classes where a class adds no boundary

### 5.1 `spec.py` — experiment configuration

**Purpose:** Hashable experiment document. Not an artifact kind; hash stored
on `RunContext.config_hash`.

```python
class DatasetRef:
    payload_id: str | None = None  # sha256 hex of dataset bytes
    logical_key: str | None = None  # at least one field required


class FeatureSpec:
    views: tuple[str, ...]  # FeatureView.name values
    columns: tuple[str, ...] | None  # optional allowlist after align
    as_of: date | None  # passed through to views


class TargetSpec:
    column: str
    task: Literal["classification", "regression"]


class ModelSpec:
    family: str  # opaque label for the hash, not a lookup key
    params: dict[str, JsonValue]  # JSON-serializable only


class SplitSpec:
    method: Literal["holdout", "kfold", "temporal"]
    seed: int
    test_size: float | None  # holdout / temporal
    validation_size: float | None  # holdout / temporal three-way
    n_splits: int | None  # kfold
    stratify: bool = False
    as_of: date | None


class MetricSpec:
    name: str  # e.g. "accuracy", "rmse"
    params: dict[str, JsonValue]


class RetrievalSpec:
    corpus_payload_ids: tuple[str, ...]
    k: int
    embedder_payload_id: str | None
    prompt_payload_id: str | None
    hybrid: bool = False
    as_of: date | None


class ExperimentSpec:
    experiment_id: str
    dataset: DatasetRef
    features: FeatureSpec
    target: TargetSpec
    model: ModelSpec
    split: SplitSpec
    metrics: tuple[MetricSpec, ...]
    seed: int
    retrieval: RetrievalSpec | None = None  # hashed when present


def experiment_config_hash(spec: ExperimentSpec) -> str: ...
```

**Uses existing types:** `canonical_json_bytes`, `sha256_hex` →
`RunContext.config_hash`.

**Private:** field validators, JSON dump helpers.

**Not public:** model-family dispatch, experiment registry.

---

### 5.2 `capabilities.py` — task protocols

**Purpose:** Structural typing for adapters. Not an ABC hierarchy.

```python
class Classifier(Protocol):
    def fit(self, features: FeatureTable, y: Sequence[str | int]) -> None: ...
    def predict(self, features: FeatureTable) -> Sequence[str | int]: ...


class Regressor(Protocol):
    def fit(self, features: FeatureTable, y: Sequence[float]) -> None: ...
    def predict(self, features: FeatureTable) -> Sequence[float]: ...
```

**Reserved for future phases (documented only; do not stub until needed):**
`Ranker`, `SurvivalModel`, `Clusterer`, `Embedder`, `ProbabilisticModel`,
`CausalEstimator`.

**Private:** optional `HasPredictProba` protocol; `run_experiment` may
duck-type `predict_proba` if present but must not require it.

**Intentionally not public:** `Predictor`, `task_capability()`.

Two protocols that both define `predict` is **not** a universal predict API.
The runner branches on `TargetSpec.task`.

---

### 5.3 `features.py` — representations

**Purpose:** Compose named views into an id-aligned table. Not a feature
store.

```python
type Scalar = str | int | float | bool | None


class FeatureTable:
    entity_ids: tuple[str, ...]
    columns: tuple[str, ...]
    values: tuple[tuple[Scalar, ...], ...]  # rows aligned to entity_ids
    source_payload_ids: tuple[str, ...]


class FeatureView(Protocol):
    @property
    def name(self) -> str: ...
    def transform(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        as_of: date | None = None,
    ) -> FeatureTable: ...


def align_feature_tables(
    tables: Sequence[FeatureTable],
    *,
    how: Literal["inner"] = "inner",
) -> FeatureTable: ...


def select_columns(table: FeatureTable, columns: Sequence[str]) -> FeatureTable: ...
def extract_column(table: FeatureTable, column: str) -> tuple[Scalar, ...]: ...
```

**Uses existing types:** `source_payload_ids` are artifact `payload_id`s.

**Private:** join and column-prefix helpers.

**Intentionally not public:** online serving, feature registration, point-in-
time store APIs.

---

### 5.4 `split.py` — entity splits

**Purpose:** Split on **entity ids**, not raw rows, to reduce leakage when
one entity has many observations.

```python
class SplitAssignment:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]   # empty unless validation_size is set
    test_ids: tuple[str, ...]

def split_entities(
    entity_ids: Sequence[str],
    spec: SplitSpec,
    *,
    labels: Sequence[object] | None = None,
    timestamps: Sequence[date] | None = None,
) -> tuple[SplitAssignment, ...]:
    # holdout → length 1; kfold → length n_splits

def apply_split(
    table: FeatureTable,
    assignment: SplitAssignment,
) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
    # train, validation, test (validation may be empty)
```

If `timestamps` and `spec.as_of` are set, entities with
`timestamp > as_of` are dropped before shuffling.

**v1 runner:** `run_experiment` accepts **holdout only** and errors on
`kfold`. Use `run_kfold` for cross-validation orchestration. Optional
`fitted_transform` is fit on the train slice only, then applied to
train/validation/test before `adapter.fit`.

`FeatureSchema` is a hashable column-kind document (`numeric`,
`categorical`, `text`, `passthrough`). It is not stored on `FeatureTable`
cells. Persist it with `put_feature_schema` as `kind=document`.

**Private:** RNG and stratify implementation.

---

### 5.5 `evaluate.py` — metrics

**Purpose:** Evaluation is separate from model adapters.

```python
class EvaluationReport:
    metrics: dict[str, float]
    n: int
    n_missing: int
    notes: tuple[str, ...]


def evaluate(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    metrics: Sequence[MetricSpec],
    *,
    y_proba: Sequence[Sequence[float]] | None = None,
) -> EvaluationReport: ...
```

**Implemented generic metric functions (plain functions, not a plugin system):**
`accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, `rmse`, `mae`,
`r2`, `log_loss`, `brier_score`, `roc_auc_binary`. Probability metrics use
`y_proba`; `run_experiment` forwards adapter probabilities and duck-typed
`classes`. `roc_auc_binary` accepts `MetricSpec.params["pos_label"]`.
`EvaluationReport.notes` records which metrics used probabilities.

**Private:** name → function dispatch dict. **Do not export a metric registry.**

Domain-specific slices (by borough, by mechanic, etc.) stay in consumer
projects.

---

### 5.6 `records.py` — artifact persistence

**Purpose:** Thin helpers over existing `Store`, `ArtifactRecord`, and
`put_record`. Not a registry.

```python
def put_model_artifact(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:          # payload_id, record_id

def put_feature_dataset(...) -> tuple[str, str]:   # kind=dataset

def put_prediction_artifact(
    store: Store,
    data: bytes,
    *,
    run: RunContext,
    inputs: Sequence[str],
    media_type: str,
    logical_key: str | None = None,
    contract_ref: ContractRef | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:

class PredictionRow:
    entity_id: str
    y_pred: Scalar
    y_true: Scalar | None = None
    y_proba: tuple[float, ...] | None = None
    evidence: tuple[Evidence, ...] = ()

def prediction_payload_bytes(rows: Sequence[PredictionRow]) -> bytes: ...

def put_evaluation_artifact(
    store: Store,
    report: EvaluationReport,
    *,
    run: RunContext,
    subject_payload_id: str,
    inputs: Sequence[str],
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    # kind=evaluation, related=[evaluation_of → subject]

def put_comparison_artifact(
    store: Store,
    report: ComparisonReport,
    *,
    run: RunContext,
    inputs: Sequence[str],
    logical_key: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, str]:
    # kind=evaluation, related=[evaluation_of → left, evaluation_of → right]
```

**Uses existing types:** `Store`, `RunContext`, `ContractRef`, `ArtifactKind`,
`ArtifactRecord`, `RelatedRef`, `RelationType.EVALUATION_OF`, `Evidence`,
`put_record`, `payload_id`.

**Intentionally not public:** `list_models`, `get_champion`, `register_experiment`.

---

### 5.7 `compare.py` — model comparison and fold aggregation

**Purpose:** Compare two evaluations and aggregate k-fold scores without a
model registry or champion selector.

```python
class ComparisonReport:
    method: Literal["holdout_delta", "paired_bootstrap"]
    left_payload_id: str
    right_payload_id: str
    deltas: Mapping[str, float]   # left - right
    interval_low: Mapping[str, float] | None = None
    interval_high: Mapping[str, float] | None = None
    n: int
    seed: int | None = None
    notes: tuple[str, ...] = ()


def compare_evaluations(
    left: EvaluationReport,
    right: EvaluationReport,
    *,
    left_payload_id: str,
    right_payload_id: str,
) -> ComparisonReport: ...

def paired_bootstrap(
    y_true: Sequence[object],
    y_pred_left: Sequence[object],
    y_pred_right: Sequence[object],
    metrics: Sequence[MetricSpec],
    *,
    left_payload_id: str,
    right_payload_id: str,
    seed: int,
    n_resamples: int = 1000,
    y_proba_left: Sequence[Sequence[float] | None] | None = None,
    y_proba_right: Sequence[Sequence[float] | None] | None = None,
    classes: Sequence[object] | None = None,
) -> ComparisonReport: ...

def aggregate_fold_reports(
    reports: Sequence[EvaluationReport],
) -> EvaluationReport: ...
```

`compare_evaluations` requires matching metric names and the same `n`.
`paired_bootstrap` resamples aligned prediction tuples with stdlib `random`
only. `aggregate_fold_reports` returns the unweighted mean of fold metrics.

Persist comparisons with `put_comparison_artifact` as `kind=evaluation`,
citing both compared evaluation payloads via dual `evaluation_of` relations.

---

### 5.8 `run.py` — thin orchestration

**Purpose:** Holdout and k-fold runners so lineage edges stay consistent. Not
a training manager.

```python
class ExperimentResult:
    run_id: str
    config_hash: str
    feature_payload_id: str
    model_payload_id: str
    prediction_payload_id: str
    evaluation_payload_id: str
    report: EvaluationReport


class KFoldResult:
    run_id: str
    config_hash: str
    feature_payload_id: str
    aggregated_evaluation_payload_id: str
    aggregated_report: EvaluationReport
    folds: tuple[KFoldFoldResult, ...]


def run_experiment(
    spec: ExperimentSpec,
    *,
    rows: Sequence[Mapping[str, object]],
    feature_views: Mapping[str, FeatureView],
    adapter: Classifier | Regressor,
    store: Store,
    run: RunContext,
    serialize_model: Callable[[object], bytes],
    model_media_type: str = "application/octet-stream",
    fitted_transform: FittedTransform | None = None,
) -> ExperimentResult: ...


def run_kfold(
    spec: ExperimentSpec,
    *,
    rows: Sequence[Mapping[str, object]],
    feature_views: Mapping[str, FeatureView],
    adapter_factory: Callable[[], Classifier | Regressor],
    store: Store,
    run: RunContext,
    fitted_transform_factory: Callable[[], FittedTransform] | None = None,
) -> KFoldResult: ...
```

**v1 behavior:**

- `run_experiment` requires `spec.split.method == "holdout"`
- `run_kfold` requires `spec.split.method == "kfold"` and creates a fresh
  adapter per fold via `adapter_factory`; model artifacts are not persisted
- Require `spec.retrieval is None` (retrieval composed in project
  `FeatureView`, not inside runner)
- Do **not** construct an adapter from `ModelSpec.family` — family is for
  hashing only
- Extract labels via `extract_column` / `TargetSpec.column`
- Duck-type `predict_proba` if adapter provides it

**Uses existing types:** `Store`, `RunContext`.

---

### 5.9 `retrieval.py` — retrieval contracts

**Purpose:** RAG as composable data/ML nodes, not a framework.
`RetrievalSpec` lives in `spec.py` (declarative, hashed).

```python
class RetrievedItem:
    evidence: Evidence  # citation + excerpt + method
    score: float | None = None
    source_timestamp: datetime | None = None


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        as_of: date | None,
        k: int,
    ) -> Sequence[RetrievedItem]: ...


def filter_temporal(
    items: Sequence[RetrievedItem],
    as_of: date,
    *,
    require_timestamp: bool = True,
) -> tuple[RetrievedItem, ...]: ...


def evidence_payload_ids(items: Sequence[RetrievedItem]) -> tuple[str, ...]: ...
```

**Uses existing types:** `Evidence`, `Citation`.

**Intentionally not public:** `retrieve_to_feature_table` (would encode a
domain schema), generic RAG orchestration, MCP server types.

---

### 5.10 `modeling/__init__.py` — v1 public exports (implemented)

Export:

- All spec types and `experiment_config_hash`
- `Classifier`, `Regressor`
- `FeatureTable`, `FeatureView`, `align_feature_tables`, `select_columns`,
  `extract_column`
- `SplitAssignment`, `split_entities`, `apply_split`
- `EvaluationReport`, `evaluate`, named generic metrics
- `ComparisonReport`, `compare_evaluations`, `paired_bootstrap`,
  `aggregate_fold_reports`
- `put_model_artifact`, `put_feature_dataset`, `put_prediction_artifact`,
  `PredictionRow`, `prediction_payload_bytes`, `put_evaluation_artifact`,
  `comparison_report_bytes`, `put_comparison_artifact`
- `ExperimentResult`, `run_experiment`, `KFoldFoldResult`, `KFoldResult`,
  `run_kfold`
**Intentionally not public (including deferred retrieval/RAG)**

| Name | Reason |
| --- | --- |
| `Predictor` | Universal predict API rejected by architecture freeze |
| `task_capability()` | Registry-like dispatch |
| Public metric registry | Plugin system not warranted |
| Model-family dispatch | `ModelSpec.family` is opaque for hashing only |
| `HasPredictProba` | Duck-type internally |
| Feature-store abstractions | No serve/register/get |
| Model / experiment registries | MLflow + aliases handle this externally |
| Generic RAG framework | Composition only |
| MCP framework | Tool boundary deferred to consumer |
| Algorithm-specific platform classes | Thin adapters in consumer or optional extra |

---

## 6. Experiment lifecycle

### v1 flow (implemented)

```text
dataset (existing artifact, consumer-owned logical_key)
  ↓
FeatureView.transform(rows, as_of) → FeatureTable
  ↓
align_feature_tables / select_columns
  ↓
split_entities / apply_split
  ↓
Classifier | Regressor.fit / predict
  ↓
PredictionRow[] → prediction artifact
  ↓
evaluate → evaluation artifact
  ↓
review (optional, existing kind=review)
```

### Platform vs consumer

| Step | ds-platform (v1) | Consumer project |
| --- | --- | --- |
| Dataset ingest | — | ingestion pipeline, `kind=dataset` |
| Row loading | — | JSONL/Parquet → `Mapping[str, object]` |
| Feature views | `FeatureView` protocol | domain transforms |
| Align / select | `align_feature_tables`, … | column allowlists in `FeatureSpec` |
| Split | `split_entities`, `apply_split` | timestamps for temporal tasks |
| Model | protocols only | library wrapper + serialization |
| Run | `run_experiment` | wires views, adapter, store |
| Evaluation | `evaluate`, `put_evaluation_artifact` | optional slice metrics |
| Champion | — | alias record citing evaluation |

Runtime call flow is **not** import direction. Consumers import
`ds_platform`; the runner receives views and adapters as **arguments**.

---

## 7. Feature representation

A **feature view** is a named, deterministic transform:

```text
structured rows (generic mappings)
        ↓
FeatureView.transform(..., as_of=...)
        ↓
FeatureTable (entity_ids + columns + values + source_payload_ids)
        ↓
optional materialization as kind=dataset artifact
```

Multimodal ablations are different **view lists** in `FeatureSpec`, not
different platform subsystems:

- structured only
- structured + text embeddings
- structured + RAG-derived columns
- combinations joined by `entity_ids`

**Not a feature store:** no online serving, no registration API, no
historical point-in-time retrieval beyond explicit `as_of` passed into views.

---

## 8. Splitting and leakage control

- Splits operate on **entity ids**, not duplicate rows per entity.
- `SplitSpec.as_of` plus optional `timestamps` on `split_entities` drop
  entities observed after the cutoff.
- `FeatureSpec.as_of` is passed into each `FeatureView.transform` so views
  can exclude future inspections, reviews, or documents.
- Retrieval uses `filter_temporal` with the same cutoff semantics.

Consumers define what a timestamp means (inspection date, review date, etc.).
The platform only applies the comparison.

---

## 9. Evaluation

Evaluation is **not** a method on model adapters.

```text
predictions + labels + MetricSpec[]
        ↓
evaluate(...)
        ↓
EvaluationReport
        ↓
put_evaluation_artifact(..., related=evaluation_of)
```

For RAG workflows (future), evaluate separately when workloads exist:

```text
retrieval        → retrieval metrics
generation/extraction → extraction metrics
downstream prediction → prediction metrics
```

Do not collapse these into one score.

Quality artifacts (`kind=quality`) remain separate from evaluation.

---

## 10. Artifact and provenance integration

No parallel ML metadata system. Use existing fields:

| Artifact | Kind | Provenance |
| --- | --- | --- |
| Feature table | `dataset` | `inputs` = parent dataset(s); `logical_key` e.g. `{project}:features/...` |
| Experiment run | embedded in each record | `RunContext.config_hash` = `experiment_config_hash(spec)` |
| Model | `model` | `inputs` = feature (+ optional split assignment) payload ids |
| Prediction | `prediction` | `inputs` = model, features, split |
| Evaluation | `evaluation` | `related` = `evaluation_of` → prediction or model; `inputs` = label suite |
| Champion (later) | alias on `model` | `logical_key` + `champion_of`; cite evaluation |
| Retrieved evidence | `document` or `dataset` | `inputs` = source docs; citations on rows |
| LLM step | `prompt` + `InferenceRecord` in trace `document` | existing attribution types |
| Retrieval index | `index` | derived; not source of truth |
| MLflow | — | `RunContext.external_run_ids.mlflow` only |

---

## 11. Retrieval / RAG

RAG is **composition of existing nodes**, not a `rag/` package.

```text
RetrievalSpec (hashed in ExperimentSpec when used)
  ↓
Retriever.retrieve(query, as_of, k) → RetrievedItem
  ↓
filter_temporal
  ↓
Evidence / Citation (existing types)
  ↓
PROJECT FeatureView maps passages → FeatureTable
  ↓
same split / model / evaluate path
```

Use cases (consumer-implemented):

- retrieval-augmented feature generation
- RAG-derived structured features for downstream sklearn/boosting
- evidence-backed weak supervision (`Claim.layer=inferred`, not ground truth)
- retrieval vs embedding vs no-retrieval ablations
- error analysis using cited excerpts

The platform does **not** ship a class that connects retrieval to features.
The join is a project `FeatureView` that calls a `Retriever`.

---

## 12. MCP boundary

MCP is a **tool-interface adapter**, not a model family.

- **v1:** ordinary Python functions in the consumer (`fetch_inspections`, …)
- **Future (only if justified):** optional `Tool` protocol in platform
  (`name`, `parameters_schema`, `call`) when an external agent must invoke
  lookup without importing the consumer package
- **MCP SDK** implements `Tool` in the **consumer** (or a later optional extra)

**Trigger for MCP:** an external process must call retrieval/lookup without
importing the consumer. Until then, MCP is out of scope.

Do not build an agent loop, planner, or tool-registry service in
`ds-platform`.

---

## 13. External ML library boundary

| Library | Role | Where |
| --- | --- | --- |
| pydantic | specs | core (existing) |
| scikit-learn | adapter | consumer first; optional `ds-platform[sklearn]` only after cross-project pressure |
| XGBoost / LightGBM / CatBoost | adapters | consumer or optional extra when a workload needs them |
| MLflow | experiment UI / registry | consumer; cite run id on `RunContext` |
| Airflow / Dagster | orchestration | consumer |
| dbt / SQLMesh | SQL transforms | consumer |
| PyTorch / Hugging Face / sentence-transformers | representation | consumer when text/image exists |
| lifelines / scikit-survival | survival | consumer or extra when labels exist |
| PyMC / Stan | Bayesian | consumer |
| faiss / sqlite-vec | index payload | consumer; `kind=index` artifact |
| MCP SDK | tool transport | consumer when boundary trigger met |

**Rules**

- Default `import ds_platform` and `import ds_platform.modeling` stay
  vendor-free (same invariant as existing forbidden-import tests).
- Adapters convert `FeatureTable` ↔ library arrays inside the consumer or
  optional extra — never in core signatures.
- `ModelSpec.params` is JSON-serializable only; no vendor client objects.

---

## 14. Restaurant-intelligence consumer example

**Today:** ingestion-only (NYC DOHMH → `Restaurant` documents → dataset →
quality). No modeling code.

**Proposed usage (when both platform modeling and RI consumer code exist):**

```python
# restaurant_intelligence/modeling/views.py (consumer — not ds-platform)
class StructuredRestaurantView:
    name = "structured"

    def transform(self, rows, *, as_of=None) -> FeatureTable: ...


# restaurant_intelligence/modeling/run_cuisine.py (consumer)
spec = ExperimentSpec(
    experiment_id="cuisine-v0",
    dataset=DatasetRef(logical_key="ri:dataset:v0"),
    features=FeatureSpec(views=("structured",), as_of=cutoff_date),
    target=TargetSpec(column="cuisine", task="classification"),
    model=ModelSpec(family="sklearn_logistic", params={"C": 1.0}),
    split=SplitSpec(method="holdout", seed=42, test_size=0.2, stratify=True),
    metrics=(MetricSpec(name="accuracy"), MetricSpec(name="macro_f1")),
    seed=42,
)

result = run_experiment(
    spec,
    rows=load_restaurant_mappings(jsonl_path),
    feature_views={"structured": StructuredRestaurantView()},
    adapter=SklearnClassifierWrapper(LogisticRegression()),  # RI-owned
    store=store,
    run=run_context,
    serialize_model=pickle.dumps,
)
```

RI owns: `Restaurant`, NYC client, column semantics, sklearn wrapper, logical
keys (`ri:features/...`), prediction JSON schema, MLflow client when used.

---

## 15. Board-game-analysis consumer example

BGA is **not required** to adopt modeling. If it later trains on corpus
metadata:

```python
# board_game_analysis/modeling/views.py (consumer — hypothetical)
class StructuredGameView:
    name = "structured"

    def transform(self, rows, *, as_of=None) -> FeatureTable: ...


spec = ExperimentSpec(
    experiment_id="complexity-v0",
    dataset=DatasetRef(logical_key="bga:corpus:v1"),
    features=FeatureSpec(views=("structured",)),
    target=TargetSpec(column="complexity", task="regression"),
    model=ModelSpec(family="sklearn_linear", params={}),
    split=SplitSpec(method="holdout", seed=42, test_size=0.2),
    metrics=(MetricSpec(name="rmse"), MetricSpec(name="mae")),
    seed=42,
)

result = run_experiment(
    spec,
    rows=load_game_mappings(jsonl_path),
    feature_views={"structured": StructuredGameView()},
    adapter=SklearnRegressorWrapper(LinearRegression()),  # BGA-owned
    store=store,
    run=run_context,
    serialize_model=pickle.dumps,
)
```

BGA never imports restaurant-intelligence. Both import only `ds_platform`.

---

## 16. Testing and reverse-import enforcement

### ds-platform tests (v1 implemented)

**Unit (fixtures only, no consumer packages, no HTTP):**

- spec hash stability and `extra=forbid`
- capability checks with dummy protocol implementations
- `align_feature_tables`, `select_columns`, `extract_column`
- split seed, disjoint ids, temporal cutoff
- `evaluate` on toy vectors; unknown metric errors
- `filter_temporal` drops post-cutoff; missing timestamp + `as_of` → error
- record helpers set correct kind, `inputs`, `related`

**Integration:**

- `run_experiment` with dummy view + dummy Classifier + `LocalStore`
- identical inputs → identical `payload_id`s
- `evaluation_of` link present

**Import-graph invariant (extend existing tests):**

After `import ds_platform` and `import ds_platform.modeling`,
`sys.modules` must not contain `restaurant_intelligence` or
`board_game_analysis`. Existing tests forbid vendor stacks but not consumers
today — implementation must close this gap.

Platform tests use generic rows (`{"id": "e1", "x": 1.0}`), never
`Restaurant` or `Game`.

### Consumer tests (later, in each project)

- Domain views and targets
- End-to-end experiment on fixtures
- No network in CI

---

## 17. Implementation phases

| Phase | Status | Work |
| --- | --- | --- |
| 0 | **Done** | Approved architecture (this document) |
| 1 | **Done** | `src/ds_platform/modeling/` v1: spec, capabilities, features, split, evaluate, records, run, public exports, lifecycle tests, forbidden-import tests |
| 2 | **Done** | Pointer in `architecture.md`; this document updated to implemented vs deferred |
| 2b | **Done** | Representation foundation: `RepresentationTable`, pairs, perspectives, sequences, interventions, geometry; no consumer adoption |
| 3 | Not started | `restaurant-intelligence` consumer v0: one structured view, fixture experiment |
| 4 | Not started | RI data prerequisite: larger NYC corpus before claiming real results |
| 5 | Not started | First real RI experiment |
| 6 | Not started | Optional `ds-platform[sklearn]` if BGA also needs shared adapter |
| 7 | Not started | MLflow in consumer when comparison UI needed |
| 8 | Not started | `retrieval.py` when permitted document source exists |
| 9 | Not started | `Tool` / MCP only on process-boundary trigger |

---

## 18. Explicit non-goals

- `BaseModel` / per-algorithm class hierarchies (`XGBoostModel`, …)
- `Predictor` or any universal predict API
- Feature store, vector database, embedding store
- ExperimentRegistry, ModelRegistry, PredictionManager, EvaluationManager,
  TrainingManager
- Catalog APIs / `Store.list`
- New `ArtifactKind` values for experiments or features
- Generic RAG or agent framework
- MCP server without a concrete workload trigger
- Airflow / dbt / MLflow reimplementations
- sklearn / torch / mlflow imports on the core import path
- Domain types for restaurants, board games, or any consumer
- Reverse dependencies on consumer projects
- Claiming retrieval/RAG or MCP contracts exist before their phase lands

---

## Document history

- **v1 (proposed):** API-surface review approved. Documentation-only pass.
- **v1 (implemented):** Modeling subpackage shipped with holdout runner,
  public exports on `ds_platform.modeling`, lifecycle integration test.
  Retrieval/RAG, MCP, vendor extras, and consumer adoption remain deferred.
- **ML foundation:** Domain-agnostic representation, pair, perspective,
  sequence, intervention, and geometry primitives. No BGA types, no
  `Predictor`, no new artifact kinds.
