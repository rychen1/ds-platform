# Architecture

This document is the architecture for `ds-platform`: an independent design,
then a critical attack on that design.

It is not a product spec for Dagster, SQLMesh, MLflow, AWS, or Streamlit.
Those systems are treated as peers. It is also not a restatement of an
`ArtifactEnvelope`-centered sketch. That idea is evaluated and only kept
where it survives the evaluation.

Implementation has not started. This document decides what may be frozen
before code exists.

---

## Contents

1. [Purpose](#1-purpose)
2. [What the platform actually is](#2-what-the-platform-actually-is)
3. [Independent design](#3-independent-design)
4. [Standards](#4-standards)
5. [Technology roles](#5-technology-roles)
6. [Critical review](#6-critical-review)
7. [Recommendations](#7-recommendations)

---

## 1. Purpose

`ds-platform` exists because two (and later more) real projects will
otherwise invent incompatible answers to the same questions:

- What is a durable unit of work product?
- How do you name it so a later run can cite it?
- How do you attach provenance, quality, and evaluation without mixing
  those layers into the payload?
- How do you move from a laptop to S3 / cloud compute without rewriting
  domain objects?
- How do you record models, predictions, LLM calls, and human judgments
  without pretending they are the same kind of object?

The expensive failures in systems like this are not "we picked Polars
instead of Pandas." They are semantic: identity, attribution, mutation,
and hidden coupling to an orchestrator or a cloud vendor.

### 1.1 Design constraints

The platform must be:

- modular and reusable across projects
- locally usable and cloud-compatible
- scalable in *data volume* without forcing a distributed runtime on
  small work
- reproducible and provenance-aware
- lineage-aware without owning a lineage product
- evaluation-oriented
- governance-aware without becoming a policy engine
- compatible with conventional ML, deep learning, LLMs, and agents
- unwilling to recreate mature systems
- unwilling to lock core types to a vendor or framework

### 1.2 Explicit non-goals

- A general workflow engine
- A warehouse, lakehouse, or semantic layer product
- A model registry
- A feature store
- An LLM gateway
- An agent runtime
- A Streamlit control plane
- Domain types for board games, restaurants, or any other project
- Placeholder packages for Ray, Spark, Athena, Redshift, or AWS

---

## 2. What the platform actually is

The useful product is a **small, versioned contract library**, not a
platform in the vendor sense.

Three layers, only the first of which is this repository:

| Layer | Lives in | Examples |
| --- | --- | --- |
| Semantics | `ds-platform` | artifact identity, kinds, sidecar records, citation / evidence / claim types, run context, store protocol |
| Execution | each project + external systems | Dagster jobs, SQLMesh models, local scripts, later Ray/Spark jobs |
| Infrastructure | each project's deploy | local filesystem, later S3, PostgreSQL, ECR, EC2/ECS, Athena |

`ds-platform` ships types, validation, hashing, and a local store.
Projects compose those types. Orchestrators and clouds are not imported
by the core package.

If a proposed module cannot be explained as "a semantic boundary that is
expensive to change," it does not belong here.

---

## 3. Independent design

### 3.1 The unit of durability is an artifact, not a dataframe

A project produces many kinds of durable objects: raw API responses,
normalized tables, JSON documents, plots, trained models, prompt files,
prediction files, quality reports, evaluation reports, human reviews,
and later agent traces.

Trying to make "dataset" the root type fails as soon as a model binary
or a review note appears. Trying to make "MLflow run" the root type
fails as soon as a raw XML dump needs provenance and no model exists
yet.

**Decision:** the root noun is **artifact**.

An artifact is:

1. a **payload** — bytes with a media type
2. an **identity** — derived from the payload, not from where it sits
3. a **kind** — a closed vocabulary of what the payload *is for*
4. a **record** — structured metadata that cites the payload and the
   run that produced it

The payload is never required to be a table. Tables are a common
payload, not the abstraction.

### 3.2 `ArtifactEnvelope` is a serialization, not the architecture

A single struct that "holds everything" is a tempting center. It is also
how these systems rot.

If quality reports, claim lists, full lineage graphs, evaluation scores,
governance attestations, and storage URIs all live *inside* one
envelope, three things happen:

- the sidecar becomes large and rewrite-heavy
- independent objects cannot be cited without loading the parent
- every new concern becomes "add a field to the envelope"

That is a god object with a respectable name.

The valuable idea hiding under `ArtifactEnvelope` is narrower:

> Metadata that must survive a copy of the payload should travel as a
> sidecar document, content-addressed separately from the payload.

**Decision:** keep an envelope as the portable **record document**.
Reject the envelope as a container for every related concern.

The envelope binds:

- payload identity and media type
- artifact kind
- producing run
- citations to *other* artifacts (inputs, producer, related reports)
- optional inherited policy references
- a small, stable set of descriptive fields (name, logical key, created
  at, schema ref)

The envelope does **not** inline:

- quality results
- evaluation results
- claim sets
- full lineage graphs
- model metrics
- human reviews

Those are artifacts of their own kinds. The envelope *points at* them.

This is the main disagreement with an envelope-centric sketch. The
envelope is a passport, not a filing cabinet.

### 3.3 Identity: three names, not one

Most identity bugs come from collapsing three different names.

| Name | Meaning | Changes when |
| --- | --- | --- |
| `payload_id` | `sha256` of payload bytes | bytes change |
| `record_id` | `sha256` of the canonical envelope | metadata is revised |
| `logical_key` | human/project name, e.g. `bga:corpus:v0` | almost never; it is a pointer |

**`payload_id` is the artifact identity.** Two files with the same bytes
are the same artifact, wherever they are stored.

**`record_id` is the identity of a particular annotation** of that
artifact. The same payload may acquire a later envelope (better
citations, a linked quality report). Lineage cites `payload_id`.
Audit of "what we believed when" cites `record_id`.

**`logical_key` is an alias, not an identity.** "Current champion" and
"latest corpus" are aliases. Promoting a champion writes a new alias
record. It does not mutate the model artifact.

A URI, path, or S3 key is a **location**, never an identity.

This split is irreversible in practice. If identity is a path, cloud
migration rewrites history. If identity is the envelope hash, adding a
citation creates a "new dataset." If there is no logical key, humans
invent mutable filenames and the catalog becomes folklore.

### 3.4 Artifact kinds are a closed vocabulary

Open-ended `type: string` looks flexible and produces unqueryable junk.

v0 kinds (design, not code):

| Kind | Payload |
| --- | --- |
| `raw` | source bytes (XML, HTML, PDF, API JSON) |
| `dataset` | tabular or row-oriented data (Parquet preferred) |
| `document` | structured JSON / JSONL domain objects |
| `model` | serialized model, adapter, or tokenizer bundle |
| `prompt` | prompt template + tool/schema bindings |
| `prediction` | model or LLM outputs joined to inputs |
| `evaluation` | scored comparison against a suite or labels |
| `quality` | data-quality report |
| `claim_set` | cited claims about other artifacts |
| `review` | human judgment / adjudication |
| `run` | execution record (optional materialization of `RunContext`) |
| `index` | derived catalog/manifest, not source of truth |

Kinds can be added. Kinds should not be subclassed into
`dataset.training` vs `dataset.holdout` via the type system. Those are
logical keys or tags.

A model is an artifact. So is a prompt. Treating "the model" as only
weights is already wrong for LLM systems.

### 3.5 Contracts live at ports, not inside payloads

A contract answers: what must be true for a consumer to use this
artifact?

Contracts are not the data. They are not quality reports. A dataset can
fail a quality check and still *be* the dataset. A port can reject it.

**Decision:**

- A **contract** is a versioned JSON Schema (and, for tables, column
  constraints) identified by URI + hash.
- Artifacts *refer* to the contract they claim to satisfy.
- Projects own domain schemas (`Game`, restaurant entities). The
  platform does not.
- Alignment with ODCS is at the *field level* (schema, quality
  expectations, owner, version), not by storing native ODCS documents.

Inbound ports (what ingestion must accept) and outbound ports (what a
pipeline promises) are different contracts. One artifact may satisfy
several ports. That is why the contract is a reference, not a type
hierarchy on the payload.

### 3.6 Attribution: citation, evidence, claim

These three are routinely collapsed. They are not the same.

**Citation** — a pointer. "This value came from artifact `P`, record
`R`, optional JSON Pointer / row id / byte range." Citations are cheap
and should be common.

**Evidence** — a citation plus a quote or excerpt and, optionally, a
method. "The rulebook sentence that was used." Evidence is for
extraction and audit. Most tabular pipelines have none. That is fine.

**Claim** — a statement with a layer and a confidence policy.

```
Claim.layer ∈ { observed, derived, inferred, asserted }
```

- `observed` — copied from a source
- `derived` — computed by a published procedure
- `inferred` — model, LLM, or heuristic
- `asserted` — human or policy statement

Claims are first-class *types* in the library because mixing these
layers is the actual scientific bug in the intended consumer projects.
They are not first-class *fields* on every envelope.

**Decision:** ship `Citation`, `Evidence`, and `Claim` as types. Store
collections of claims as `claim_set` artifacts. A dataset envelope may
cite a claim set. It does not carry 10⁵ claims inline.

Conflicting claims are allowed. The platform does not resolve them. A
later review artifact may adjudicate.

### 3.7 Provenance vs lineage

**Provenance** answers: what went into this, under what code, policy,
and inputs?

**Lineage** answers: how do artifacts connect, for graph query and
impact analysis?

They share edges. They are not the same product.

`RunContext` (native, small):

- `run_id` (platform-issued)
- `project`
- `code_ref` (git sha + dirty flag)
- `config_hash`
- `started_at` / `completed_at`
- `external_run_ids` (`dagster`, `mlflow`, `sqlmesh`, `otel_trace` …)
- `environment` (local | ci | cloud), not a cloud SDK object

Lineage on an envelope is **refs**, not a graph database:

- `produced_by` run
- `inputs[]` payload ids
- `derived_from[]` if the relationship is more specific than input
- `related[]` typed links (`quality_for`, `evaluation_of`,
  `claims_about`, `reviews`, `champion_of`)

OpenLineage is an **export** of those refs, not the store.

Do not put a lineage service in `ds-platform`. Do not make OpenLineage
types the native model. Job/Run/Dataset in OpenLineage do not cover
claims, reviews, or prompt artifacts cleanly.

### 3.8 Quality and evaluation are artifacts, not flags

A `quality_ok: true` bit on a dataset is how quality becomes
unreproducible.

**Quality** inspects an artifact against expectations (nulls, ranges,
referential integrity, freshness). Result: a `quality` artifact citing
the subject and the contract.

**Evaluation** compares predictions or models against a suite, labels,
or a human rubric. Result: an `evaluation` artifact citing the model or
predictions, the suite, and the scores.

These look similar and must stay distinct. Quality can run on raw XML.
Evaluation requires a task. Champion selection reads evaluations, not
quality reports.

Human evaluation is a `review` artifact, possibly aggregated into an
`evaluation`. It is not a checkbox in a Streamlit app that leaves no
record.

### 3.9 Governance is data, not an engine

Governance that ships as a rules engine will be wrong for two solo
research projects and still wrong for the first team.

What *is* expensive to retrofit:

- every run records `inherited_policy_refs` (license, retention,
  PII class, source ToS)
- artifacts are **write-once**
- aliases (champion, latest) are explicit
- SPDX license expressions are allowed as strings
- review artifacts exist as a kind

What is deferred: OPA/Cedar, SLSA build attestations, in-toto, a
clearance workflow product.

The platform can refuse to *invent* policy. It should not refuse to
*carry* policy identifiers.

### 3.10 Execution: no backend interface

Workloads will include:

- Arrow → Polars → DuckDB on a laptop
- later Arrow/Parquet → Ray or Dask
- rarely Spark, and only when a job is already a Spark job

A generic `ExecutionBackend` does not make those swaps cheap. It makes
every job go through a lowest-common-denominator API that none of the
engines want.

**Decision:** the platform does not execute work.

Reproducibility is `RunContext` + hashed inputs + hashed code/config +
write-once outputs. The job may be a Python function, a Dagster asset,
a SQLMesh model, or a Ray job. The platform records that it happened.

Arrow and Parquet are **conventions at the table boundary**, not
dependencies of `ds-platform`. Core code does not import `pyarrow`,
`polars`, `duckdb`, `pandas`, `ray`, `dask`, or `pyspark`.

When a project writes a table artifact, the payload should be Parquet
(or JSONL for documents). Readers in the project choose the engine.
That is the actual portability mechanism. It is a file format, not a
`TableBackend` protocol.

### 3.11 Storage: one small protocol, one implementation now

Storage *does* change in a way that is painful if paths leak into
domain code: `data/processed/...` today, `s3://...` later.

The justified interface is small:

```
Store
  put(payload_id, bytes, *, media_type) -> Location
  get(payload_id) -> bytes
  exists(payload_id) -> bool
  locate(payload_id) -> Location | None
```

`Location` is a URI. Callers persist `payload_id`. Only the store
adapter knows schemes.

v0 implementation: local filesystem, content-addressed layout
(`ab/cd/<sha256>`). No boto3. An S3 adapter is added when a project
actually stores there.

This is not a "storage backend strategy pattern" with five classes. It
is one protocol because the semantic leak (paths as IDs) is real.

Envelopes are objects too (`record_id` as key, `application/json`).
Do not invent a second store.

A catalog/index that can list a million artifacts is **not** the store.
The store is a kv. The catalog is a later derived `index` artifact or
an external database. Do not query production history by globbing
envelope files once the count is large. Do not build that index now.

### 3.12 Interchange and analytical evolution

The intended evolution:

```
local files → S3
DuckDB on Parquet → Athena on the same Parquet
Polars in-process → Ray/Dask for a specific job
```

The thing that makes this cheap is **stable table artifacts in
Parquet/Arrow IPC**, plus identity that does not include the engine.

Athena is appropriate when data already lives as Parquet in S3 and the
question is SQL over it. Redshift is a warehouse product with its own
load path and concurrency model. It is not on the default path. Do not
architect for Redshift "just in case."

PostgreSQL is appropriate for **mutable operational state**: aliases,
run indexes, UI session data, later application APIs. It is a poor
system of record for write-once scientific artifacts. Do not put
corpus payloads in Postgres because it is already on the resume.

### 3.13 ML and AI

Treat this as a recording and identity problem, not as a
scikit-learn wrapper.

#### Models are artifacts

A trained sklearn pipeline, a torch checkpoint, a LoRA adapter, and a
tokenizer bundle are `model` artifacts. They have `payload_id`s.
MLflow may also register them. MLflow remains the **registry and
experiment UI**. The platform does not replace it.

Champion/challenger is an **alias** (`logical_key` such as
`ri:ranker:champion`) pointing at a `payload_id`. Promotion writes a
new alias record and should cite the `evaluation` that justified it.
Rollback points the alias at a previous `payload_id`. No
`is_champion: true` field on the model.

#### Do not unify call interfaces

Classifiers, rankers, deep models, LLM completions, tool-using agents,
and deterministic algorithms do not share a useful `predict(x)`
abstraction. Forcing one produces a protocol that is either empty or
dishonest.

What *is* shared is the **record** of an invocation:

`InferenceRecord` (design type):

- subject artifact (model and/or prompt)
- input ref
- output ref or inline for tiny outputs
- provider / route (string, not an SDK)
- params hash (temperature, decoding, feature flags)
- token/latency/cost if known
- `run_id`
- optional OpenTelemetry span id

Training produces a `model` artifact and an MLflow run id in
`external_run_ids`. The platform does not own training loops.

#### LLM and agent specifics

For LLM work, "the model" is incomplete. Identity of a generative
system is at least:

- provider model id (informative, not unique)
- prompt artifact hash
- tool/schema hash
- decoding params hash

A change to the prompt is a new system, even if the vendor model
string is unchanged.

Agents are not a model class. An agent run is a **trace**: a tree of
`InferenceRecord`s plus tool results. Store the trace as an artifact
(`document` or a later `trace` kind). Do not build an agent framework.

Multiple inference providers are strings on the record. An
`InferenceProvider` protocol is unwarranted until two providers share
an adapter that is actually reused. Projects may call OpenAI, a local
model, or a custom HTTP endpoint directly.

#### Evaluation and production observation

Offline evaluation and production observation are different artifacts
that may share score schemas.

- Offline: `evaluation` citing a frozen suite
- Production: later `prediction` batches + optional online metrics
  export; not a second MLflow

Human-in-the-loop is `review` artifacts, optionally sampled from
production predictions. The UI that collects the review is
project-specific.

Embeddings are artifacts (`dataset` of vectors plus a `model` /
`prompt` citation). Do not invent an embedding store in this package.

### 3.14 Unified project UI

Streamlit is a reasonable **project** research UI. It is a bad
platform product.

If each project later has a UI for pipeline state, artifacts, lineage,
quality, models, evaluations, champions, and costs, the shared need is
**read models**:

- resolve `logical_key` → `payload_id`
- load envelope by `record_id`
- list artifacts by kind / run
- follow refs

Those read APIs belong in `ds-platform` only when a second project
needs them. Until then, a project can read JSON sidecars.

The UI must not:

- replace Dagster's run view
- replace MLflow's experiment view
- execute pipelines by importing Dagster internals from this package

Triggering a run from Streamlit is a project adapter (subprocess,
Dagster GraphQL, or CI dispatch). Rebuild/promote/rollback are writes
of new artifacts and aliases, not edits.

### 3.15 AWS

AWS is a deployment environment, not a semantic dependency.

| Service | Role | When |
| --- | --- | --- |
| S3 | object store implementing `Store` | when local disk is insufficient or sharing is required |
| PostgreSQL/RDS | alias + run index, app state | when file manifests are not enough |
| ECR | image registry for deployable jobs | when a project ships a container |
| EC2 / ECS | compute | when local/CI compute is not enough |
| Athena | SQL over Parquet in S3 | when that query pattern exists |
| Redshift | not assumed | only if a project needs a warehouse, not a lake query engine |

Core Python code does not import `boto3`. Credentials never appear in
envelopes. Account ids and bucket names are location configuration.

### 3.16 What ships in the package later (not now)

When implementation starts, the first useful surface is:

- identity hashing (canonical JSON, sha256)
- envelope + run context types (Pydantic)
- `Citation` / `Evidence` / `Claim`
- artifact kind enum
- `Store` protocol + local filesystem implementation
- JSON Schema for the envelope

Not in the first implementation: adapters for Dagster, SQLMesh,
MLflow, S3, Streamlit, Ray, or Spark.

---

## 4. Standards

Incorporate a standard only when it saves a conversion that would
otherwise be invented badly.

| Standard | Role | Why |
| --- | --- | --- |
| JSON Schema | **native** for contracts and envelope documents | already the language of the consumer projects; validation is cheap |
| Content addressing (sha256) | **native** identity | cheaper than any catalog id |
| OpenLineage | **export** | widely understood Job/Run/Dataset events; poor fit as the native claim/review model |
| OpenTelemetry | **correlation** | `trace_id` / `span_id` on `RunContext` and `InferenceRecord`; no SDK dependency in core |
| ODCS (Bitol) | **alignment** | field overlap with contracts (schema, quality, ownership); do not require ODCS YAML as the stored form |
| W3C PROV | **conceptual** | Entity/Activity/Agent maps to artifact/run/actor; do not store PROV-JSON |
| MLflow | **external system** | registry + experiment tracking; map via `external_run_ids` and model `payload_id` |
| SPDX | **string field** | license expressions on policy refs |
| CycloneDX | **future export** | environment/SBOM of a run, not artifact identity |
| in-toto / SLSA | **future** | CI provenance for published models; unused until there is a publish path |
| ML Metadata (MLMD) | **excluded as native store** | TFX-centric graph store; optional later indexer, never the contract |
| OCI | **excluded for now** | relevant if models are shipped as images; artifacts here are objects, not images |

OpenLineage and OTel are complementary: one describes data jobs, the
other describes process traces. Neither replaces artifact identity.

---

## 5. Technology roles

These tools work *with* the platform. None are implemented inside it.

| Technology | Role | Conflict / caution |
| --- | --- | --- |
| **Dagster** | default orchestrator for Python assets and schedules | source of `external_run_ids.dagster`; not required for a local script |
| **SQLMesh** | SQL-first transformation mesh **when a project has one** | overlaps Dagster if both "own" transformations; do not adopt as a required peer of Dagster in v0 |
| **MLflow** | experiment tracking and model registry | do not duplicate a second metric store in `ds-platform` |
| **Arrow / Parquet** | table interchange | convention, not a core import |
| **Polars / DuckDB** | default local analytics | project dependencies |
| **Pandas** | interop and ecosystem glue | allowed; not the platform's dataframe type |
| **Ray / Dask** | optional distributed compute for a job | chosen per workload; no wrapper |
| **Spark** | last resort for already-Spark shops or extreme scale | not on the default path |
| **Streamlit** | project research UI | not a control plane product |
| **AWS / S3** | deploy + object store | adapter later; not core |
| **PostgreSQL** | indexes and aliases | not the artifact byte store |
| **Athena** | SQL over S3 Parquet | after that layout exists |
| **Redshift** | omitted from the default architecture | warehouse-specific; high lock-in relative to Athena-on-Parquet |
| **ECR / EC2 / ECS** | packaging and compute | infrastructure, not semantics |

The important load-bearing combination is:

**Parquet artifacts + content hashes + Dagster (or scripts) + MLflow
for models + S3 when needed.**

SQLMesh, Spark, Redshift, and a platform Streamlit app are not part of
that combination until a project demonstrates the need.

---

## 6. Critical review

This section attacks the design above. It is not a summary.

### 6.1 Overall quality

Strength for the intended purpose is **good, not excellent**.

The design is correctly small. It refuses to become a warehouse or an
orchestrator. It separates payload identity from annotation identity
from aliases. It refuses a unified model-call interface. Those are the
judgments that matter for `board-game-analysis` and a future
`restaurant-intelligence`.

Weaknesses that are already visible:

1. **Kind vocabulary can still rot.** Twelve kinds is already a lot.
   `document` vs `dataset` vs `claim_set` will be argued over. If kinds
   proliferate (`trace`, `cost_report`, `card`, `manifest`), the
   "closed vocabulary" becomes a junk drawer. Mitigate by treating new
   kinds as a versioned, deliberate change, not an open string.

2. **The store protocol is still an abstraction.** It is justified, but
   it can be implemented too early. A project can write
   `data/processed/foo.parquet` and a sidecar JSON for months. The
   protocol earns its keep at the *second* backend. Building it before
   the envelope types exist is ceremony.

3. **Claim types may be unused.** `board-game-analysis` currently needs
   layer discipline more than a claim graph. Shipping empty claim
   machinery is still better than mixing layers in domain models — but
   only the *types* should exist. No claim store, no reasoner.

4. **No catalog is a scale cliff.** The design admits this and then
   defers it. That is correct for two projects and dishonest at
   10⁶–10⁷ artifacts. See §6.2.

5. **`logical_key` is underspecified.** Naming schemes, uniqueness
   scope (global vs per project), and alias history are where catalogs
   die. This must be frozen more tightly than the rest of the envelope
   (see Architecture Freeze).

6. **RunContext can become a junk drawer** of every external id.
   `external_run_ids` should be a small typed map, not arbitrary JSON.

The architecture is weaker as a "platform" and stronger as a
**discipline library**. That is the correct trade for the portfolio.
Calling it a platform oversells the runtime and undersells the
contracts. The repository name is fine; the implementation must stay
humble.

### 6.2 Future-proofing

| Pressure | Survival | Risk |
| --- | --- | --- |
| 1M artifacts | envelopes + object store: fine; listing: not fine | implicit assumption that directories are a catalog |
| 10M artifacts | requires an index (Postgres, warehouse, or lakehouse table) | if `payload_id` or kind encodings change, reindex is possible; if identity is path-based, it is not |
| Multiple projects | good if `logical_key` is namespaced by project | collision if keys are global and short |
| Multiple teams | weak; no tenancy, auth, or review workflow | do not fake tenancy now; add `project` + `actor` fields and stop |
| Multiple execution environments | good (`environment` + `external_run_ids`) | leaking host paths into envelopes |
| Cloud migration | good if identity ≠ location | S3 adapter that rewrites ids would be a fatal implementation mistake |
| Distributed compute | good (non-goal to wrap engines) | temptation to add `ExecutionBackend` after the first Ray job |
| Multiple model families | good (models as artifacts, no common predict) | MLflow flavor lock-in if projects store *only* in MLflow |
| LLM / agent workflows | adequate if prompts and traces are artifacts | under-specified trace schema; that can wait |
| Sophisticated evaluation | adequate (eval as artifact, suite as artifact) | no slice/metric protocol yet — add when a second eval exists |
| Complex lineage | refs export to OpenLineage | do not natively store a graph in envelopes |
| Evolving governance | policy refs + write-once | a late policy engine will want to rewrite history; refuse |

Decisions that become extremely expensive later:

- collapsing `payload_id` / `record_id` / `logical_key`
- putting quality/eval/claims *inside* dataset envelopes
- using storage paths as foreign keys in consumer projects
- importing Dagster/MLflow/boto3 in core
- making champion a mutable flag
- treating vendor model names as model identity
- stuffing payloads into PostgreSQL
- adopting OpenLineage types as the native artifact model

### 6.3 Cutting-edge status

Honest placement relative to current serious practice:

| Idea | Status |
| --- | --- |
| Content-addressed artifacts + aliases | established (git, OCI, nix, lakehouse tables) |
| Sidecar metadata | established (MLMD events, OpenLineage, Hugging Face cards) — **conventional**, not novel |
| Write-once + separate annotation | established (object versioning, event sourcing lite) |
| Claims with epistemic layers | **useful and slightly uncommon** in data platforms; closer to scientific computing and knowledge graphs than to typical MLOps |
| Evaluation as a first-class artifact | established in modern AI eval (Inspect, HELM, internal eval harnesses); still missing from many MLOps stacks |
| Prompts as versioned artifacts | **current best practice** for LLM ops; not exotic |
| Agent traces as artifacts | **emerging**; OTel GenAI semantic conventions are the interoperability bet |
| Capability-based model interfaces | often **sounds modern**; usually a weak abstraction — correctly rejected here |
| Swappable `*Backend` suites | **sounds modular**; frequently fake flexibility — correctly rejected except `Store` |
| OpenLineage export | established |
| Avoiding MLMD as core | good judgment; MLMD is not the industry center of gravity |
| Lakehouse path (Parquet + optional Athena) | established; not cutting-edge |
| Human review as an artifact kind | established in labeling systems; often forgotten in homegrown platforms |

This architecture is **modern in the sense of 2024–2026 LLM-aware
MLOps**: artifacts, evals, prompts, traces, aliases. It is not a
research contribution. It should not pretend to be.

It is **behind** current practice in:

- automated eval gates in CI (convention not designed in detail)
- production tracing of generative systems (OTel GenAI is named, not
  mapped)
- data-contract tooling as used in platform teams (this is a thinner
  cousin of ODCS/data-contract-cli)
- catalog/search (intentionally)

It is **not behind** in the place that matters for a two-project
library: it refuses a technology zoo.

More technologies would make it look more "cutting-edge" and worse.

### 6.4 Missing modern ideas

Only items with architectural value:

1. **Prompt + tool schema as part of generative identity.** Included
   above. This was easy to miss if "models are artifacts" is taken to
   mean weights only.

2. **Eval suite versioning.** An evaluation that does not cite a frozen
   suite is not comparable. The suite is an artifact. Add this to the
   freeze; it is a one-line kind policy, not a product.

3. **Human review as a kind.** Included. Without it, HITL leaks into
   UI state.

4. **Port vs dataset contracts.** Included. Prevents "the Game schema
   is the only contract."

5. **OpenTelemetry GenAI semantic conventions** as the *export shape*
   for traces, rather than a custom agent schema. Worth adopting as a
   mapping later; not a native type system now.

6. **Dataset / model cards as artifacts.** Cheap, useful, easy to
   defer. A `document` with a convention is enough; no new kind yet.

7. **Cost records.** LLM and AWS spend will matter. A `cost` field on
   `InferenceRecord` is enough at first. A dedicated kind is
   premature.

8. **Policy-as-code engines, feature stores, vector databases,
   semantic layers, LakeFS/DVC-as-core, Unity Catalog clones.**
   Real products. Wrong to embed. DVC/lakeFS overlap content
   addressing; do not wrap them. If a project wants lakeFS, it can
   still store `payload_id`s.

9. **Differential privacy, formal data access control, confidential
   compute.** Not relevant to the current projects.

10. **Competing-claim resolution and knowledge-graph materialization.**
    Interesting for board-game interpretations. Keep claim identity
    stable so this can appear later. Do not build it.

I am **not** recommending a metadata lake, a feature store, or an
internal LLM gateway. Those are gravity wells.

### 6.5 Irreversible decisions

**Decide now (painful to change after adoption):**

- artifact as root noun; tables are not the root
- `payload_id` = sha256(bytes); location is not identity
- `record_id` separate from `payload_id`
- `logical_key` is an alias with project namespace
- write-once payloads; updates are new artifacts
- quality / evaluation / claims are separate artifacts
- models and prompts are artifacts
- champion is an alias
- core package has no Dagster / SQLMesh / MLflow / boto3 / Ray /
  Spark / Streamlit imports
- OpenLineage is export-only
- JSON Schema for contracts
- Parquet (or JSONL for documents) at table/document boundaries
- `Store` is the only I/O protocol
- no unified predict/agent interface

**Intentionally unfrozen:**

- whether SQLMesh is used in a given project
- DuckDB vs Athena vs (later) something else for SQL
- Polars vs Pandas vs Ray for a job
- Streamlit vs another UI toolkit
- Postgres vs a Parquet manifest for the first catalog
- S3 vs another object store (the URI location model is frozen; the
  vendor is not)
- Pydantic vs msgspec for implementation
- exact kind list beyond the ones that appear in the freeze
- trace schema for agents
- SBOM / SLSA
- any `*Backend` besides `Store`

### 6.6 Abstraction audit

| Abstraction | Verdict | Why |
| --- | --- | --- |
| Artifact (payload + kind + identity) | **keep** | this is the expensive boundary |
| Envelope / record document | **keep, slim** | portable metadata; not a filing cabinet |
| `payload_id` / `record_id` / `logical_key` | **keep** | highest conversion cost if wrong |
| Artifact kinds | **keep, tight** | query and governance need a vocabulary |
| `Citation` / `Evidence` / `Claim` | **keep types; don't store graphs** | layer discipline is the consumer-project bug |
| `RunContext` | **keep** | reproducibility without an executor |
| Contract-as-schema-ref | **keep** | domain stays out of the platform |
| `Store` protocol | **keep, implement late** | location leak is real; second backend not here yet |
| `ExecutionBackend` | **reject** | engines do not share a useful API; migration cost is rewriting the *job*, not an interface |
| `TableBackend` | **reject** | Parquet *is* the port; a protocol adds nothing until someone cannot read Parquet |
| `StorageBackend` (generic) | **reject name** | one `Store` is enough; a family of backends invites fake symmetry |
| `ModelBackend` | **reject** | MLflow + files; no platform model loader |
| `EvaluationBackend` | **reject** | eval is a job that writes an artifact |
| `RegistryBackend` | **reject** | MLflow is the registry; aliases are local records |
| `LineageBackend` | **reject** | export OpenLineage; do not wrap Marquez |
| `InferenceProvider` | **reject for now** | record the provider string; wrap SDKs in projects |
| Capability `Predictor` protocol | **defer** | useful only after two in-repo implementations share code |
| Streamlit "platform app" | **reject** | project UI over read models |
| AWS facade | **reject** | infrastructure as deploy, not a Python package layer |

The heuristic: **abstraction is justified when a consumer project would
otherwise embed a location, a vendor id, or a mixed epistemic layer
into a domain object.** It is not justified when it only makes a
diagram look symmetric.

### 6.7 Technology selection

**Dagster** — correct default orchestrator for Python-heavy projects.
Does not belong in core.

**SQLMesh** — good tool, wrong *required* peer. It overlaps Dagster on
ownership of transform graphs and overlaps "table artifacts" if both
emit the same Parquet. Adopt it in a project that is actually
SQL-mesh-shaped (warehouse models, environments, virtual data
environments). `restaurant-intelligence` might get there.
`board-game-analysis` is not there. **Change from a dual-orchestrator
assumption.**

**MLflow** — correct registry. Conflict only if envelopes grow a second
metrics database. Keep MLflow as the experiment UI; cite run ids.

**Arrow / Parquet / Polars / DuckDB** — correct local stack. Platform
must not import them. The architectural bet is the file format.

**Ray / Dask** — optional, job-scoped. No conflict if jobs still write
Parquet artifacts.

**Spark** — omit from the default story. Adding it "because scale"
before a Spark-shaped problem is how portfolios become zoos.

**AWS / S3** — correct object store when needed. Core stays unaware.

**PostgreSQL** — correct for aliases and indexes. Conflict if used as
the lake.

**Athena** — correct *after* S3+Parquet. Not a v0 concern.

**Redshift** — **do not plan around it.** It pulls loads, WLM, and
warehouse modeling into the architecture for no current workload.

**Streamlit** — fine as a per-project lab. Architectural conflict if it
becomes the place where promotions happen *without* writing alias
artifacts.

Overlap summary: Dagster ∩ SQLMesh on transforms; MLflow ∩ envelopes
on metrics (accept, cite); DuckDB ∩ Athena on SQL (sequence, not
competition); Postgres ∩ S3 on storage (different jobs).

Gap: a real catalog. Accepted for now.

### 6.8 Portfolio value

This architecture demonstrates judgment if the repo stays small and
the first implementation is types + hashing + local store.

It demonstrates **weak** judgment if the next commits add empty
`adapters/dagster`, `adapters/mlflow`, `backends/spark`, and a
Streamlit skeleton. That is a technology zoo. Interviewers and future
you can smell it.

Where it can honestly show capability, once implemented in order:

| Area | How |
| --- | --- |
| Data engineering | ingestion and tables as artifacts with contracts |
| Data architecture | identity, kinds, lake-oriented Parquet, Postgres as index |
| ML engineering | models/predictions as artifacts; no fake estimator base class |
| AI engineering | prompts, traces, eval suites |
| MLOps | MLflow as registry; champion aliases; eval-gated promotion |
| Cloud engineering | S3 `Store` later; IAM and compute remain project deploy |
| Reproducibility | `RunContext` + hashes + write-once |
| Experimentation | MLflow + cited runs |
| Evaluation | evaluation artifacts, not printed scores |
| Production systems | only after aliases + a real serving path exist — do not claim this now |

The architecture is **not** a production system. Claiming otherwise
would be the least impressive thing in the portfolio.

---

## 7. Recommendations

### Keep

- Artifact as the root noun; payload identity as sha256.
- Slim envelope as a sidecar record, not a container for reports.
- Separate `payload_id`, `record_id`, and `logical_key`.
- Write-once artifacts; aliases for latest/champion.
- Citation / evidence / claim as distinct types; claims layered
  `observed | derived | inferred | asserted`.
- Quality, evaluation, and reviews as their own artifacts.
- Models and prompts as artifacts; MLflow as the external registry.
- `RunContext` with `external_run_ids` instead of an executor.
- Parquet/JSONL conventions instead of a table backend.
- One `Store` protocol; AWS kept out of core.
- OpenLineage and OTel as export/correlation, not native stores.
- No unified predict / agent / provider interface.

### Change

- **Do not treat `ArtifactEnvelope` as the center of the architecture.**
  The center is identity + kinds + write-once + refs. The envelope is
  how a record is serialized.
- **Do not inline claims, quality, or evals on dataset envelopes.**
  Point at them.
- **Do not treat SQLMesh as a required peer of Dagster.** Adopt it
  per-project when SQL-mesh workloads exist.
- **Do not put Redshift on the default path.** Athena-on-Parquet is
  the scale SQL option if one is needed.
- **Do not add a `Predictor` protocol or `InferenceProvider` in v0.**
  Freeze `InferenceRecord` fields only.
- **Do not implement `Store` before envelope types exist.** The
  protocol is approved; the code waits.
- **Tighten `logical_key` now:** `{project}:{name}` plus optional
  `:{version}`; uniqueness is per-project; aliases are a history of
  pointers, not overwritten files.

### Defer

- S3 adapter, Athena, ECR, ECS/EC2 wiring
- SQLMesh, Ray, Dask, Spark wrappers (the last three: never wrap)
- Streamlit and any shared UI package
- Catalog/index implementation
- OpenLineage emitter
- OTel SDK dependency
- CycloneDX / in-toto / SLSA
- Agent trace schema beyond "document artifact + inference records"
- Cost-report kind
- Dataset/model card kind
- Claim graph / conflict resolver
- Policy engine
- Adoption by `board-game-analysis` or `restaurant-intelligence`

### Architecture Freeze

These decisions are stable before substantial implementation:

1. **Root noun:** artifact = payload bytes + kind + `payload_id`.
2. **Identity:** `payload_id = sha256(payload)`; `record_id =
   sha256(canonical envelope)`; `logical_key = {project}:{name}[:version]`;
   location is not an id.
3. **Mutation:** payloads are write-once; aliases move.
4. **Record document:** slim envelope (identity, kind, run, refs,
   contract ref, policy refs). No inlined reports.
5. **Separate artifacts:** `quality`, `evaluation`, `claim_set`,
   `review`; evals cite a frozen suite artifact.
6. **Epistemic types:** `Citation` ≠ `Evidence` ≠ `Claim`;
   `Claim.layer` is the four-way enum above.
7. **ML/AI:** models and prompts are artifacts; champion is an alias;
   invocations recorded as `InferenceRecord`; no common predict API.
8. **Peers, not internals:** Dagster, MLflow, cloud SDKs stay outside
   the core package.
9. **Interchange:** tables as Parquet; documents as JSON/JSONL; Arrow
   as the in-memory convention in projects.
10. **I/O:** at most one `Store` protocol; local filesystem first;
    no other `*Backend` types.
11. **Standards:** JSON Schema native; OpenLineage export-only; OTel
    ids only; ODCS aligned not stored; MLMD not native.
12. **Scope:** no domain models; no consumer-project wiring in this
    repository until a later, explicit adoption task.

### Deliberately Unfrozen

- Orchestrator choice per project (Dagster default, not mandatory)
- SQLMesh adoption
- Compute engine per job (Polars, DuckDB, Ray, Dask, Spark)
- Cloud vendor and specific AWS services
- Catalog technology
- UI toolkit
- Serialization library
- Exact extra kinds (`trace`, `card`, …)
- Provider SDKs and any future inference adapter
- When, and whether, to emit OpenLineage
- Environment SBOM / SLSA
- PostgreSQL schema for aliases

---

## 8. Disagreements with the brief

The brief is mostly aimed at the right problems. The places I do not
follow it:

- **`ArtifactEnvelope` as a central concept** — useful document, wrong
  center. Identity and kinds are the center.
- **Intended ecosystem as a peer set** — listing Dagster *and*
  SQLMesh *and* Ray *and* Dask *and* Spark *and* Redshift describes a
  market, not an architecture. The architecture picks a default path
  and treats the rest as optional exits.
- **Capability-based model interface** — not more appropriate than
  inheritance if there is only one implementation. Records beat
  interfaces here.
- **Unified project UI in the platform** — the shared thing is read
  access to records, and even that waits for a second consumer.
- **"Platform"** — the durable value is a contract library. If
  implementation grows runtimes, it has failed this document.

When implementation starts, implement the freeze set in the order:
hashing and canonical JSON → types → local store → nothing else.
