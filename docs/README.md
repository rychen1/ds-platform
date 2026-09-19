# Documentation

| Document | Purpose |
| --- | --- |
| [architecture.md](architecture.md) | Independent architecture, critical review, freeze set (implemented core) |
| [modeling-architecture-plan-v1.md](modeling-architecture-plan-v1.md) | Reusable DS/ML contracts (`ds_platform.modeling` v1 + representation foundation implemented; retrieval/RAG still proposed) |

The first slice (identity, types, LocalStore) is implemented. Modeling v1
(specs, features, split, evaluate, artifact helpers, holdout runner) ships in
`ds_platform.modeling` and is imported separately from the top-level package.
The architecture document remains the source of truth for the artifact model.
