# ds-platform

Reusable semantic contracts for data, ML, and AI project artifacts.

This repository will become a versioned Python library consumed by real
projects. It is **not** an orchestrator, warehouse, model registry, or
application framework. Mature systems such as Dagster, SQLMesh, MLflow, and
AWS remain external.

## Status

Pre-alpha. The package exists so the architecture can live in a dedicated
repository. Implementation of identity, envelopes, stores, and integrations
has **not** started.

Read [docs/architecture.md](docs/architecture.md) before writing code. That
document is the current source of truth: independent design, critical
review, and the freeze set.

## What this is for

Projects that need stable answers to:

- what is an artifact, and how is it identified
- how metadata travels with data without becoming the data
- how provenance, lineage, quality, and evaluation stay separable
- how models, predictions, and LLM/agent traces are recorded
- how local work later moves to object storage and cloud compute
  without rewriting domain code

Intended first consumers (not connected yet):

- `board-game-analysis`
- `restaurant-intelligence`

## Non-goals

- Recreating Dagster, SQLMesh, MLflow, or a cloud control plane
- Domain models for any consumer project
- Swappable-backend interfaces for technologies not in use
- A Streamlit product UI inside this package

## Development

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run pyright
```

## License

MIT. See [LICENSE](LICENSE).
