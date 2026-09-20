# Boosting library integrations

Optional vendor adapters live under `ds_platform.integrations`. Core
`ds_platform.modeling` remains vendor-neutral.

## Installation

```bash
pip install 'ds-platform[xgboost]'
pip install 'ds-platform[lightgbm]'
pip install 'ds-platform[catboost]'
pip install 'ds-platform[boosting]'   # all three
```

Each extra installs `numpy` and `pandas` for `FeatureTable` conversion.
The XGBoost and LightGBM extras also install `scikit-learn` because their
sklearn-compatible estimator wrappers require it. CatBoost does not.

## Usage

```python
from ds_platform.modeling import ExperimentSpec, run_experiment
from ds_platform.integrations.xgboost import XGBoostClassifier, MEDIA_TYPE
from ds_platform.integrations.lightgbm import LightGBMClassifier
from ds_platform.integrations.catboost import CatBoostClassifier

adapter = XGBoostClassifier(
    params={"n_estimators": 100, "max_depth": 4},
    random_state=spec.seed,
)

result = run_experiment(
    spec,
    rows=rows,
    feature_views=views,
    adapter=adapter,
    store=store,
    run=run,
    serialize_model=lambda model: model.serialize(),
    model_media_type=MEDIA_TYPE,
)

reloaded = XGBoostClassifier.deserialize(store.get(result.model_payload_id))
```

Public adapter entry points:

- `ds_platform.integrations.xgboost`: `XGBoostClassifier`, `XGBoostRegressor`
- `ds_platform.integrations.lightgbm`: `LightGBMClassifier`, `LightGBMRegressor`
- `ds_platform.integrations.catboost`: `CatBoostClassifier`, `CatBoostRegressor`

Adapters wrap vendor estimators. They are not drop-in replacements for
vendor classes.

## Feature conversion

Adapters convert `FeatureTable` → pandas DataFrame internally:

- column order and names are preserved
- row order follows `entity_ids`
- `None` becomes NaN
- booleans become integers

No vendor types appear in core `FeatureTable`.

## Categorical features

Core `FeatureTable` does not encode column kinds. Specify categorical
columns at the integration boundary:

```python
XGBoostClassifier(categorical_columns=("borough", "cuisine"))
LightGBMClassifier(categorical_columns=("borough", "cuisine"))
CatBoostClassifier(cat_features=("borough", "cuisine"))
```

CatBoost uses native categorical handling when `cat_features` is set.

Text features, embedding features, ranking, and early stopping are not
supported by these adapters yet.

## Serialization

Each adapter exposes:

- `serialize() -> bytes`
- `deserialize(data) -> adapter`

Native vendor formats are embedded in a small deterministic integration
envelope (canonical JSON header + native payload):

| Library | Media type | Native payload |
| --- | --- | --- |
| XGBoost | `application/x-xgboost+ubj` | UBJSON model |
| LightGBM | `application/x-lightgbm+txt` | Booster text model |
| CatBoost | `application/x-catboost+cbm` | CBM binary model |

Pass `serialize_model=lambda adapter: adapter.serialize()` to
`run_experiment`.

## Seeds and reproducibility

Pass `random_state=spec.seed` (XGBoost/LightGBM) or
`random_state=spec.seed` (CatBoost maps to `random_seed`) when
constructing adapters. Duplicate seed keys in `ModelSpec.params` are
ignored in favor of the explicit adapter argument.

Bit-for-bit reproducibility is not guaranteed across library versions,
CPU/GPU backends, or thread counts.

## Optional dependency behavior

`import ds_platform.modeling` does not import vendor libraries.
Constructing an adapter checks for its optional dependency and raises a
clear `ImportError` with install instructions when missing.

## Intentionally not supported

- Ranking (`XGBRanker`, `LGBMRanker`, `CatBoostRanker`)
- Early stopping / validation-set wiring through `run_experiment`
- CatBoost text and embedding feature types
- GPU-specific configuration in platform adapters
- Model registries or `ModelSpec.family` dispatch
