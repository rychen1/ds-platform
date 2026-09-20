"""Optional vendor integrations for ds-platform modeling adapters.

Core ``ds_platform.modeling`` remains vendor-neutral. Import integration
modules explicitly when an optional extra is installed:

- ``ds_platform.integrations.xgboost``
- ``ds_platform.integrations.lightgbm``
- ``ds_platform.integrations.catboost``

This package does not import vendor libraries at import time.
"""

__all__: list[str] = []
