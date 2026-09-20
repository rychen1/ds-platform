"""Import boundary tests for optional vendor integrations."""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def test_core_modeling_import_is_vendor_free_subprocess() -> None:
    script = """
import sys
import ds_platform.modeling
forbidden = {"xgboost", "lightgbm", "catboost", "numpy", "pandas", "sklearn"}
loaded = forbidden.intersection(sys.modules)
if loaded:
    raise SystemExit(f"forbidden modules loaded: {sorted(loaded)}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_integrations_package_import_is_vendor_free_subprocess() -> None:
    script = """
import sys
import ds_platform.integrations
forbidden = {"xgboost", "lightgbm", "catboost", "numpy", "pandas", "sklearn"}
loaded = forbidden.intersection(sys.modules)
if loaded:
    raise SystemExit(f"forbidden modules loaded: {sorted(loaded)}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_xgboost_module_import_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ImportError("No module named 'xgboost'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("ds_platform.integrations.xgboost", None)
    module = __import__(
        "ds_platform.integrations.xgboost",
        fromlist=["XGBoostClassifier"],
    )
    with pytest.raises(ImportError, match="ds-platform\\[xgboost\\]"):
        module.XGBoostClassifier()


def test_lightgbm_module_import_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lightgbm" or name.startswith("lightgbm."):
            raise ImportError("No module named 'lightgbm'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("ds_platform.integrations.lightgbm", None)
    module = __import__(
        "ds_platform.integrations.lightgbm",
        fromlist=["LightGBMClassifier"],
    )
    with pytest.raises(ImportError, match="ds-platform\\[lightgbm\\]"):
        module.LightGBMClassifier()


def test_catboost_module_import_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "catboost" or name.startswith("catboost."):
            raise ImportError("No module named 'catboost'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("ds_platform.integrations.catboost", None)
    module = __import__(
        "ds_platform.integrations.catboost",
        fromlist=["CatBoostClassifier"],
    )
    with pytest.raises(ImportError, match="ds-platform\\[catboost\\]"):
        module.CatBoostClassifier()
