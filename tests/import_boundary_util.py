"""Helpers for vendor-free import boundary tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def assert_import_does_not_pull(module: str, forbidden: set[str]) -> None:
    """Assert importing ``module`` in a fresh process does not load forbidden deps."""
    forbidden_literal = repr(sorted(forbidden))
    script = f"""
import importlib
import sys

before = set(sys.modules)
importlib.import_module({module!r})
loaded = set(sys.modules) - before
forbidden = set({forbidden_literal})
bad = forbidden & loaded
if bad:
    raise SystemExit(f"forbidden modules loaded: {{sorted(bad)}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
