"""Unit test package for xbitinfo."""

import importlib

import pytest
from packaging.version import InvalidVersion, Version


def _import_or_skip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            try:
                if Version(mod.__version__) < Version(minversion):
                    raise ImportError("Minimum version not satisfied")
            except InvalidVersion:
                raise ImportError(f"Invalid version for {modname}: {mod.__version__}")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_julia, requires_julia = _import_or_skip("julia")
