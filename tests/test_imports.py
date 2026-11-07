"""Every module of the package must import cleanly.

Guards against packaging mistakes, e.g. a source folder accidentally excluded
from git by an over-broad .gitignore rule (this happened once with data/).
"""

import importlib
import pkgutil

import alpha_research


def test_every_module_imports():
    names = [m.name for m in pkgutil.walk_packages(alpha_research.__path__, prefix="alpha_research.")]
    assert "alpha_research.data.clean" in names            # the data layer is part of the package
    for name in names:
        importlib.import_module(name)
