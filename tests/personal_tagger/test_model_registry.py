"""Ensure legacy in-code model registry is gone.

Import should raise ImportError (sentinel or absence). Any success is a regression.
"""

import importlib
import pytest


def test_legacy_model_registry_removed():
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("imageworks.apps.personal_tagger.core.model_registry")
