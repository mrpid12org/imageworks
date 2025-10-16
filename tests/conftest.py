import shutil
import sys
from pathlib import Path

import pytest

from imageworks.model_loader import registry as registry_module

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def isolated_configs_dir(tmp_path, monkeypatch):
    """Provide an isolated copy of the configs directory and chdir into it.

    Ensures tests never mutate the real registry files. Also points the registry
    loader to the copied directory via IMAGEWORKS_REGISTRY_DIR.
    """

    source = ROOT / "configs"
    dest = tmp_path / "configs"
    shutil.copytree(source, dest)
    monkeypatch.setenv("IMAGEWORKS_REGISTRY_DIR", str(dest))
    registry_module._REGISTRY_CACHE = None  # type: ignore[attr-defined]
    registry_module._REGISTRY_PATH = None  # type: ignore[attr-defined]
    yield dest
    registry_module._REGISTRY_CACHE = None  # type: ignore[attr-defined]
    registry_module._REGISTRY_PATH = None  # type: ignore[attr-defined]
