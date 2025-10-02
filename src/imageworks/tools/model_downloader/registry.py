"""Legacy downloader registry module (deprecated).

This module has been superseded by the unified deterministic registry in
`imageworks.model_loader.registry` plus the download adapter in
`imageworks.model_loader.download_adapter`.

Any attempt to import and use the old API now raises an ImportError to make
the transition explicit. Update downstream code to use the unified adapter:

  from imageworks.model_loader.download_adapter import record_download, list_downloads, remove_download

If you intentionally need historical JSON from the old system, retrieve it
from any existing `models.json` file directly (read-only) without using this API.
"""

RAISE_MESSAGE = (
    "imageworks.tools.model_downloader.registry is deprecated. Use the unified registry via "
    "imageworks.model_loader.registry and download_adapter (record_download, list_downloads, remove_download)."
)


class ModelRegistry:  # pragma: no cover - legacy shim for old tests
    """Very small backward-compatible shim for old tests.

    The historical downloader had a JSON file based ModelRegistry abstraction.
    Tests still import `ModelRegistry` from this module to construct a registry
    path fixture. We keep a minimal shell so those tests don't explode while the
    new unified deterministic registry is adopted. All functionality has moved
    to `imageworks.model_loader.download_adapter` & friends.
    """

    def __init__(self, path):  # noqa: D401
        self.path = path
        # ensure directory exists for any test writing temporary file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def save(self):  # noqa: D401
        # legacy no-op
        return self.path


class _LegacyRegistryStub:
    def __init__(self, *_, **__):  # noqa: D401
        raise ImportError(RAISE_MESSAGE)


def get_registry():  # noqa: D401
    raise ImportError(RAISE_MESSAGE)


def find_existing_model(*_, **__):  # noqa: D401
    raise ImportError(RAISE_MESSAGE)


__all__ = ["get_registry", "find_existing_model", "ModelRegistry"]
