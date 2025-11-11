import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def proxy_app(monkeypatch, tmp_path):
    config_path = tmp_path / "proxy.toml"
    monkeypatch.setenv("CHAT_PROXY_CONFIG_FILE", str(config_path))
    for key in list(os.environ.keys()):
        if key.startswith("CHAT_PROXY_") and key != "CHAT_PROXY_CONFIG_FILE":
            monkeypatch.delenv(key, raising=False)

    import imageworks.chat_proxy.config as config_module
    import imageworks.chat_proxy.config_loader as loader_module
    import imageworks.chat_proxy.app as app_module

    importlib.reload(config_module)
    importlib.reload(loader_module)
    loader_module.update_config_file({})
    importlib.reload(app_module)

    client = TestClient(app_module.app)
    return loader_module, client, config_path


def test_read_proxy_config_endpoint(proxy_app):
    loader_module, client, config_path = proxy_app

    response = client.get("/v1/config/chat-proxy")
    assert response.status_code == 200

    payload = response.json()
    assert payload["config_file_path"] == str(config_path)
    assert "runtime" in payload
    assert "file" in payload


def test_update_proxy_config_endpoint(proxy_app):
    loader_module, client, _ = proxy_app

    response = client.put(
        "/v1/config/chat-proxy",
        json={"port": 8123, "max_image_bytes": 5_500_000},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "written"
    assert payload["file"]["port"] == 8123
    assert payload["file"]["max_image_bytes"] == 5_500_000

    file_cfg = loader_module.load_file_config()
    assert file_cfg["port"] == 8123
    assert file_cfg["max_image_bytes"] == 5_500_000
