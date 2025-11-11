import os

import pytest

from imageworks.chat_proxy import config_loader


@pytest.fixture(autouse=True)
def clear_chat_proxy_env(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("CHAT_PROXY_") and key != config_loader.CONFIG_FILE_ENV:
            monkeypatch.delenv(key, raising=False)
    yield


def test_load_proxy_config_creates_file(tmp_path, monkeypatch):
    config_path = tmp_path / "proxy.toml"
    monkeypatch.setenv(config_loader.CONFIG_FILE_ENV, str(config_path))

    cfg = config_loader.load_proxy_config()

    assert config_path.exists()
    assert cfg.port == 8100
    assert cfg.max_image_bytes == 6_000_000
    assert cfg.config_file_path == str(config_path)


def test_update_config_file_writes_changes(tmp_path, monkeypatch):
    config_path = tmp_path / "proxy.toml"
    monkeypatch.setenv(config_loader.CONFIG_FILE_ENV, str(config_path))

    cfg = config_loader.update_config_file(
        {"port": 8201, "max_image_bytes": 7_500_000, "loopback_alias": "proxy.local"}
    )

    file_cfg = config_loader.load_file_config()

    assert file_cfg["port"] == 8201
    assert file_cfg["max_image_bytes"] == 7_500_000
    assert file_cfg["loopback_alias"] == "proxy.local"

    # Runtime view should mirror the file when no env overrides are present
    assert cfg.port == 8201
    assert cfg.max_image_bytes == 7_500_000


def test_env_overrides_take_precedence(tmp_path, monkeypatch):
    config_path = tmp_path / "proxy.toml"
    monkeypatch.setenv(config_loader.CONFIG_FILE_ENV, str(config_path))
    config_loader.update_config_file({"port": 8101})

    monkeypatch.setenv("CHAT_PROXY_PORT", "9010")

    cfg = config_loader.load_proxy_config()
    file_cfg = config_loader.load_file_config()
    env_overrides = config_loader.list_env_overrides()

    assert file_cfg["port"] == 8101
    assert cfg.port == 9010  # environment wins at runtime
    assert env_overrides["CHAT_PROXY_PORT"] == "9010"
