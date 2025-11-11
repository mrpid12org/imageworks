# Chat Proxy Configuration Guide

The chat proxy now reads its defaults from `configs/chat_proxy.toml`. This file is a
human-editable TOML document that mirrors the settings exposed in the GUI’s “Proxy
Settings” tab.

## Precedence

Settings are resolved in the following order:

1. **Environment variables** – any variable prefixed with `CHAT_PROXY_` overrides both the config file and built-in defaults (for example `CHAT_PROXY_PORT=9001`).
2. **Config file** – values saved in `configs/chat_proxy.toml`.
3. **Built-in defaults** – the dataclass defaults shipped with ImageWorks.

The `/v1/config/chat-proxy` API and the GUI both report which environment overrides are currently active so it’s easy to see why a value might not change.

## Editing the config

There are two ways to update the config:

- **GUI** – open the Models → Advanced → “🛠️ Proxy Settings” tab, adjust the values, and click **Save Changes**. The GUI writes to `configs/chat_proxy.toml` and reminds you to restart the proxy so the new defaults take effect.
- **Manual edit** – edit `configs/chat_proxy.toml` directly. The file is created automatically on first launch and grouped into sections (server, timeouts, limits, vLLM defaults, etc.).

> ⚠️ Changes do not hot-reload the running proxy. Restart the `imageworks-chat-proxy` service (via the GUI button or `docker restart imageworks-chat-proxy`) after editing.

### Vision preprocessing defaults

Vision requests now have their own section in `configs/chat_proxy.toml`:

- `max_image_pixels` – fallback long-edge size (defaults to 448 px).
- `image_jpeg_quality` – quality when we recompress inline `data:` images.
- `vision_downscale_backends` – list of backend identifiers (e.g. `["vllm"]`) that should receive the resized copies. Backends not listed (Ollama by default) receive the original uploads so their own preprocessors can run.

These values can also be overridden via environment variables such as `CHAT_PROXY_VISION_DOWNSCALE_BACKENDS="vllm,lmdeploy"`.

## API endpoints

The chat proxy exposes two helper endpoints:

- `GET /v1/config/chat-proxy` returns the current runtime view (after environment overrides), the file defaults, config path, and override list.
- `PUT /v1/config/chat-proxy` accepts a JSON object with updates (only the keys being changed are required). The endpoint writes to the TOML file and returns an updated snapshot; environment variables still win until they are unset.

These endpoints back the GUI but can be scripted as needed.
