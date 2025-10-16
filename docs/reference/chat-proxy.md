## ImageWorks Chat Proxy (OpenAI-compatible)

The chat proxy exposes a minimal OpenAI-compatible API over your ImageWorks registry so UIs like OpenWebUI can select and chat with models using a single base URL. It unifies naming, forwards requests to backends (Ollama, vLLM, etc.), and applies a few sensible defaults.

### Endpoints
- GET `/v1/health` – readiness of the proxy itself
- GET `/v1/models` – OpenAI-compatible model list
- POST `/v1/chat/completions` – Chat with text/vision/tool support
- GET `/v1/metrics` – Optional rolling metrics (disabled by default)

### Naming
- Returns simplified names that match the CLI list by default.
- Quantization is appended in parentheses when known, e.g. `llama 12.2b (Q4 K M)`.
- Decorations like backend/format are suppressed by default to keep names clean.

### Default behavior
- Suppress extra decorations in model list: `CHAT_PROXY_SUPPRESS_DECORATIONS=1`
- Exclude non-installed entries (installed-only): `CHAT_PROXY_INCLUDE_NON_INSTALLED=0`
  - Installed-only means the proxy will only list entries whose `download_path` exists on disk from inside the proxy's filesystem view.
  - If you run the proxy in Docker, either:
    - Mount your HF weights directory into the container at the same absolute path, or
    - Set `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` to list logical entries even if files aren’t visible in the container.

### Environment variables
| Variable | Purpose | Default |
|---------|---------|---------|
| CHAT_PROXY_HOST | Bind address | 127.0.0.1 |
| CHAT_PROXY_PORT | Listen port | 8100 |
| CHAT_PROXY_SUPPRESS_DECORATIONS | Hide backend/format/quant fields in models output | 1 |
| CHAT_PROXY_INCLUDE_NON_INSTALLED | Include entries without visible files | 0 |
| CHAT_PROXY_LOOPBACK_ALIAS | Replace localhost targets in backend URLs (e.g. `host.docker.internal`) | *(unset)* |
| CHAT_PROXY_ENABLE_METRICS | Enable `/v1/metrics` | 0 |
| CHAT_PROXY_REQUIRE_TEMPLATE | Enforce presence of chat template | 1 |
| CHAT_PROXY_MAX_IMAGE_BYTES | Max decoded image size | 6000000 |
| CHAT_PROXY_BACKEND_TIMEOUT_MS | Upstream request timeout | 120000 |
| CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS | Streaming idle cutoff | 60000 |
| CHAT_PROXY_LOG_PATH | JSONL chat log (rotates by size) | logs/chat_proxy.jsonl |
| CHAT_PROXY_MAX_LOG_BYTES | Log rotation threshold in bytes | 25000000 |
| CHAT_PROXY_DISABLE_TOOL_NORMALIZATION | Pass through backend tool payloads unchanged | 0 |
| CHAT_PROXY_LOG_PROMPTS | Include prompt payloads in JSONL log | 0 |
| CHAT_PROXY_SCHEMA_VERSION | Schema version advertised in models list | 1 |
| CHAT_PROXY_AUTOSTART_ENABLED | Enable backend autostart commands | 0 |
| CHAT_PROXY_AUTOSTART_MAP | JSON map of logical model → command | *(unset)* |
| CHAT_PROXY_AUTOSTART_GRACE_PERIOD_S | Delay before the proxy marks autostart failures | 120 |

### Logging & autostart
- Requests are appended to `CHAT_PROXY_LOG_PATH` in JSONL format. When the file
  exceeds `CHAT_PROXY_MAX_LOG_BYTES` it is truncated and restarted. Set
  `CHAT_PROXY_LOG_PROMPTS=1` to include full prompt payloads (disabled by
  default).
- Autostart is optional. Enable it with `CHAT_PROXY_AUTOSTART_ENABLED=1` and
  supply `CHAT_PROXY_AUTOSTART_MAP` (JSON mapping logical model names to shell
  commands). The proxy waits `CHAT_PROXY_AUTOSTART_GRACE_PERIOD_S` seconds
  before reporting a startup failure.

### Docker Compose usage

The repo includes `docker-compose.openwebui.yml` with a `chat-proxy` service and an `openwebui` service wired together. Key bits:

```yaml
services:
  chat-proxy:
    build:
      context: .
      dockerfile: Dockerfile.chat-proxy
    environment:
      - CHAT_PROXY_HOST=0.0.0.0
      - CHAT_PROXY_PORT=8100
      - CHAT_PROXY_SUPPRESS_DECORATIONS=1
      - CHAT_PROXY_INCLUDE_NON_INSTALLED=0
      - CHAT_PROXY_LOOPBACK_ALIAS=host.docker.internal
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:8100/v1/health"]
    volumes:
      # Mount HF weights into the container at the same absolute path so installed-only checks pass
      - /home/you/ai-models/weights:/home/you/ai-models/weights:ro
    extra_hosts:
      - host.docker.internal:host-gateway

  openwebui:
    image: ghcr.io/open-webui/open-webui:latest
    depends_on:
      chat-proxy:
        condition: service_healthy
    environment:
      - OPENAI_API_BASE_URL=http://chat-proxy:8100/v1
      - OPENAI_API_KEY=EMPTY
      - ENABLE_OLLAMA_API=false
      - OLLAMA_BASE_URLS=
      - OLLAMA_API_CONFIGS=
      - RESET_CONFIG_ON_START=true
```

### Verification
- From host: `curl -s http://127.0.0.1:8100/v1/models`
- From the OpenWebUI container: `docker exec -t openwebui curl -s http://chat-proxy:8100/v1/models`

If you see fewer models than expected, it’s usually because the proxy can’t see your HF paths. Add the HF mount (above) or set `CHAT_PROXY_INCLUDE_NON_INSTALLED=1`.

### Tool normalization
Some backends return legacy `function_call` fields. The proxy can synthesize a standard `tool_calls` array for OpenAI compatibility. Disable with `CHAT_PROXY_DISABLE_TOOL_NORMALIZATION=1` if you prefer raw passthrough.

### Security note
There is no authentication in Phase 1. Keep the proxy bound to localhost for single-machine use or place behind your reverse proxy with auth if exposing beyond your LAN.
