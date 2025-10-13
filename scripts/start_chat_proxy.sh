#!/usr/bin/env bash
set -euo pipefail

# Start the ImageWorks chat proxy.
# Uses current virtual environment / uv to run the FastAPI app.

export CHAT_PROXY_SUPPRESS_DECORATIONS=${CHAT_PROXY_SUPPRESS_DECORATIONS:-1}
export CHAT_PROXY_INCLUDE_NON_INSTALLED=${CHAT_PROXY_INCLUDE_NON_INSTALLED:-0}

exec uv run python -m imageworks.chat_proxy.app
