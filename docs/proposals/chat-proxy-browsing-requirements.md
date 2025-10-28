# Browsing Tool Support Requirements

## Overview
- Extend the chat proxy so selected models can invoke a web-browsing tool during conversations.
- Present capabilities through the OpenAI-compatible API so UI clients (e.g., OpenWebUI) can toggle browsing on models that support it.
- Maintain centralised control over outbound network activity and enforce logging, timeouts, and rate limits.

## Goals
- Allow tool-capable models (starting with gpt-oss-20b) to resolve the “no internet access” limitation.
- Standardise how tools are declared, routed, executed, and logged so additional tools (Python exec, structured outputs, etc.) can be added later.
- Keep the behaviour opt-in per model via registry metadata.

## Non-Goals
- Building a full browser UI or custom search backend.
- Implementing every Harmony tool immediately (focus on browsing first).
- Altering existing model prompting or Harmony templates.

## High-Level Design
- **Registry metadata**
  - Add a `tool_use` capability flag in curated entries for models that can use tools.
  - Optionally store per-model defaults (e.g., preferred tool list) alongside `generation_defaults`.
- **Tool registry (proxy)**
  - Central mapping of tool names → executor callables.
  - Each executor returns structured JSON that matches the tool schema.
  - Browsing executor wraps a configurable HTTP client hitting a search provider / summariser.
- **Payload transformation**
  - When forwarding a request for a tool-capable model, attach the tool definitions (`tools` array, `tool_choice`) to the payload if the client did not supply them.
  - Ensure multi-turn tool interactions are preserved in the conversation history.
- **Response handling loop**
  - Detect tool call instructions in model responses (OpenAI function-call format or Harmony’s channel markers).
  - Execute the tool locally, append the tool output as a new assistant/tool message, and resubmit to the model until a normal assistant message is produced.
  - Guard against infinite loops with a max tool-call depth.
- **Client visibility**
  - `/v1/models` should return tool metadata for supported models so clients can surface the feature.
  - Expose a configuration switch to disable tool injection if a client handles the full tool protocol itself.

## Browsing Tool Specification
- **Name**: `browser.search` (initial function)
  - Inputs: `query` (string), optional `top_n`, optional `geography`.
  - Output schema: `results` array containing `{ title, url, snippet, source }`, plus metadata (`provider`, `latency_ms`).
- Optionally add `browser.page` later to fetch full page content with content-length safeguards.

## Proxy Configuration & Controls
- Environment variables for enabling tools globally, selecting search provider, API keys, timeout values, and maximum response size.
- Allow per-model overrides (e.g., disable browsing on a model even if global flag is on).
- Provide a kill switch that forces “tools disabled” responses without code changes.

## Security & Compliance
- Enforce outbound request allow-list (domains/providers) and upper bound on fetched content size.
- Sanitize and truncate tool outputs before returning them to models.
- Log each execution with timestamp, model id, query, provider, and latency; redact user-identifiable data if necessary.

## Telemetry & Observability
- Metrics: tool invocation count, success vs failure, average latency, provider error rate.
- Structured logs for troubleshooting and auditing.
- Optional sampling of tool inputs/outputs to a secure log store for debugging.

## Open Questions / Follow-Up
- Which search provider/API should we adopt first (SerpAPI, Brave, in-house index)?
- Do we require caching of frequent queries to reduce cost/latency?
- How should user authentication interact with tool permissions (per user vs per workspace)?
- When introducing additional tools (Python, structured outputs), do we enforce mutual exclusion or allow concurrent tool usage?
- Should we expose tool usage stats back to the UI?
