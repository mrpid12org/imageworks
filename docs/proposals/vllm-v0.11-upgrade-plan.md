# vLLM 0.11 Upgrade Plan

## Objectives
- Restore parity with the working 0.10.2 deployment while adopting vLLM 0.11 features.
- Re-enable `/v1/chat/completions` for Qwen2.5 VL models without empty responses.
- Keep Docker and bare-metal installs aligned (dependency set, launch flags, health probes).
- Validate structured-output pathways (JSON/text), tool-calling, and multimodal workloads.

## Requirements
1. **Functional parity**
   - Chat completions must return textual content for Qwen2.5-VL AWQ and other registry models.
   - Tool-calling (`tool_calls`, `response_format`) must round-trip via the proxy.
   - Streaming, metrics, and health endpoints behave as in 0.10.2.
2. **Structured output support**
   - Container image includes the optional dependencies required by vLLM 0.11 structured output (`xgrammar >= 0.1.25`).
   - Proxy should pass `response_format` when requested and handle the new error semantics.
3. **Launch configuration**
   - Registry extras cover the new defaults: `--tool-call-parser openai`, `--response-role assistant`, `--chat-template-content-format openai`, `--return-tokens-as-token-ids`, etc.
   - Allow feature flag to switch between `/v1/chat/completions` and `/v1/completions` fallback while migrating.
4. **Testing**
   - Automated: extend chat proxy integration tests to exercise both completion paths and structured output.
   - Manual: run dockerized OpenWebUI against vLLM 0.11 and verify typical prompts, tools, and multimodal payloads.
5. **Roll-out**
   - Provide rollback instructions (pin image to 0.10.2) and document the new env vars.

## Implementation Plan
1. **Dependency + image updates**
   - Update `Dockerfile.chat-proxy` to install `vllm[vision,xgrammar]==0.11.x` once ready.
   - Document matching change for local `uv` environment.
2. **Proxy adjustments**
   - Detect backend version (or feature flag) to decide when to add `response_format` and structured-output keys.
   - Add fallback path that rebuilds prompts and calls `/v1/completions` if chat returns empty/invalid payload.
   - Surface a configuration toggle (`CHAT_PROXY_VLLM_USE_COMPLETIONS_FALLBACK`).
3. **Registry changes**
   - Ensure Qwen entries include the extra args mentioned above; audit other vLLM models.
   - Provide optional knob for `--compilation-config` in case FULL_AND_PIECEWISE causes regressions.
4. **Testing work**
   - Add regression tests hitting `/v1/chat/completions` with dummy vLLM mocked responses reflecting 0.11 behaviour.
   - Update integration harness to spin the container with vLLM 0.11 and run sanity prompts.
5. **Documentation**
   - Update `docs/runbooks/vllm-deployment-guide.md` with new flags, structured-output notes, and troubleshooting.
   - Note dependency on `xgrammar` and new defaults.

## Open Questions / Follow-ups
- Confirm whether Qwen AWQ requires special `response_format` settings in future vLLM releases.
- Decide if we want to expose structured-output schemas via the proxy or keep pass-through behaviour.
- Assess GPU memory impact of v0.11 defaults (cudagraph FULL_AND_PIECEWISE) on 16 GB cards.
