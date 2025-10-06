from __future__ import annotations


def _coerce_str(val) -> str:
    """Coerce None to empty string and non-strings to str."""
    if val is None:
        return ""
    return val if isinstance(val, str) else str(val)


def _normalize_tool_calls(container: dict) -> None:
    """Normalize tool_calls array in either message or delta containers."""
    tool_calls = container.get("tool_calls")
    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            tc.setdefault("id", f"call_{idx}")
            fn = tc.get("function") or {}
            # Ensure arguments are strings
            args = fn.get("arguments")
            if not isinstance(args, str):
                fn["arguments"] = _coerce_str(args)


def normalize_response(resp_dict: dict, *, disabled: bool) -> dict:
    if disabled:
        return resp_dict
    try:
        obj = resp_dict.get("object")
        if obj not in {"chat.completion", "chat.completion.chunk"}:
            return resp_dict

        choices = resp_dict.get("choices") or []
        for c in choices:
            # Non-stream responses use "message"; stream chunks use "delta"
            if obj == "chat.completion.chunk":
                delta = c.get("delta") or {}
                # Coerce content to string to avoid client-side None errors
                if "content" in delta:
                    delta["content"] = _coerce_str(delta.get("content"))
                # Normalize tool_calls if present in delta
                _normalize_tool_calls(delta)
            else:
                msg = c.get("message") or {}
                if "content" in msg:
                    msg["content"] = _coerce_str(msg.get("content"))
                # Legacy single function_call -> tool_calls
                function_call = msg.get("function_call")
                tool_calls = msg.get("tool_calls")
                if function_call and not tool_calls:
                    tc = {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": function_call.get("name", "unknown"),
                            "arguments": _coerce_str(
                                function_call.get("arguments", "{}")
                            ),
                        },
                    }
                    msg["tool_calls"] = [tc]
                    msg.pop("function_call", None)
                # Ensure tool_calls arguments are strings
                _normalize_tool_calls(msg)

        return resp_dict
    except Exception:  # noqa: BLE001
        # Be conservative: if normalization fails, return original response
        return resp_dict
