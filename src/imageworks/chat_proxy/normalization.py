from __future__ import annotations


def normalize_response(resp_dict: dict, *, disabled: bool) -> dict:
    if disabled:
        return resp_dict
    try:
        if resp_dict.get("object") not in {"chat.completion", "chat.completion.chunk"}:
            return resp_dict
        choices = resp_dict.get("choices") or []
        for c in choices:
            msg = c.get("message") or {}
            function_call = msg.get("function_call")
            tool_calls = msg.get("tool_calls")
            # Legacy single function_call -> tool_calls
            if function_call and not tool_calls:
                tc = {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": function_call.get("name", "unknown"),
                        "arguments": (
                            function_call.get("arguments", "{}")
                            if isinstance(function_call.get("arguments"), str)
                            else str(function_call.get("arguments"))
                        ),
                    },
                }
                msg["tool_calls"] = [tc]
                msg.pop("function_call", None)
            else:
                # Ensure each tool call has id + string arguments
                if tool_calls:
                    for idx, tc in enumerate(tool_calls):
                        tc.setdefault("id", f"call_{idx}")
                        fn = tc.get("function") or {}
                        args = fn.get("arguments")
                        if not isinstance(args, str):
                            fn["arguments"] = str(args)
        return resp_dict
    except Exception:  # noqa: BLE001
        return resp_dict
