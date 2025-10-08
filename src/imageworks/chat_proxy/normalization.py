from __future__ import annotations
import json
import re


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


def _maybe_extract_function_call_from_content(content: str) -> dict | None:
    """Heuristic: if model emitted a function call as inline JSON/XML in content,
    extract it and return {name, arguments}.

    Handles patterns like:
    - ```json\n{"name":"foo","arguments":{...}}\n```
    - ```\n{...}\n```
    - <response> {"name": "foo", "arguments": {...}} </response>
    - Raw JSON object text
    """
    if not content or not isinstance(content, str):
        return None
    s = content.strip()
    # Strip code fences
    if s.startswith("```"):
        # remove first fence line
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) == 2 else ""
        # remove trailing fence
        if s.endswith("```"):
            s = s[:-3]
    # Strip simple xml wrapper
    s = re.sub(r"</?response[^>]*>", "", s, flags=re.IGNORECASE)
    # Find first JSON object
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("name")
            if not name and isinstance(obj.get("function"), dict):
                name = obj["function"].get("name")
                args = obj["function"].get("arguments")
            else:
                args = obj.get("arguments")
            if name:
                return {"name": name, "arguments": args}
    # Try XML-style function_call as fallback (or primary if no JSON)
    try:
        name_m = re.search(r"<name>([\s\S]*?)</name>", s, flags=re.IGNORECASE)
        args_m = re.search(r"<arguments>([\s\S]*?)</arguments>", s, flags=re.IGNORECASE)
        if name_m:
            name = name_m.group(1).strip()
            arguments_raw = (args_m.group(1) if args_m else "").strip()
            # If arguments look like JSON, keep; otherwise pass as string
            try:
                parsed_args = json.loads(arguments_raw)
            except Exception:
                parsed_args = arguments_raw
            if name:
                return {"name": name, "arguments": parsed_args}
    except Exception:
        pass
    return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    # Some models might nest under {function: {name, arguments}}
    if not name and isinstance(obj.get("function"), dict):
        name = obj["function"].get("name")
        args = obj["function"].get("arguments")
    else:
        args = obj.get("arguments")
    if not name:
        return None
    return {"name": name, "arguments": args}


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
                # Fallback: attempt to parse inline function-call JSON from content
                if not delta.get("tool_calls"):
                    fc = _maybe_extract_function_call_from_content(
                        delta.get("content", "")
                    )
                    if fc:
                        delta["tool_calls"] = [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": fc["name"],
                                    "arguments": _coerce_str(fc.get("arguments", "{}")),
                                },
                            }
                        ]
                        # Optionally clear content to avoid mixed signals
                        delta["content"] = ""
                # Ensure modifications are reflected
                c["delta"] = delta
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
                # Fallback: if still no tool_calls, try extracting from content
                if not msg.get("tool_calls"):
                    fc = _maybe_extract_function_call_from_content(
                        msg.get("content", "")
                    )
                    if fc:
                        msg["tool_calls"] = [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": fc["name"],
                                    "arguments": _coerce_str(fc.get("arguments", "{}")),
                                },
                            }
                        ]
            msg["content"] = ""
        # Ensure modifications are reflected
        c["message"] = msg

        return resp_dict
    except Exception:  # noqa: BLE001
        # Be conservative: if normalization fails, return original response
        return resp_dict
